import argparse
import sys
import os
import time
import warnings
import torch
from rich.console import Console

# Suppress pyannote's verbose warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")

from core.audio import load_audio
from core.format import OutputFormatter
from core.platform import detect_platform, platform_summary
from core.registry import create_transcriber, list_supported_models
from core.vad import UnifiedVAD
from core.cache import VADCache
from core.benchmark import BenchmarkTracker
from core.segments import sanitize_segments

console = Console()

def run_transcription(audio, audio_duration, model_spec, vad_mode, segments, args, language=None, tracker_enabled=False):
    """Run transcription for a single model."""
    # Initialize Benchmark Tracker for this specific model
    tracker = BenchmarkTracker(model_spec, vad_mode, args.device) if tracker_enabled else None
    if tracker: tracker.set_duration(audio_duration)

    # --- Stage 2: Load Model ---
    console.print(f"[bold magenta]Loading model:[/bold magenta] {model_spec}")
    load_start = time.time()
    
    # Extract model-specific kwargs
    model_kwargs = {}
    if "voxtral" in model_spec:
        model_kwargs = {"precision": args.precision, "flash_attn": args.flash_attn, "compile_model": args.compile}
    
    try:
        transcriber = create_transcriber(model_spec, device=args.device, language=language, **model_kwargs)
        transcriber.load()
        if tracker: tracker.mark_load_done(load_start)
    except Exception as e:
        console.print(f"[bold red]Error loading model {model_spec}:[/bold red] {e}")
        return

    # --- Stage 4: Transcribe ---
    console.print(f"[bold green]Transcribing segments with {model_spec}...[/bold green]")
    trans_start = time.time()
    
    final_data = []
    last_context = None
    sampling_rate = 16000
    
    for i, seg in enumerate(segments):
        start_samp = int(seg["start"] * sampling_rate)
        end_samp = int(seg["end"] * sampling_rate)
        duration = seg["end"] - seg["start"]

        if duration < 0.3: # Skip very short glitches
            continue

        segment_audio = audio[start_samp:end_samp]
        speaker = seg.get("speaker", "SPEAKER_00")

        # Context carry (if supported and it's the same speaker)
        context = None
        if transcriber.supports_context_carry:
            if final_data and final_data[-1]["speaker"] == speaker:
                context = last_context

        # Log-friendly progress (one per line, but only the index)
        if (i + 1) % 5 == 0 or i == 0 or i == len(segments) - 1:
            console.print(f"  [{i+1}/{len(segments)}] @{seg['start']:.1f}s — Processing...")
            
        text = transcriber.transcribe_segment(segment_audio, context=context)

        if not text:
            continue

        last_context = text

        # Merge segments with same speaker ONLY if the gap is small (e.g. < 0.8s) 
        # to preserve conversation structure even in VAD-only mode.
        silence_gap = (seg["start"] - final_data[-1]["end"]) if final_data else 0
        
        if final_data and final_data[-1]["speaker"] == speaker and silence_gap < 0.8:
            final_data[-1]["end"] = round(seg["end"], 3)
            final_data[-1]["text"] += " " + str(text)
        else:
            final_data.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "speaker": speaker,
                "text": str(text)
            })
    console.print()

    if tracker: tracker.mark_transcription_done(trans_start)

    # --- Stage 5: Save & Finalize ---
    if tracker: tracker.finalize()
    
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    model_tag = model_spec.replace(":", "-").replace("/", "-")
    output_base = os.path.join(args.output_dir, f"{base_name}_{model_tag}")
    
    OutputFormatter.to_json(final_data, f"{output_base}.json")
    OutputFormatter.to_markdown(final_data, f"{output_base}.md")
    OutputFormatter.to_txt(final_data, f"{output_base}.txt")
    OutputFormatter.to_srt(final_data, f"{output_base}.srt")

    console.print(f"[bold green]Done![/bold green] Outputs saved to [cyan]{output_base}.*[/cyan]")

    if tracker:
        tracker.save()
        tracker.print_summary(console)
    
    # Clean up model to free VRAM for next model
    del transcriber
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="VoxHub: Multi-Model Transcription Platform")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    
    # Model Selection
    parser.add_argument(
        "--model", 
        default="voxtral:mini-3b",
        help=f"Model specifier, comma-separated list, or 'all'. Supported: {', '.join(list_supported_models())}"
    )
    parser.add_argument("--lang", default=None, help="Language code (e.g., 'fr', 'en', 'es', 'de') for transcription.")
    
    # VAD & Diarization
    parser.add_argument(
        "--vad",
        choices=["silero", "pyannote", "hybrid", "none"],
        default="silero",
        help=(
            "VAD strategy: silero (fast), pyannote (HQ/diarization), "
            "hybrid (Silero gate + Pyannote refiner, best recall+precision), "
            "none (no segmentation)"
        )
    )
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization (with --vad pyannote or hybrid)")
    parser.add_argument("--no-cache", action="store_true", help="Disable VAD segment caching")

    # Hybrid VAD tuning
    parser.add_argument(
        "--silero-threshold", type=float, default=0.35,
        help="Silero sensitivity for hybrid mode (lower=more sensitive, default: 0.35)"
    )
    parser.add_argument(
        "--override-threshold", type=float, default=0.8,
        help="Silero confidence above which segments are kept even if Pyannote disagrees (default: 0.8)"
    )
    
    # Segment post-processing
    parser.add_argument(
        "--no-sanitize", action="store_true",
        help="Disable segment sanitization (overlap resolution, micro-turn absorption)"
    )
    parser.add_argument(
        "--refine-boundaries", action="store_true",
        help="Use wav2vec2 to snap segment boundaries to exact speech onset/offset"
    )
    parser.add_argument(
        "--min-turn", type=float, default=1.5,
        help="Speaker turns shorter than this (seconds) are absorbed into surrounding turns (default: 1.5)"
    )

    # Performance & Device
    parser.add_argument("--device", choices=["auto", "cuda", "rocm", "cpu"], default="auto", help="Hardware device")
    parser.add_argument("--precision", default="auto", help="Model precision (Voxtral only: auto, fp16, fp8, q4, q8)")
    parser.add_argument("--flash-attn", action="store_true", help="Enable Flash Attention (Voxtral only)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (Voxtral only)")
    parser.add_argument("--benchmark", action="store_true", help="Enable benchmark mode (logs performance data)")
    
    # Metadata hints
    parser.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum number of speakers")
    parser.add_argument("--hf-token", help="HF token for Pyannote/Canary")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        console.print(f"[bold red]Error:[/bold red] Input file {args.input} not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse model(s)
    if args.model.lower() == "all":
        models_to_run = list_supported_models(enabled_only=True)
    else:
        models_to_run = [m.strip() for m in args.model.split(",")]
    
    console.print(f"[bold yellow]Models to run:[/bold yellow] {', '.join(models_to_run)}")

    # --- Stage 1: Load audio ---
    console.print(f"[bold blue]Loading audio:[/bold blue] {args.input}")
    audio = load_audio(args.input)
    audio_duration = len(audio) / 16000
    console.print(f"  Audio duration: {audio_duration:.1f}s")

    # --- Stage 3: VAD / Segmentation (Shared across all models) ---
    console.print(f"[bold cyan]Running VAD ({args.vad})...[/bold cyan]")
    vad_start = time.time()
    
    segments = None
    cache = VADCache()
    vad_params = {
        "mode": args.vad, "diarize": args.diarize, 
        "num_speakers": args.num_speakers, "min_speakers": args.min_speakers, "max_speakers": args.max_speakers
    }
    
    if not args.no_cache:
        segments = cache.load(audio.tobytes()[:1000000], vad_params)
        if segments:
            console.print("  [green]Using cached VAD segments[/green]")

    if not segments:
        vad_engine = UnifiedVAD(
            mode=args.vad, hf_token=args.hf_token,
            silero_threshold=args.silero_threshold,
            override_threshold=args.override_threshold,
        )
        segments = vad_engine.detect(
            audio, diarize=args.diarize, 
            num_speakers=args.num_speakers, min_speakers=args.min_speakers, max_speakers=args.max_speakers
        )
        if not args.no_cache:
            cache.save(audio.tobytes()[:1000000], vad_params, segments)

    console.print(f"  {len(segments)} segments found")

    # --- Stage 3b: Segment post-processing ---
    if not args.no_sanitize and len(segments) > 1:
        console.print(f"[bold cyan]Sanitizing segments (min_turn={args.min_turn}s)...[/bold cyan]")
        segments = sanitize_segments(segments, min_turn_duration=args.min_turn)
        console.print(f"  {len(segments)} segments after sanitization")

    if args.refine_boundaries:
        console.print("[bold cyan]Refining boundaries with wav2vec2...[/bold cyan]")
        from core.segments import BoundaryRefiner
        refiner = BoundaryRefiner(device=args.device)
        segments = refiner.refine_boundaries(audio, segments)
        console.print("  Boundaries refined")

    # --- Stage 4/5: Process each model ---
    for model_spec in models_to_run:
        run_transcription(audio, audio_duration, model_spec, args.vad, segments, args, language=args.lang, tracker_enabled=args.benchmark)

if __name__ == "__main__":
    main()
