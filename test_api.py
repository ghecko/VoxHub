import requests
import os
import sys
import time

def test_transcription(file_path, model="whisper:turbo", api_url="http://localhost:8000"):
    print(f"--- Testing Transcription with {model} ---")
    
    url = f"{api_url}/v1/audio/transcriptions"
    
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "model": model,
            "response_format": "verbose_json",
        }
        
        # Add API Key if set
        headers = {}
        api_key = os.getenv("VOXBENCH_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        try:
            print("Transcribing, please wait (this may take a minute for large files)...")
            start_time = time.time()
            response = requests.post(url, files=files, data=data, headers=headers)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                print(f"Success! (Duration: {duration:.2f}s)")
                result = response.json()
                print(f"Language: {result.get('language')}")
                print(f"Full Text sample: {result.get('text')[:200]}...")
                print(f"Segment count: {len(result.get('segments', []))}")
                
                if result.get('segments'):
                    first_seg = result['segments'][0]
                    print(f"First segment: [{first_seg.get('start')} - {first_seg.get('end')}] Speaker: {first_seg.get('speaker')}")
            else:
                print(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Fail: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_audio>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    test_transcription(audio_file)
