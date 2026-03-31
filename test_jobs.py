import requests
import os
import sys
import time

def test_jobs_api(file_path, model="whisper:turbo", api_url="http://localhost:8000"):
    print(f"--- Testing Asynchronous Jobs API with {model} ---")
    
    # 1. Submit Job
    submit_url = f"{api_url}/v1/audio/transcriptions/jobs"
    
    headers = {}
    api_key = os.getenv("VOXBENCH_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "model": model,
            "response_format": "verbose_json",
        }
        
        print("Submitting job...")
        response = requests.post(submit_url, files=files, data=data, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to submit: {response.text}")
            return
            
        job_data = response.json()
        job_id = job_data["job_id"]
        status_url = f"{api_url}{job_data['links']['status']}"
        result_url = f"{api_url}{job_data['links']['result']}"
        
        print(f"Job created: {job_id}")
        
    # 2. Poll Status
    print("Polling for status...")
    while True:
        status_resp = requests.get(status_url, headers=headers)
        if status_resp.status_code != 200:
            print(f"Error polling: {status_resp.text}")
            break
            
        status_info = status_resp.json()
        status = status_info["status"]
        progress = status_info["progress"]
        
        print(f"Status: {status} | Progress: {progress}%")
        
        if status == "completed":
            print("Job completed successfully!")
            break
        elif status == "failed":
            print(f"Job failed: {status_info.get('error')}")
            return
            
        time.sleep(2)
        
    # 3. Get Result
    print("Retrieving result...")
    result_resp = requests.get(result_url, headers=headers)
    if result_resp.status_code == 200:
        result = result_resp.json()
        print(f"Success! Full Text sample: {result.get('text')[:200]}...")
        print(f"Total segments: {len(result.get('segments', []))}")
    else:
        print(f"Failed to get result: {result_resp.text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_jobs.py <path_to_audio>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    test_jobs_api(audio_file)
