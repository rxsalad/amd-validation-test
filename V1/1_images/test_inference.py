import time
import subprocess
import json
import os
import requests
import threading
import sys
from helper import Uploader
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv()


# The downloaded models will be saved in the host
# The pod's /root/.cache/huggingface -> the host's /root/.cache/huggingface


URL     = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL   = os.getenv("MODEL", "")


INPUT_PROMPT = "Who are you? Please tell me how to learn AI and ML, using 1000+ words"
PAYLOAD = { "model": MODEL, "messages": [{"role": "user", "content": INPUT_PROMPT}] }


TASK_NAME   = os.getenv("TASK_NAME", "")
BUCKET      = os.getenv("BUCKET", "")
FOLDER      = os.getenv("FOLDER", "")
SUB_FOLDER  = "llama"
NODE_NAME = os.getenv("NODE_NAME", "") # DOKS WOKER NAME
LOCAL_LOG_FILE = "/app/final.log"
temp_log_file = "/app/vllm_server.log"


RESULT = {}
RESULT['node name']              = NODE_NAME
RESULT['task name']              = TASK_NAME
RESULT['type']                   = "inference"
RESULT['online utc']             = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")
RESULT['model']                  = MODEL
RESULT['inference server']       = "vllm 0.11.1"
RESULT['state']                  = "pending"  # "pending", "running", "restarted"
RESULT['message']                = "" 
RESULT['startup time_s']         = 9999.9999 
RESULT['running time_s']        = 0.0
RESULT['inference number']       = 0
RESULT['generated token number'] = 0


# Report the initial state to cloud storage
print("Report the initial state...", flush=True)
print(json.dumps(RESULT, indent=2), flush=True)
with open(LOCAL_LOG_FILE, 'w') as f: # Write the RESULT to the log file
    json.dump(RESULT, f, indent=2)
Uploader(LOCAL_LOG_FILE, BUCKET, f"{FOLDER}/{SUB_FOLDER}/{NODE_NAME}.log")    


# Start the vLLM inference server
START = time.perf_counter()

print("Starting the vllm inference server...", flush=True)
env = os.environ.copy()
env["VLLM_USE_V1"] = "1"
cmd = [
    "vllm", "serve", MODEL,
    "--enforce-eager",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--tensor-parallel-size", "8",
    "--seed", "1024",
    "--dtype", "float16",
    "--max-model-len", "10000",
    "--max-num-batched-tokens", "10000",
    "--max-num-seqs", "256",
    "--trust-remote-code",
    "--gpu-memory-utilization", "0.9"
]

with open(temp_log_file, "w") as f: # Open log file for both stdout and stderr
    process = subprocess.Popen(
        cmd, 
        env=env, 
        stdout=f, 
        stderr=subprocess.STDOUT)  # redirect stderr → same file
    # Run command in background
print(f"vLLM server started with PID {process.pid}")


# Health Check
HEALTH_ENDPOINT = f"http://0.0.0.0:8000/health"  # some vLLM endpoints may have /health
RETRY_INTERVAL = 5  # seconds
MAX_RETRIES = 180   # try for up to 15 minutes

def check_vllm_ready():
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=2)
            if response.status_code == 200:
                print("vLLM server is ready!", flush=True)
                return True
        except requests.RequestException:
            pass  # server not ready yet

        print(f"Attempt {attempt}: vLLM server not ready, retrying in {RETRY_INTERVAL}s...", flush=True)
        time.sleep(RETRY_INTERVAL)

    print("vLLM server did not become ready within the timeout period.", flush=True)
    return False

if not check_vllm_ready(): # Wait until vLLM server is ready.
    RESULT['state'] = "restarted" 
    RESULT['message'] = "Cannot start in 15 minutes" 
else:
    END = time.perf_counter()
    RESULT['state'] = "running"
    RESULT['startup time_s'] = round(END - START,3)

# Report the running state to cloud storage
print("Report the running state...", flush=True)
print(json.dumps(RESULT, indent=2), flush=True)
with open(LOCAL_LOG_FILE, 'w') as f: # Write the RESULT to the log file
    json.dump(RESULT, f, indent=2)
Uploader(LOCAL_LOG_FILE, BUCKET, f"{FOLDER}/{SUB_FOLDER}/{NODE_NAME}.log")  

# Many instances running concurrently may generate too many model download requests, which could be throttled by Hugging Face.
# If the vLLM server fails due to model download issues, restart the pod after some time (15 minutes).
# Run 10 instances every 10 minutes (for Llama 3 8B).
# Improvement: pre-load the model to the host first.
if RESULT['state'] != "running":
    print("vLLM not ready in 15 minutes. Exiting to trigger container restart...", flush=True)
    sys.exit(1)  # Exit with non-zero code


# Inference function
def run_inference():
    
    while True:
        try:
            response = requests.post(URL, json=PAYLOAD, headers=HEADERS, timeout=600)

            if response.status_code == 200:
                data = response.json()
                tokens = data.get('usage', {}).get('total_tokens', 'N/A')
                print(f"Success, tokens returned: {tokens}", flush=True)
            else:
                print(f"Failed with status code {response.status_code}", flush=True)

        except requests.RequestException as e:
            print(f"Error: {e}", flush=True)

        END = time.perf_counter()
        RESULT['running time_s'] = round(END - START,3)
        RESULT['inference number'] = RESULT['inference number'] + 1
        RESULT['generated token number'] = RESULT['generated token number'] + tokens if tokens != 'N/A' else 0

        time.sleep(2)


# Create and start a thread
inference_thread = threading.Thread(target=run_inference, daemon=True)
inference_thread.start()

# Main thread can continue doing other things
print("Inference thread started and running in the background...")

# Upload log file every X minutes
while True:

    time.sleep(120)

    with open(LOCAL_LOG_FILE, 'w') as f: # Write the RESULT to the log file
        json.dump(RESULT, f, indent=2)

    # GPU State Info
    temp = subprocess.run( ["rocm-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
    with open(LOCAL_LOG_FILE, "a") as f:
        f.write("\n\n") 
        f.write("-" * 40 + "> rocm-smi\n") 
        f.write(temp.stdout)    

    # GPU Driver Info
    temp = subprocess.run( ["amd-smi", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
    with open(LOCAL_LOG_FILE, "a") as f:
        f.write("\n\n") 
        f.write("-" * 40 + "> amd-smi version\n") 
        f.write(temp.stdout)

    # GPU Product Info
    temp = subprocess.run( ["rocm-smi", "--showproduct"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
    with open(LOCAL_LOG_FILE, "a") as f:
        f.write("\n\n") 
        f.write("-" * 40 + "> rocm-smi --showproduct\n") 
        f.write(temp.stdout)

    # Inference Logs
    with open(LOCAL_LOG_FILE, "a") as f1, open(temp_log_file, "r") as f2:
        f1.write("\n\n")  
        f1.write("-" * 40 + "> Inference Logs\n") 
        f1.write(f2.read())

    Uploader(LOCAL_LOG_FILE, BUCKET, f"{FOLDER}/{SUB_FOLDER}/{NODE_NAME}.log")

    print("Running...", flush=True)
    print(json.dumps(RESULT, indent=2), flush=True)
