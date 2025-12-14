import time
import subprocess
import json
import os
from helper import Uploader
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv()


# Megatron-LM builds the model from scratch using the layer sizes you specify


TASK_NAME   = os.getenv("TASK_NAME", "")
BUCKET      = os.getenv("BUCKET", "")
FOLDER      = os.getenv("FOLDER", "")
SUB_FOLDER  = "megatron"
NODE_NAME   = os.getenv("NODE_NAME", "") # DOKS WOKER NAME
LOCAL_LOG_FILE = "/workspace/Megatron-LM/final.log"
temp_log_file = "/workspace/Megatron-LM/megatron.log"


RESULT = {}
RESULT['node name']        = NODE_NAME
RESULT['task name']        = TASK_NAME
RESULT['type']             = "training"
RESULT['online utc']       = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")
RESULT['command']          = "MOCK_DATA=1 TEE_OUTPUT=1 MBS=5 BS=120 TP=8 TE_FP8=0 NO_TORCH_COMPILE=1 SEQ_LENGTH=4096 TOTAL_ITERS=12 bash /workspace/Megatron-LM/examples/llama/train_llama2.sh"
RESULT['state']            = "pending" # "pending", "success", "failure"
RESULT['duration_s']       = 0.0
RESULT['message']          = "" 


# Report the initial state to cloud storage
print("Report the initial state...", flush=True)
print(json.dumps(RESULT, indent=2), flush=True)
with open(LOCAL_LOG_FILE, 'w') as f: # Write the RESULT to the log file
    json.dump(RESULT, f, indent=2)
Uploader(LOCAL_LOG_FILE, BUCKET, f"{FOLDER}/{SUB_FOLDER}/{NODE_NAME}.log")    


# Start training
START = time.perf_counter()

print("Starting training...", flush=True)
env = os.environ.copy() # Set environment variables
env.update({
    "MOCK_DATA": "1",
    "TEE_OUTPUT": "1",
    "MBS": "5",
    "BS": "120",
    "TP": "8",
    "TE_FP8": "0",
    "NO_TORCH_COMPILE": "1",
    "SEQ_LENGTH": "4096",
    "TOTAL_ITERS": "12",
})
cmd = [ "bash", "examples/llama/train_llama2.sh" ]

with open(temp_log_file, "w") as f: # Open log file for both stdout and stderr
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT  # redirect stderr → same file
    )
    retcode = process.wait()

END = time.perf_counter()
RESULT['duration_s'] = round(END - START,3)

if retcode == 0:
    print("Successful!", flush=True)
    RESULT['state'] = "success"
else:
    RESULT['state'] = "failure"
    RESULT['message'] = f"The training failed with return code {retcode}"
    print(RESULT['message'], flush=True)

# Report the final results to cloud storage
print("Report the final results...", flush=True)

with open(LOCAL_LOG_FILE, 'w') as f: # Write the RESULT to the log file
    json.dump(RESULT, f, indent=2)

# GPU Driver Info
temp = subprocess.run( ["amd-smi", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
with open(LOCAL_LOG_FILE, "a") as f:
    f.write("\n\n") 
    f.write("-" * 40 + "> amd-smi version\n") 
    f.write(temp.stdout)

# GPU State Info
temp = subprocess.run( ["rocm-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
with open(LOCAL_LOG_FILE, "a") as f:
    f.write("\n\n") 
    f.write("-" * 40 + "> rocm-smi\n") 
    f.write(temp.stdout)

# GPU Product Info
temp = subprocess.run( ["rocm-smi", "--showproduct"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
with open(LOCAL_LOG_FILE, "a") as f:
    f.write("\n\n") 
    f.write("-" * 40 + "> rocm-smi --showproduct\n") 
    f.write(temp.stdout)

# Training Logs
with open(LOCAL_LOG_FILE, "a") as f1, open(temp_log_file, "r") as f2:
    f1.write("\n\n")  
    f1.write("-" * 40 + "> Training Logs\n") 
    f1.write(f2.read())

Uploader(LOCAL_LOG_FILE, BUCKET, f"{FOLDER}/{SUB_FOLDER}/{NODE_NAME}.log")

print(json.dumps(RESULT, indent=2), flush=True)
print("Exiting...", flush=True)