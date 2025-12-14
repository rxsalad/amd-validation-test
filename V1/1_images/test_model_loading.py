import os
import sys
import json
import shutil
from helper import Downloader, Uploader, Check_Cloud_Folder, Check_Local_Folder, Get_Folder_Size
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv()

# environment variables: TASK_NAME, NODE_NAME, BUCKET, FOLDER, MODEL, MODEL_FOLDER, OVERRIDE

TASK_NAME         = os.getenv("TASK_NAME", "test-model-loading-2025")
NODE_NAME         = os.getenv("NODE_NAME", "test-node")
LOCAL_LOG_FILE    = "./final.log"

# Unique folder for each validation test, ds:{BUCKET}/{FOLDER}, and save the test results (megatron & llama) and the benchmark results 
# ds:{BUCKET}/{FOLDER}/megatron
# ds:{BUCKET}/{FOLDER}/llama
# ds:{BUCKET}/{FOLDER}/benchmark/model_loading
BUCKET            = os.getenv("BUCKET", "rs-validation-test")
FOLDER            = os.getenv("FOLDER", "test200251212")
SUB_FOLDER        = "benchmark/model_loading"      # Hardcoded at this time

# The model should be pre-downloaded and saved here: ds:{BUCKET}/models/{MODEL_FOLDER}, and then loaded to "/root/.cache/huggingface/hub/{MODEL_FOLDER}" by test workloads
MODEL             = os.getenv("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
MODEL_PREFIX      = "models"                       # Hardcoded at this time
MODEL_FOLDER      = os.getenv("MODEL_FOLDER", "models--meta-llama--Llama-3.1-8B-Instruct")
HF_CACHE_FOLDER   = "/root/.cache/huggingface/hub" # Hardcoded at this time
LOCAL_PATH        = os.path.join(HF_CACHE_FOLDER, MODEL_FOLDER)

OVERRIDE          = int(os.getenv("OVERRIDE", "1"))       # Optional, hardcoded at this time

# To keep benchamrk results
RESULT = {}
RESULT['task name']           = TASK_NAME
RESULT['node name']           = NODE_NAME
RESULT['type']                = "model_loading"
RESULT['online utc']          = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S")
RESULT['state']               = "pending" # "pending", "success", "failure"
RESULT['duration_s']          = 9999.9999
RESULT['data size_GB']        = 0
RESULT['dl_throughput_Gbps']  = 0
RESULT['message']             = "" 
START = time.perf_counter()

# Test Case
# BUCKET            = "rs-validation-test1"
# HF_CACHE_FOLDER    = "/root/.cache/huggingface/hub1" 
# MODEL_FOLDER      = "models--meta-llama--Llama-3.1-8B-Instruct11"

if RESULT['state'] == "pending":
    # We don't check the integrity of the model files at this time
    temp = Check_Cloud_Folder(BUCKET, f'{MODEL_PREFIX}/{MODEL_FOLDER}') 
    if temp == 0:
        RESULT['state'] = "failure"
        RESULT['message'] = f"The model folder ds:{BUCKET}/{MODEL_PREFIX}/{MODEL_FOLDER} does not exist or is empty!"  
        print(RESULT['message'], flush=True)

if RESULT['state'] == "pending":
    # We don't check the integrity of the model files at this time
    temp = Check_Local_Folder(LOCAL_PATH)
    if temp != 0 and OVERRIDE == 0:
        RESULT['state'] = "success"
        RESULT['message'] = f"The local model folder {HF_CACHE_FOLDER}/{MODEL_FOLDER} already exists!"
        print(RESULT['message'], flush=True)

if RESULT['state'] == "pending":
    if os.path.exists(LOCAL_PATH) and os.path.isdir(LOCAL_PATH):
        shutil.rmtree(LOCAL_PATH)
    temp = Downloader(BUCKET, f'{MODEL_PREFIX}/{MODEL_FOLDER}', LOCAL_PATH, chunk_size_mbype="100M", concurrency="10")  
    if temp == 0:
        RESULT['state'] = "failure"
        RESULT['message'] = f"Failed to download the model folder ds:{BUCKET}/{MODEL_PREFIX}/{MODEL_FOLDER} to local path {LOCAL_PATH}!"
        print(RESULT['message'], flush=True)

if RESULT['state'] == "pending": 
    temp = Check_Local_Folder(LOCAL_PATH)
    if temp == 0:
        RESULT['state'] = "failure"
        RESULT['message'] = f"Failed to download the model folder ds:{BUCKET}/{MODEL_PREFIX}/{MODEL_FOLDER} to local path {LOCAL_PATH}!"
        print(RESULT['message'], flush=True)
    else: 
        RESULT['state'] = "success"
        RESULT['message'] = f"Successfully downloaded the model to local path {LOCAL_PATH}!"
        print(RESULT['message'], flush=True)    

END = time.perf_counter()
RESULT['duration_s'] = round(END - START,3)

if RESULT['state'] == "success":
    RESULT['data size_GB']  = round(Get_Folder_Size(LOCAL_PATH)/1_000_000_000, 3)  # GB 
    RESULT['dl_throughput_Gbps']  = round(RESULT['data size_GB'] * 8/RESULT['duration_s'], 3)  # Gbps 

with open(LOCAL_LOG_FILE, 'w') as f: # Write the RESULT to the log file
    json.dump(RESULT, f, indent=2)
Uploader(LOCAL_LOG_FILE, BUCKET, f"{FOLDER}/{SUB_FOLDER}/{NODE_NAME}.log")

print(json.dumps(RESULT, indent=2), flush=True)

print(f"Exiting with exit code {1 if RESULT['state'] == 'failure' else 0}...", flush=True)

if RESULT['state'] == "failure":
    sys.exit(1)
else:
    sys.exit(0)