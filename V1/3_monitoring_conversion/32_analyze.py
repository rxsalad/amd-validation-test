import json
import os
from dotenv import load_dotenv
load_dotenv()

# Unique folder for each validation test, ds:{BUCKET}/{FOLDER}, and save the test results (megatron & llama) and the benchmark results 
# ds:{BUCKET}/{FOLDER}/megatron
# ds:{BUCKET}/{FOLDER}/llama
# ds:{BUCKET}/{FOLDER}/benchmark/model_loading
FOLDER = os.getenv("FOLDER")

SUBFOLDE_MEGATRON                = "megatron"
SUBFOLDE_BENCHMARK_MODEL_LOADING = "benchmark/model_loading"
SUBFOLDE_LLAMA                   = "llama"

def count_files(path):
    if not os.path.exists(path):
        return 0

    total = 0
    for root, dirs, files in os.walk(path):
        total += len(files)
    return total


def analyze_logs(folder, subfolder):
    path = os.path.join(folder, subfolder)
    count = count_files(path)
    print(f"Files in {path}: {count}")
    if count == 0:
        return
    
    log_files = [f for f in os.listdir(path) if f.endswith(".log")]
    for log_file in log_files:
        json_lines = []
        with open(os.path.join(path, log_file)) as f:
            for line in f:
                if line.strip() == "":  
                    break
                json_lines.append(line)
        json_text = "".join(json_lines)
        data = json.loads(json_text)
        if data['state'] != "running" and data['state'] !="success":
            print('Attention: ', data['node name'], data['state'], data['message'])
        else:
            if data["type"] == "training":
                print(f"Node: {data['node name']}, Training Time: {data['duration_s']} seconds")
            elif data["type"] == "model_loading":
                print(f"Node: {data['node name']}, Data Size: {data['data size_GB']} GB, Duration: {data['duration_s']} seconds, Throughput: {data['dl_throughput_Gbps']} Gbps")
            elif data["type"] == "inference":
                print(f"Node: {data['node name']}, Startup Time: {data['startup time_s']} seconds, Running Time: {data['running time_s']} seconds, Inference Number: {data['inference number']}, Generated Token Number: {data['generated token number']}")
            else: # Others
                pass


print("\n---------> Analyze the megatron log files...")
analyze_logs(FOLDER, SUBFOLDE_MEGATRON)

print("\n---------> Analyze the model loading log files...")
analyze_logs(FOLDER, SUBFOLDE_BENCHMARK_MODEL_LOADING)

print("\n---------> Analyze the llama log files...")
analyze_logs(FOLDER, SUBFOLDE_LLAMA)






