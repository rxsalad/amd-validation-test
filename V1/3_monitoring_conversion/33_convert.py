import os
import shutil
from dotenv import load_dotenv
load_dotenv()

# Unique folder for each validation test, ds:{BUCKET}/{FOLDER}, and save the test results (megatron & llama) and the benchmark results 
# ds:{BUCKET}/{FOLDER}/megatron
# ds:{BUCKET}/{FOLDER}/llama
# ds:{BUCKET}/{FOLDER}/benchmark/model_loading
FOLDER = os.getenv("FOLDER")

mapping = {}

mapping_file = "mapping.txt"

removed_values = ["atl1node59999", "atl1node59998", "atl1node59997"] # Exceptions

OUTPUT_FOLDER = FOLDER + "_converted"

SUBFOLDE_MEGATRON                = "megatron"
SUBFOLDE_BENCHMARK_MODEL_LOADING = "benchmark/model_loading"
SUBFOLDE_LLAMA                   = "llama"

# Ensure the local folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, SUBFOLDE_MEGATRON), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, SUBFOLDE_BENCHMARK_MODEL_LOADING), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, SUBFOLDE_LLAMA), exist_ok=True)

# Generate the mapping dictionary using both the mapping file and exceptions
with open(mapping_file, "r") as f:
    for line in f:
        # Remove leading/trailing whitespace and split by '|'
        parts = [p.strip() for p in line.strip().split("|")]
        if len(parts) == 5:
            _, _, key, value, _ = parts
            mapping[key] = value
print("\n----> The number of mappings:", len(mapping))

#for k, v in mapping.items():
#    print(k, "=>", v)

def convert_file(source, destinaion, subfolder, mapping):
    source_path = os.path.join(source, subfolder)
    destination_path = os.path.join(destinaion, subfolder)

    log_files = [f for f in os.listdir(source_path) if f.endswith(".log")]
    for log_file in log_files:
        base_name = log_file[:-4]  # Extract the base name without ".log"
        if base_name in mapping:
            print(f"{log_file} → Found in mapping: {mapping[base_name]}", end="")

            if mapping[base_name] in removed_values:
                print(f", which is in the removed values. Skipping...")
            else:
                converted_log_file = os.path.join(destination_path, mapping[base_name] + ".log")
                shutil.copyfile(os.path.join(source_path, log_file), converted_log_file)
                print()
        else:
            print(f"{log_file} → NOT found in mapping. Skipping...")


print("\n----> Convert the megatron log files...")
convert_file(FOLDER, OUTPUT_FOLDER, SUBFOLDE_MEGATRON, mapping)

print("\n----> Convert the model loading log files...")
convert_file(FOLDER, OUTPUT_FOLDER, SUBFOLDE_BENCHMARK_MODEL_LOADING, mapping)

print("\n----> Convert the llama log files...")
convert_file(FOLDER, OUTPUT_FOLDER, SUBFOLDE_LLAMA, mapping)

