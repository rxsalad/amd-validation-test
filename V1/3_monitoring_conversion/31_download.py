import subprocess
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv()

# Unique folder for each validation test, ds:{BUCKET}/{FOLDER}, and save the test results (megatron & llama) and the benchmark results 
# ds:{BUCKET}/{FOLDER}/megatron
# ds:{BUCKET}/{FOLDER}/llama
# ds:{BUCKET}/{FOLDER}/benchmark/model_loading
BUCKET = os.getenv("BUCKET")
FOLDER = os.getenv("FOLDER")

# Ensure the local folder exists
os.makedirs(FOLDER, exist_ok=True)

# Build the command
cmd = ["rclone", "copy", f"ds:{BUCKET}/{FOLDER}", FOLDER]
print(" ".join(cmd), flush=True)    

# Run the command
try:
    subprocess.run(cmd, check=True)
    print(f"Copied ds:{BUCKET}/{FOLDER} -> {FOLDER} successfully")
except subprocess.CalledProcessError as e:
    print(f"Error copying files: {e}")


