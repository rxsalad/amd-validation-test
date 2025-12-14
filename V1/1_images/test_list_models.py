import os
import subprocess
from dotenv import load_dotenv
load_dotenv()

BUCKET            = os.getenv("BUCKET")
MODEL_PREFIX      = "models"                     

cmd = f'rclone lsf ds:{BUCKET}/{MODEL_PREFIX}'
print(cmd, flush=True)    
try:                                      
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    file_list = result.stdout.splitlines()
    for file in file_list:
        print(file, flush=True) 
except subprocess.CalledProcessError as e:
    print(f"The error message: {e}", flush=True)
    