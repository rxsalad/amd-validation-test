import os
import subprocess
import time
from dotenv import load_dotenv
load_dotenv()

# Access to DO Spaces
SPACES_URL    = os.getenv("AWS_ENDPOINT_URL", "")
SPACES_REGION = os.getenv("AWS_REGION", "")
SPACES_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
SPACES_KEY    = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# Create the configuration file for rclone
# https://developers.cloudflare.com/r2/examples/rclone/
filename = os.path.expanduser("~")+"/.config/rclone/rclone.conf"
with open(filename,'w') as f:
    f.write("[ds]\n")
    f.write("type = s3\n")
    f.write("provider = DigitalOcean\n")
    f.write("access_key_id = {}\n".format(SPACES_ID))
    f.write("secret_access_key = {}\n".format(SPACES_KEY))
    f.write("region = {}\n".format(SPACES_REGION))
    f.write("endpoint = {}\n".format(SPACES_URL))
    f.write("bucket_acl = private")

# upload_local_to_cloud, a file or directory
# https://rclone.org/commands/rclone_copyto/
def Uploader(local, bucket, key, chunk_size_mbype="10M", concurrency="10"):    
    cmd = f'rclone copyto {local} ds:{bucket}/{key} --s3-chunk-size={chunk_size_mbype} --transfers={concurrency} --ignore-times'
    print(cmd, flush=True)
    try:                                      
        subprocess.run(cmd, shell=True, check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"The error message: {e}", flush=True)
        return 0
    return 1

# download_cloud_to_local, a file or directory
# https://rclone.org/commands/rclone_copyto/
def Downloader(bucket, key, local, chunk_size_mbype="10M", concurrency="10"):    
    cmd = f'rclone copyto ds:{bucket}/{key} {local} --s3-chunk-size={chunk_size_mbype} --transfers={concurrency} --ignore-times'
    print(cmd, flush=True)
    try:                                      
        subprocess.run(cmd, shell=True, check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"The error message: {e}", flush=True)
        return 0
    return 1

# Check the folder in Cloud
def Check_Cloud_Folder(bucket, model_folder):
    cmd = f'rclone ls ds:{bucket}/{model_folder}'
    print(cmd, flush=True)    
    try:                                      
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        file_list = result.stdout.splitlines()
        print(f"The number of files in {bucket}/{model_folder}: {len(file_list)}", flush=True)
        return len(file_list) # 0 or more
    except subprocess.CalledProcessError as e:
        print(f"The error message: {e}", flush=True)
        return 0
    
# Check the local folder
def Check_Local_Folder(local_path):
    try:
        #file_list = os.listdir(local_path)
        file_list = []
        for root, dirs, files in os.walk(local_path):
            for file in files:
                file_list.append(os.path.join(root, file))
        print(f"The number of files in {local_path}: {len(file_list)}", flush=True)
        return len(file_list) # 0 or more
    except Exception as e:
        print(f"The error message: {e}", flush=True)
        return 0
    
# For the download/upload throughput calculation
def Get_Folder_Size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):  # Make sure it's a file
                total_size += os.path.getsize(fp)
    return total_size
