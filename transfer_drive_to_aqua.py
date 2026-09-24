import os
import sys
import subprocess
import argparse

# Configuration
DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/1bzwadTmTkp69kNkbdPNb-2tm7DKAvd7Q?usp=sharing"
AQUA_HOST = "aqua.iitm.ac.in"
AQUA_PORT = "40826"
AQUA_USER = "na22b025"
AQUA_SCRATCH_DEST = "/scratch/na22b025/Face_Dataset"
LOCAL_TEMP_DIR = os.path.expanduser("~/Downloads/Face_Dataset_Temp")

def run_cmd(cmd, shell=True):
    print(f"[EXEC] {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    subprocess.run(cmd, shell=shell, check=True)

def step1_download_google_drive(local_dir):
    """
    Step 1: Download Google Drive folder locally to local_dir using gdown.
    """
    print(f"\n========================================================")
    print(f"Step 1: Downloading Google Drive 'Face_Dataset' locally...")
    print(f"Target Local Path: {local_dir}")
    print(f"========================================================\n")
    
    os.makedirs(local_dir, exist_ok=True)
    
    # Verify/Install gdown
    try:
        import gdown
    except ImportError:
        print("Installing gdown Python package...")
        run_cmd(f"{sys.executable} -m pip install gdown")
        
    cmd = f"{sys.executable} -m gdown --folder '{DRIVE_FOLDER_LINK}' -O '{local_dir}' --remaining-ok"
    run_cmd(cmd)

def step2_upload_to_aqua_scratch(local_dir, scratch_dest):
    """
    Step 2: Upload local dataset folder to AQUA cluster /scratch directory via SSH rsync.
    """
    print(f"\n========================================================")
    print(f"Step 2: Uploading Local Dataset to AQUA Cluster /scratch")
    print(f"Destination: {AQUA_USER}@{AQUA_HOST}:{scratch_dest}")
    print(f"========================================================\n")
    
    # Create remote scratch directory over SSH
    mkdir_cmd = f"ssh -p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} 'mkdir -p {scratch_dest}'"
    run_cmd(mkdir_cmd)
    
    # High-speed rsync upload over SSH port 40826 with resume capability
    rsync_cmd = (
        f"rsync -avzP --partial "
        f"-e 'ssh -p {AQUA_PORT}' "
        f"'{local_dir}/' "
        f"{AQUA_USER}@{AQUA_HOST}:'{scratch_dest}/'"
    )
    run_cmd(rsync_cmd)
    
    print(f"\n========================================================")
    print(f"✓ Upload Complete! Dataset successfully transferred to:")
    print(f"  AQUA Scratch: {AQUA_USER}@{AQUA_HOST}:{scratch_dest}")
    print(f"========================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Local Script to Download Google Drive Face_Dataset and Upload to AQUA Cluster /scratch Space")
    parser.add_argument("--local_dir", type=str, default=LOCAL_TEMP_DIR, help="Local temporary directory for Google Drive download")
    parser.add_argument("--scratch_dest", type=str, default=AQUA_SCRATCH_DEST, help="Remote /scratch path on AQUA cluster")
    parser.add_argument("--skip_download", action="store_true", help="Skip Google Drive download and proceed straight to upload")
    args = parser.parse_args()
    
    if not args.skip_download:
        step1_download_google_drive(args.local_dir)
        
    step2_upload_to_aqua_scratch(args.local_dir, args.scratch_dest)

if __name__ == "__main__":
    main()
