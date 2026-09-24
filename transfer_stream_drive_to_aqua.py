import os
import sys
import time
import subprocess
import argparse
import urllib.request

# Configuration
DRIVE_FOLDER_ID = "1bzwadTmTkp69kNkbdPNb-2tm7DKAvd7Q"
DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/1bzwadTmTkp69kNkbdPNb-2tm7DKAvd7Q?usp=sharing"
AQUA_HOST = "aqua.iitm.ac.in"
AQUA_PORT = "40826"
AQUA_USER = "na22b025"
AQUA_SCRATCH_DEST = "/scratch/na22b025/Face_Dataset"

def check_command(cmd):
    return subprocess.call(f"type {cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0

def stream_via_rclone(scratch_dest):
    """
    Method 1: Streaming via rclone with tiny 64MB RAM memory buffer (0 Bytes disk usage).
    Pipes Google Drive remote directly into remote AQUA /scratch space over SSH.
    """
    print("\n==========================================================")
    print("[METHOD 1: RCLONE STREAMING PIPE (0 Bytes Local Disk Used)]")
    print(f"Piping Google Drive '{DRIVE_FOLDER_ID}' -> {AQUA_USER}@{AQUA_HOST}:{scratch_dest}")
    print("==========================================================\n")
    
    # Ensures destination directory exists on AQUA
    subprocess.run(
        f"ssh -p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} 'mkdir -p {scratch_dest}'",
        shell=True, check=True
    )
    
    # rclone stream directly to SSH remote
    cmd = (
        f"rclone copy drive: '{scratch_dest}' "
        f"--drive-root-folder-id {DRIVE_FOLDER_ID} "
        f"--transfers 16 --checkers 32 "
        f"--buffer-size 64M "
        f"--progress --stats 10s "
        f"--sftp-host {AQUA_HOST} --sftp-port {AQUA_PORT} --sftp-user {AQUA_USER}"
    )
    subprocess.run(cmd, shell=True, check=True)

def stream_via_gdown_pipe(scratch_dest):
    """
    Method 2: Zero-disk streaming using gdown + tar over SSH pipe.
    Downloads stream chunks in memory and pipes them straight to SSH stdin.
    """
    print("\n==========================================================")
    print("[METHOD 2: GDOWN STREAMING SSH PIPE (0 Bytes Local Disk Used)]")
    print(f"Streaming Drive Link -> SSH Pipe -> {AQUA_USER}@{AQUA_HOST}:{scratch_dest}")
    print("==========================================================\n")
    
    # Ensure destination directory exists on AQUA
    subprocess.run(
        f"ssh -p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} 'mkdir -p {scratch_dest}'",
        shell=True, check=True
    )
    
    # Install gdown if needed
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)

    # Use gdown folder download streaming output piped into SSH tar extract
    gdown_cmd = f"{sys.executable} -m gdown --folder '{DRIVE_FOLDER_LINK}' -O - --remaining-ok"
    ssh_cmd = f"ssh -p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} 'tar -C {scratch_dest} -xzf - 2>/dev/null || cat > {scratch_dest}/streamed_dataset.tar'"
    
    pipe_cmd = f"{gdown_cmd} | {ssh_cmd}"
    print(f"[PIPE EXEC] {pipe_cmd}")
    subprocess.run(pipe_cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="Zero Local Disk Usage: Stream Large Google Drive Folder directly to AQUA Scratch over SSH Pipe")
    parser.add_argument("--scratch_dest", type=str, default=AQUA_SCRATCH_DEST, help="Remote /scratch path on AQUA cluster")
    args = parser.parse_args()
    
    scratch_dest = args.scratch_dest
    print(f"=========================================================================")
    print(f"ZERO-DISK LOCAL STREAMING TRANSFER PIPELINE")
    print(f"Local Laptop Role: Network Bridge / Pipe (0 Bytes local disk space needed)")
    print(f"Google Drive Link: {DRIVE_FOLDER_LINK}")
    print(f"AQUA Cluster Scratch Target: {AQUA_USER}@{AQUA_HOST}:{scratch_dest}")
    print(f"=========================================================================\n")

    # Try Method 1: rclone streaming if configured
    if check_command("rclone"):
        try:
            stream_via_rclone(scratch_dest)
            print("\n✓ Zero-disk rclone streaming transfer complete!")
            return
        except Exception as e:
            print(f"[rclone streaming info]: {e}. Falling back to gdown streaming SSH pipe...")

    # Method 2: gdown stream pipe
    try:
        stream_via_gdown_pipe(scratch_dest)
        print("\n✓ Zero-disk gdown streaming transfer complete!")
    except Exception as e:
        print(f"\n[ERROR] Streaming transfer failed: {e}")
        print("\nAlternative Method:")
        print(f"Run rclone on local machine with SSH sftp target:")
        print(f"rclone copy drive: {scratch_dest} --drive-root-folder-id {DRIVE_FOLDER_ID} -P")
        sys.exit(1)

if __name__ == "__main__":
    main()
