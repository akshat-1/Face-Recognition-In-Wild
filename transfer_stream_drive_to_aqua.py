import os
import sys
import time
import datetime
import subprocess
import argparse

# Configuration
DRIVE_FOLDER_ID = "1bzwadTmTkp69kNkbdPNb-2tm7DKAvd7Q"
DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/1bzwadTmTkp69kNkbdPNb-2tm7DKAvd7Q?usp=sharing"
AQUA_HOST = "aqua.iitm.ac.in"
AQUA_PORT = "40826"
AQUA_USER = "na22b025"
AQUA_SCRATCH_DEST = "Face_Dataset" # ~/scratch/Face_Dataset

def log(msg, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)

def find_ssh_key():
    for key_name in ["id_ed25519", "id_rsa", "id_dsa"]:
        key_path = os.path.expanduser(f"~/.ssh/{key_name}")
        if os.path.exists(key_path):
            return key_path
    return os.path.expanduser("~/.ssh/id_ed25519")

def run_command_with_live_logs(cmd, label="EXEC"):
    log(f"Starting subprocess [{label}]: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        shell=isinstance(cmd, str),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"  └─ {line.strip()}", flush=True)
            
    process.stdout.close()
    return_code = process.wait()
    elapsed = time.time() - start_time
    
    if return_code == 0:
        log(f"Subprocess [{label}] completed successfully in {elapsed:.2f} seconds (Exit Code 0).")
    else:
        log(f"Subprocess [{label}] failed after {elapsed:.2f} seconds with Exit Code {return_code}.", level="ERROR")
        raise subprocess.CalledProcessError(return_code, cmd)

def stream_high_speed_rclone(dest_folder, num_transfers=32, num_checkers=64):
    """
    High-Speed Multi-Threaded rclone Stream Engine (0 Bytes Local Disk Space).
    Streams 32 concurrent SFTP channels directly from Google Drive into AQUA ~/scratch/Face_Dataset.
    """
    ssh_key = find_ssh_key()
    sftp_remote = f":sftp,host={AQUA_HOST},port={AQUA_PORT},user={AQUA_USER},key_file={ssh_key},no_auth_agent=true:scratch/{dest_folder}"
    
    log("Initializing High-Speed Multi-Threaded rclone SFTP Stream Engine")
    log(f"Target AQUA Scratch Destination: {AQUA_USER}@{AQUA_HOST}:~/scratch/{dest_folder}")
    log(f"SSH Identity Key: {ssh_key}")
    log(f"Parallel Worker Threads: Transfers={num_transfers}, Checkers={num_checkers}, RAM Buffer=128MB")
    
    # 1. Create target directory on AQUA Scratch
    mkdir_cmd = f"ssh -p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} 'mkdir -p ~/scratch/{dest_folder}'"
    run_command_with_live_logs(mkdir_cmd, label="AQUA-MKDIR")
    
    # 2. Optimized rclone stream over dynamic SFTP backend
    cmd = [
        "rclone", "copy",
        "drive:", sftp_remote,
        "--drive-root-folder-id", DRIVE_FOLDER_ID,
        "--transfers", str(num_transfers),
        "--checkers", str(num_checkers),
        "--buffer-size", "128M",
        "--drive-chunk-size", "64M",
        "--progress",
        "--stats", "2s"
    ]
    
    run_command_with_live_logs(cmd, label="RCLONE-SFTP-STREAM")

def verify_remote_destination(dest_folder):
    """
    Verifies transferred files on AQUA cluster scratch space after upload.
    """
    log("Verifying transferred files on AQUA cluster scratch space...")
    verify_cmd = f"ssh -p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} 'ls -la ~/scratch/{dest_folder} | head -n 20'"
    try:
        run_command_with_live_logs(verify_cmd, label="AQUA-VERIFY")
    except Exception as e:
        log(f"Verification check failed: {e}", level="WARNING")

def main():
    parser = argparse.ArgumentParser(description="High-Speed Zero-Disk Streaming Transfer: Google Drive -> AQUA Cluster ~/scratch/Face_Dataset over SFTP Stream")
    parser.add_argument("--dest_folder", type=str, default=AQUA_SCRATCH_DEST, help="Remote folder name inside ~/scratch/ on AQUA cluster")
    parser.add_argument("--transfers", type=int, default=32, help="Number of parallel concurrent file transfer workers")
    parser.add_argument("--checkers", type=int, default=64, help="Number of parallel checker threads")
    args = parser.parse_args()
    
    start_total_time = time.time()
    log("==========================================================================")
    log("STARTING HIGH-SPEED ZERO-LOCAL-DISK STREAMING PIPELINE")
    log(f"Local Laptop Memory Buffer: 128MB RAM (0 Bytes Local Disk Space Required)")
    log(f"Google Drive Folder ID: {DRIVE_FOLDER_ID}")
    log(f"Google Drive Link: {DRIVE_FOLDER_LINK}")
    log(f"Target AQUA Scratch: {AQUA_USER}@{AQUA_HOST}:~/scratch/{args.dest_folder}")
    log("==========================================================================")

    scratch_dest = args.dest_folder

    try:
        stream_high_speed_rclone(scratch_dest, num_transfers=args.transfers, num_checkers=args.checkers)
        verify_remote_destination(scratch_dest)
        total_elapsed = time.time() - start_total_time
        log("==========================================================================")
        log(f"✓ ZERO-DISK STREAMING TRANSFER COMPLETED IN {total_elapsed / 60.0:.2f} MINUTES")
        log("==========================================================================")
    except Exception as e:
        log(f"Streaming transfer failed: {e}", level="ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
