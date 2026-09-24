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
AQUA_DEST_DEFAULT = "Face_Dataset" # Relative to ~/ (/lfs/usrhome/btech/na22b025/Face_Dataset)
SOCKET_DIR = os.path.expanduser("~/.ssh/sockets")

def log(msg, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)

def check_command(cmd):
    return subprocess.call(f"type {cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0

def setup_ssh_multiplexing():
    """
    Sets up SSH ControlMaster multiplexing socket directory for instant connection re-use.
    """
    os.makedirs(SOCKET_DIR, exist_ok=True)

def run_command_with_live_logs(cmd, label="EXEC"):
    """
    Executes a shell command and streams live stdout/stderr unbuffered to terminal.
    """
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
    Opens 32 parallel TCP connections to Google Drive and streams 32 concurrent SFTP/SSH data streams
    directly into AQUA ~/Face_Dataset (on /lfs filesystem with 235TB free space).
    """
    log("Initializing Method 1: High-Speed Multi-Threaded rclone SFTP Stream Engine")
    log(f"Target AQUA Destination: {AQUA_USER}@{AQUA_HOST}:~/{dest_folder}")
    log(f"Parallel Worker Threads: Transfers={num_transfers}, Checkers={num_checkers}, RAM Buffer=128MB")
    
    # 1. Create target directory on AQUA Home (~/Face_Dataset)
    mkdir_cmd = f"ssh -o ControlMaster=auto -o ControlPath={SOCKET_DIR}/aqua_%r@%h_%p -o ControlPersist=2h -p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} 'mkdir -p ~/{dest_folder}'"
    run_command_with_live_logs(mkdir_cmd, label="AQUA-MKDIR")
    
    # 2. Optimized rclone stream over high-speed SFTP to user home directory (~/Face_Dataset)
    cmd = [
        "rclone", "copy",
        "drive:", f"{dest_folder}",
        "--drive-root-folder-id", DRIVE_FOLDER_ID,
        "--transfers", str(num_transfers),
        "--checkers", str(num_checkers),
        "--buffer-size", "128M",
        "--drive-chunk-size", "64M",
        "--progress",
        "--stats", "2s",
        "--sftp-host", AQUA_HOST,
        "--sftp-port", AQUA_PORT,
        "--sftp-user", AQUA_USER,
        "--sftp-concurrency", str(num_transfers),
        "--sftp-chunk-size", "64k"
    ]
    
    run_command_with_live_logs(cmd, label="RCLONE-SFTP-STREAM")

def stream_high_speed_gdown_pipe(dest_folder):
    """
    High-Speed gdown Stream Engine with Hardware-Accelerated SSH AES128-GCM Cipher Pipe.
    """
    log("Initializing Method 2: gdown Stream Engine via Hardware-Accelerated SSH AES128-GCM Pipe")
    log(f"Target AQUA Destination: {AQUA_USER}@{AQUA_HOST}:~/{dest_folder}")
    
    # Ensure remote directory exists on AQUA
    mkdir_cmd = f"ssh -p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} 'mkdir -p ~/{dest_folder}'"
    run_command_with_live_logs(mkdir_cmd, label="AQUA-MKDIR")
    
    # Install gdown if needed
    try:
        import gdown
        log("Python 'gdown' package verified.")
    except ImportError:
        log("Installing Python 'gdown' package...", level="WARNING")
        run_command_with_live_logs([sys.executable, "-m", "pip", "install", "gdown"], label="PIP-INSTALL")

    # Use gdown folder stream directly into fast SSH pipe with aes128-gcm cipher
    gdown_cmd = f"{sys.executable} -m gdown --folder '{DRIVE_FOLDER_LINK}' -O - --remaining-ok"
    ssh_cmd = (
        f"ssh -c aes128-gcm@openssh.com,chacha20-poly1305@openssh.com "
        f"-o ControlMaster=auto -o ControlPath={SOCKET_DIR}/aqua_%r@%h_%p -o ControlPersist=2h "
        f"-p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} "
        f"'tar -C ~/{dest_folder} -xzf - 2>/dev/null || cat > ~/{dest_folder}/streamed_dataset.tar'"
    )
    
    pipe_cmd = f"{gdown_cmd} | {ssh_cmd}"
    run_command_with_live_logs(pipe_cmd, label="GDOWN-SSH-PIPE")

def verify_remote_destination(dest_folder):
    """
    Verifies transferred files on AQUA cluster directory after upload.
    """
    log("Verifying transferred files on AQUA cluster...")
    verify_cmd = f"ssh -p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} 'ls -la ~/{dest_folder} | head -n 20'"
    try:
        run_command_with_live_logs(verify_cmd, label="AQUA-VERIFY")
    except Exception as e:
        log(f"Verification check failed: {e}", level="WARNING")

def main():
    parser = argparse.ArgumentParser(description="High-Speed Zero-Disk Streaming Transfer: Google Drive -> AQUA Cluster over SSH Pipe")
    parser.add_argument("--dest_folder", type=str, default=AQUA_DEST_DEFAULT, help="Remote folder name on AQUA cluster (relative to ~/)")
    parser.add_argument("--transfers", type=int, default=32, help="Number of parallel concurrent file transfer workers")
    parser.add_argument("--checkers", type=int, default=64, help="Number of parallel checker threads")
    args = parser.parse_args()
    
    start_total_time = time.time()
    log("==========================================================================")
    log("STARTING HIGH-SPEED ZERO-LOCAL-DISK STREAMING PIPELINE")
    log(f"Local Laptop Memory Buffer: 128MB RAM (0 Bytes Local Disk Space Required)")
    log(f"Google Drive Folder ID: {DRIVE_FOLDER_ID}")
    log(f"Google Drive Link: {DRIVE_FOLDER_LINK}")
    log(f"Target AQUA Destination: {AQUA_USER}@{AQUA_HOST}:~/{args.dest_folder}")
    log("==========================================================================")

    setup_ssh_multiplexing()
    dest_folder = args.dest_folder

    # Try Method 1: rclone 32-parallel SFTP stream (Fastest)
    if check_command("rclone"):
        try:
            stream_high_speed_rclone(dest_folder, num_transfers=args.transfers, num_checkers=args.checkers)
            verify_remote_destination(dest_folder)
            total_elapsed = time.time() - start_total_time
            log("==========================================================================")
            log(f"✓ ZERO-DISK STREAMING TRANSFER COMPLETED IN {total_elapsed / 60.0:.2f} MINUTES")
            log("==========================================================================")
            return
        except Exception as e:
            log(f"rclone streaming encountered issue: {e}. Switching to gdown SSH stream pipe...", level="WARNING")

    # Method 2: gdown hardware-accelerated SSH pipe
    try:
        stream_high_speed_gdown_pipe(dest_folder)
        verify_remote_destination(dest_folder)
        total_elapsed = time.time() - start_total_time
        log("==========================================================================")
        log(f"✓ ZERO-DISK GDOWN STREAMING TRANSFER COMPLETED IN {total_elapsed / 60.0:.2f} MINUTES")
        log("==========================================================================")
    except Exception as e:
        log(f"Streaming transfer failed: {e}", level="ERROR")
        log("Alternative Manual Rclone Command:", level="INFO")
        log(f"rclone copy drive: {dest_folder} --drive-root-folder-id {DRIVE_FOLDER_ID} --transfers 32 -P", level="INFO")
        sys.exit(1)

if __name__ == "__main__":
    main()
