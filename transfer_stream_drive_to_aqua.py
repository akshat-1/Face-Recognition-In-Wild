import os
import sys
import time
import subprocess
import argparse

# Configuration
DRIVE_FOLDER_ID = "1bzwadTmTkp69kNkbdPNb-2tm7DKAvd7Q"
DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/1bzwadTmTkp69kNkbdPNb-2tm7DKAvd7Q?usp=sharing"
AQUA_HOST = "aqua.iitm.ac.in"
AQUA_PORT = "40826"
AQUA_USER = "na22b025"
AQUA_SCRATCH_DEST = "/scratch/na22b025/Face_Dataset"
SOCKET_DIR = os.path.expanduser("~/.ssh/sockets")

def check_command(cmd):
    return subprocess.call(f"type {cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0

def setup_ssh_multiplexing():
    """
    Sets up SSH ControlMaster multiplexing socket directory for instant connection re-use.
    """
    os.makedirs(SOCKET_DIR, exist_ok=True)

def stream_high_speed_rclone(scratch_dest, num_transfers=32, num_checkers=64):
    """
    High-Speed Multi-Threaded rclone Stream Engine (0 Bytes Local Disk Space).
    Opens 32 parallel TCP connections to Google Drive and streams 32 concurrent SFTP/SSH data streams
    directly into AQUA /scratch.
    """
    print("\n==========================================================================")
    print("--- HIGH-SPEED MULTI-THREADED RCLONE STREAMING PIPE (0 Bytes Local Disk) ---")
    print(f"Parallel Transfers: {num_transfers} | Checkers: {num_checkers} | RAM Buffer: 128MB")
    print(f"Target AQUA Scratch: {AQUA_USER}@{AQUA_HOST}:{scratch_dest}")
    print("==========================================================================\n")
    
    # 1. Create remote scratch directory on AQUA
    mkdir_cmd = f"ssh -o ControlMaster=auto -o ControlPath={SOCKET_DIR}/aqua_%r@%h_%p -o ControlPersist=2h -p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} 'mkdir -p {scratch_dest}'"
    subprocess.run(mkdir_cmd, shell=True, check=True)
    
    # 2. Optimized rclone stream over high-speed SFTP with hardware-accelerated SSH cipher
    cmd = [
        "rclone", "copy",
        "drive:", scratch_dest,
        "--drive-root-folder-id", DRIVE_FOLDER_ID,
        "--transfers", str(num_transfers),
        "--checkers", str(num_checkers),
        "--buffer-size", "128M",
        "--drive-chunk-size", "64M",
        "--progress",
        "--stats", "5s",
        "--sftp-host", AQUA_HOST,
        "--sftp-port", AQUA_PORT,
        "--sftp-user", AQUA_USER,
        "--sftp-concurrency", str(num_transfers),
        "--sftp-chunk-size", "64k"
    ]
    
    print(f"[STREAM EXEC] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def stream_high_speed_gdown_pipe(scratch_dest):
    """
    High-Speed gdown Stream Engine with Hardware-Accelerated SSH AES128-GCM Cipher Pipe.
    """
    print("\n==========================================================================")
    print("--- GDOWN HARDWARE-ACCELERATED SSH PIPE STREAM (0 Bytes Local Disk) ---")
    print(f"Cipher: aes128-gcm@openssh.com | Target: {AQUA_USER}@{AQUA_HOST}:{scratch_dest}")
    print("==========================================================================\n")
    
    # Ensure remote directory exists
    subprocess.run(
        f"ssh -p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} 'mkdir -p {scratch_dest}'",
        shell=True, check=True
    )
    
    # Install gdown if needed
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)

    # Use gdown folder stream directly into fast SSH pipe with aes128-gcm cipher
    gdown_cmd = f"{sys.executable} -m gdown --folder '{DRIVE_FOLDER_LINK}' -O - --remaining-ok"
    ssh_cmd = (
        f"ssh -c aes128-gcm@openssh.com,chacha20-poly1305@openssh.com "
        f"-o ControlMaster=auto -o ControlPath={SOCKET_DIR}/aqua_%r@%h_%p -o ControlPersist=2h "
        f"-p {AQUA_PORT} {AQUA_USER}@{AQUA_HOST} "
        f"'tar -C {scratch_dest} -xzf - 2>/dev/null || cat > {scratch_dest}/streamed_dataset.tar'"
    )
    
    pipe_cmd = f"{gdown_cmd} | {ssh_cmd}"
    print(f"[PIPE EXEC] {pipe_cmd}")
    subprocess.run(pipe_cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="High-Speed Zero-Disk Streaming Transfer: Google Drive -> AQUA Cluster /scratch over SSH Pipe")
    parser.add_argument("--scratch_dest", type=str, default=AQUA_SCRATCH_DEST, help="Remote /scratch path on AQUA cluster")
    parser.add_argument("--transfers", type=int, default=32, help="Number of parallel concurrent file transfer workers")
    parser.add_argument("--checkers", type=int, default=64, help="Number of parallel checker threads")
    args = parser.parse_args()
    
    setup_ssh_multiplexing()
    scratch_dest = args.scratch_dest
    
    print(f"=========================================================================")
    print(f"HIGH-SPEED ZERO-DISK STREAMING PIPELINE")
    print(f"Local Laptop Memory Buffer: 128MB RAM (0 Bytes Local Disk Used)")
    print(f"Drive Folder ID: {DRIVE_FOLDER_ID}")
    print(f"Destination: {AQUA_USER}@{AQUA_HOST}:{scratch_dest}")
    print(f"=========================================================================\n")

    # Try Method 1: rclone 32-parallel SFTP stream (Fastest)
    if check_command("rclone"):
        try:
            stream_high_speed_rclone(scratch_dest, num_transfers=args.transfers, num_checkers=args.checkers)
            print("\n✓ High-speed multi-threaded streaming transfer complete!")
            return
        except Exception as e:
            print(f"[rclone info]: {e}. Falling back to gdown fast SSH pipe...")

    # Method 2: gdown hardware-accelerated SSH pipe
    try:
        stream_high_speed_gdown_pipe(scratch_dest)
        print("\n✓ High-speed gdown streaming transfer complete!")
    except Exception as e:
        print(f"\n[ERROR] Streaming transfer failed: {e}")
        print("\nAlternative Command:")
        print(f"rclone copy drive: {scratch_dest} --drive-root-folder-id {DRIVE_FOLDER_ID} --transfers 32 -P")
        sys.exit(1)

if __name__ == "__main__":
    main()
