import os
import sys
import subprocess
import argparse

# Google Drive Folder Information
DRIVE_FOLDER_ID = "1bzwadTmTkp69kNkbdPNb-2tm7DKAvd7Q"
DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/1bzwadTmTkp69kNkbdPNb-2tm7DKAvd7Q?usp=sharing"
AQUA_SCRATCH_DEFAULT = "/scratch/na22b025/Face_Dataset"
AQUA_HOME_FALLBACK = os.path.expanduser("~/Face_Dataset")

def check_command_exists(cmd):
    return subprocess.call(f"type {cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0

def download_with_rclone(dest_dir):
    """
    Downloads Google Drive folder using rclone (High-speed multi-threaded transfer).
    Requires rclone configured remote (e.g., 'drive:').
    """
    print(f"[RCLONE] Starting high-speed multi-threaded transfer of Google Drive folder '{DRIVE_FOLDER_ID}' -> {dest_dir}...")
    cmd = [
        "rclone", "copy",
        f"drive:", dest_dir,
        "--drive-root-folder-id", DRIVE_FOLDER_ID,
        "--transfers", "16",
        "--checkers", "32",
        "--progress",
        "--stats", "10s"
    ]
    subprocess.run(cmd, check=True)

def download_with_gdown(dest_dir):
    """
    Downloads Google Drive folder using Python gdown tool.
    """
    print(f"[GDOWN] Downloading Google Drive folder -> {dest_dir}...")
    cmd = [
        sys.executable, "-m", "gdown",
        "--folder", DRIVE_FOLDER_LINK,
        "-O", dest_dir,
        "--remaining-ok"
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Download large Face_Dataset folder from Google Drive to AQUA Scratch directory")
    parser.add_argument("--dest", type=str, default=AQUA_SCRATCH_DEFAULT, help="Target destination directory on AQUA cluster")
    args = parser.parse_args()
    
    dest_dir = args.dest
    # Ensure target directory exists
    os.makedirs(dest_dir, exist_ok=True)
    print(f"=========================================================================")
    print(f"Target AQUA Destination Directory: {dest_dir}")
    print(f"Google Drive Folder Link: {DRIVE_FOLDER_LINK}")
    print(f"=========================================================================\n")

    # Try rclone first if available (fastest multi-threaded option)
    if check_command_exists("rclone"):
        try:
            download_with_rclone(dest_dir)
            print("\n✓ Download completed successfully using rclone!")
            return
        except Exception as e:
            print(f"[RCLONE Warning] rclone copy encountered an error: {e}. Falling back to gdown...")
            
    # Try gdown as primary fallback
    try:
        # Install gdown if not present
        if subprocess.call(f"{sys.executable} -m gdown --help", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) != 0:
            print("Installing gdown package...")
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
            
        download_with_gdown(dest_dir)
        print("\n✓ Download completed successfully using gdown!")
    except Exception as e:
        print(f"\n[ERROR] Failed to download Google Drive folder: {e}")
        print("\nManual Alternative:")
        print(f"1. On AQUA cluster run: rclone copy drive: /scratch/na22b025/Face_Dataset --drive-root-folder-id {DRIVE_FOLDER_ID} -P")
        print(f"2. Or run: gdown --folder '{DRIVE_FOLDER_LINK}' -O /scratch/na22b025/Face_Dataset")
        sys.exit(1)

if __name__ == "__main__":
    main()
