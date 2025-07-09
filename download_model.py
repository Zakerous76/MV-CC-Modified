# A script to download the InternVideo2_Chat_8B_InternLM2_5 and save it locally
from huggingface_hub import snapshot_download
import os

video_dir = "./InternVideo2_Chat_8B_InternLM2_5"
os.makedirs(video_dir, exist_ok=True)

snapshot_download(
    repo_id="OpenGVLab/InternVideo2_Chat_8B_InternLM2_5",
    local_dir=video_dir,
    local_dir_use_symlinks=False  # safer for Windows
)
