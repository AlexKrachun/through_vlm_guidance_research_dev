from pathlib import Path

from huggingface_hub import hf_hub_download




def download_if_missing(repo_id: str, filename: str, local_dir: str | Path) -> Path:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    local_path = local_dir / filename
    if local_path.exists():
        return local_path

    downloaded_path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename,
        local_dir=local_dir, 
        local_dir_use_symlinks=False, 
    )
    
    return Path(downloaded_path)


def hf_get(sources_dir: str | Path, repo_id: str, filename: str) -> Path:
    return download_if_missing(
        repo_id=repo_id,
        filename=filename, 
        local_dir=sources_dir
    )










