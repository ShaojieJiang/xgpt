"""
Download HF model files explicitly for ease of mind when .cache needs to be cleaned.
"""

import os
import shutil

import hydra
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig


@hydra.main(config_path="./conf", config_name="download_hf_model", version_base="1.2.0")
def main(cfg: DictConfig):
    model_id = cfg.model_id

    # get filenames
    os.chdir(cfg.dst_dir)
    os.system(f"GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/{model_id}")
    dir_name = model_id.split("/")[-1]
    filenames = os.listdir(dir_name)

    exclude_fnames = ["README.md", ".gitattributes", ".git"]

    exclude_prefixes = ["flax_model", "tf_model", "model"]

    files_to_download = []
    for name in filenames:
        if name in exclude_fnames:
            continue

        should_skip = False
        for prefix in exclude_prefixes:
            if name.startswith(prefix):
                should_skip = True
                break
        if should_skip:
            continue
        files_to_download.append(name)

    shutil.rmtree(dir_name)
    os.makedirs(dir_name)

    # download files
    file_links = []
    for filename in files_to_download:
        f_link = hf_hub_download(
            repo_id=model_id, filename=filename, cache_dir=f"{dir_name}/downloading"
        )
        file_links.append(f_link)

    # resolve softlinks
    for f_link in file_links:
        basename = os.path.basename(f_link)
        resolved = os.path.realpath(f_link)
        shutil.copy(resolved, f"{dir_name}/{basename}")

    # remove working dir
    shutil.rmtree(f"{dir_name}/downloading")


if __name__ == "__main__":
    main()
