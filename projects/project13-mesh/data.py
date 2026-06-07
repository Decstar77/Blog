from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "YiboZhang2001/TexVerse"
FOLDER = "glbs/glbs_2k/000-000"
LOCAL_DIR = "data"
N_FILES = 100

api = HfApi()
files = api.list_repo_files(REPO_ID, repo_type="dataset")

# Keep only the files inside the target folder, take the first N.
folder_files = [f for f in files if f.startswith(FOLDER + "/")]
to_download = folder_files[:N_FILES]
print(f"Found {len(folder_files)} files in {FOLDER}, downloading {len(to_download)}")

for i, f in enumerate(to_download):
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
    )
    print(f"[{i + 1}/{len(to_download)}] {path}")