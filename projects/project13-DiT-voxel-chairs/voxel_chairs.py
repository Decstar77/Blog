from datasets import load_dataset

# You'll need to log in first with `huggingface-cli login` 
# or by passing your API token.
dataset = load_dataset("ShapeNet/ShapeNetCore", name="v2", use_auth_token=True)

# To download to a specific directory
# dataset = load_dataset("ShapeNet/ShapeNetCore", name="v2", cache_dir="./my_data")
