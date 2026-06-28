from datasets import load_dataset

# You'll need to log in first with `huggingface-cli login`
# or by passing your API token. You must also accept the dataset license at
# https://huggingface.co/datasets/ShapeNet/ShapeNetCore before downloading.
dataset = load_dataset("ShapeNet/ShapeNetCore", name="v2", token=True)

# To download to a specific directory
# dataset = load_dataset("ShapeNet/ShapeNetCore", name="v2", cache_dir="./my_data")






