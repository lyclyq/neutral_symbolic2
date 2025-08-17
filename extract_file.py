import os
import tarfile

for file in os.listdir("."):
    if file.endswith(".tar.gz"):
        folder_name = file.replace(".tar.gz", "")
        os.makedirs(folder_name, exist_ok=True)
        print(f"Extracting {file} into ./{folder_name}/ ...")
        with tarfile.open(file, "r:gz") as tar:
            tar.extractall(path=folder_name)
