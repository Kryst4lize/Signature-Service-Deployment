import os
import shutil
import argparse

def prepare_genuine_only_dataset(src_dir, dst_dir):
    """
    Copies ONLY genuine signature folders (ignoring '_forg' folders)
    from the source directory to the destination directory.
    Prints the full paths of the copied files.
    """
    print(f"Source Directory:      {os.path.abspath(src_dir)}")
    print(f"Destination Directory: {os.path.abspath(dst_dir)}\n")
    
    for split in ["train", "test"]:
        src_split_dir = os.path.join(src_dir, split)
        dest_split_dir = os.path.join(dst_dir, split)
        
        if not os.path.exists(src_split_dir):
            print(f"Warning: Could not find {src_split_dir}. Skipping this split.")
            continue
        
        os.makedirs(dest_split_dir, exist_ok=True)
        
        # List all folders in the Kaggle split (e.g., '001', '001_forg')
        folders = [f for f in os.listdir(src_split_dir) if os.path.isdir(os.path.join(src_split_dir, f))]
        
        count = 0
        for folder in folders:
            # STRICTLY ignore any folder that has "forg" in the name
            if "forg" not in folder.lower():
                src_path = os.path.join(src_split_dir, folder)
                dest_path = os.path.join(dest_split_dir, folder)
                
                # Copy the entire genuine folder over to the clean directory
                if not os.path.exists(dest_path):
                    shutil.copytree(src_path, dest_path)
                    count += 1
                    
                # Print the full absolute path of the files processed
                print(f"Copied: {os.path.abspath(src_path)}")
                print(f"  └── To: {os.path.abspath(dest_path)}")
                
        print(f"\n[Success] Copied {count} genuine folders for the '{split}' split.\n")

def main():
    # 1. Initialize the Argument Parser
    parser = argparse.ArgumentParser(description="Filter genuine signatures from Kaggle dataset for classification training.")
    
    # 2. Define the required arguments
    parser.add_argument("--src", type=str, required=True, help="Path to the original downloaded Kaggle data (e.g., sign_data)")
    parser.add_argument("--dst", type=str, required=True, help="Path to save the cleaned data (e.g., data/sign_data)")
    
    # 3. Parse the arguments from the command line
    args = parser.parse_args()
    
    # 4. Run the main processing function
    prepare_genuine_only_dataset(args.src, args.dst)

if __name__ == "__main__":
    main()