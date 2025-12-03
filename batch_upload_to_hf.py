#!/usr/bin/env python3
"""
Batch upload model files to HuggingFace Hub
- Checkpoint folders are uploaded to corresponding branches
- Other model files are uploaded to main branch

Usage:
    python batch_upload_to_hf.py --model_dir ./models/my-model --repo_name my-model --org LMSeed
    
Example:
    python batch_upload_to_hf.py --model_dir ./models/GPT2-Small-Distilled-100M --repo_name GPT2-Small-Distilled-100M --org LMSeed
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login
import time


def batch_upload(model_dir: str, repo_name: str, org: str = None, private: bool = False, skip_checkpoints: bool = False):
    """
    Batch upload model directory to HuggingFace Hub
    - Checkpoint folders -> branches
    - Other files -> main branch
    
    Args:
        model_dir: Path to the local model directory
        repo_name: Name of the repository on HuggingFace Hub
        org: Organization name (e.g., 'LMSeed'). If None, uploads to personal account
        private: Whether the repository should be private
        skip_checkpoints: If True, only upload main files and skip checkpoints
    """
    
    # Validate model directory
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    
    print(f"📦 Preparing batch upload from: {model_dir}")
    
    # Construct full repo ID
    if org:
        repo_id = f"{org}/{repo_name}"
    else:
        repo_id = repo_name
    
    print(f"🎯 Target repository: {repo_id}")
    
    # Login to HuggingFace
    print("🔑 Logging in to HuggingFace...")
    try:
        login()
    except Exception as e:
        print(f"⚠️  Login failed. Please run 'huggingface-cli login' first.")
        print(f"   Error: {e}")
        return False
    
    # Create repository
    print(f"📝 Creating repository '{repo_id}'...")
    try:
        api = HfApi()
        create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✅ Repository created/verified: https://huggingface.co/{repo_id}")
        print()
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return False
    
    # Separate checkpoint folders and main files
    checkpoint_folders = []
    main_files = []
    
    for item in model_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint"):
            checkpoint_folders.append(item)
        else:
            main_files.append(item)
    
    print(f"📊 Found {len(checkpoint_folders)} checkpoint folders and {len(main_files)} main files/folders")
    print()
    
    # Upload main files to main branch
    if main_files:
        print("=" * 80)
        print("📤 UPLOADING MAIN FILES TO MAIN BRANCH")
        print("=" * 80)
        
        # Create a temporary folder with only main files
        temp_main_dir = model_dir / ".temp_main_upload"
        temp_main_dir.mkdir(exist_ok=True)
        
        try:
            # Copy/link main files to temp directory
            import shutil
            for item in main_files:
                dest = temp_main_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            
            # Upload main files
            print(f"⬆️  Uploading {len(main_files)} items to main branch...")
            api.upload_folder(
                folder_path=str(temp_main_dir),
                repo_id=repo_id,
                repo_type="model",
                revision="main",
                create_pr=False,
            )
            print(f"✅ Successfully uploaded main files to: https://huggingface.co/{repo_id}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to upload main files: {e}")
            return False
        finally:
            # Clean up temp directory
            if temp_main_dir.exists():
                shutil.rmtree(temp_main_dir)
    
    # Upload checkpoint folders to branches
    if checkpoint_folders and not skip_checkpoints:
        print("=" * 80)
        print(f"📤 UPLOADING {len(checkpoint_folders)} CHECKPOINTS TO BRANCHES")
        print("=" * 80)
        print()
        
        for idx, checkpoint_folder in enumerate(sorted(checkpoint_folders), 1):
            branch_name = checkpoint_folder.name  # Use folder name as branch name
            
            print(f"[{idx}/{len(checkpoint_folders)}] 🌿 Uploading '{checkpoint_folder.name}' to branch '{branch_name}'...")
            
            try:
                # Step 1: Create the branch from main if it doesn't exist
                try:
                    api.create_branch(
                        repo_id=repo_id,
                        branch=branch_name,
                        repo_type="model",
                        exist_ok=True  # Don't error if branch already exists
                    )
                    print(f"     📝 Branch '{branch_name}' created/verified")
                except Exception as branch_error:
                    print(f"     ⚠️  Branch creation note: {branch_error}")
                
                # Step 2: Upload files to the branch
                api.upload_folder(
                    folder_path=str(checkpoint_folder),
                    repo_id=repo_id,
                    repo_type="model",
                    revision=branch_name,
                    commit_message=f"Upload {checkpoint_folder.name}",
                    create_pr=False,
                )
                print(f"     ✅ Success: https://huggingface.co/{repo_id}/tree/{branch_name}")
                print()
                
                # Add a small delay to avoid rate limiting
                if idx < len(checkpoint_folders):
                    time.sleep(1)
                    
            except Exception as e:
                print(f"     ❌ Failed to upload {checkpoint_folder.name}: {e}")
                print()
                continue
    
    print("=" * 80)
    print("🎉 BATCH UPLOAD COMPLETED!")
    print("=" * 80)
    print(f"📍 Main files: https://huggingface.co/{repo_id}")
    print(f"🌿 Branches: https://huggingface.co/{repo_id}/branches")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch upload model directory to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all files and checkpoints
  python batch_upload_to_hf.py --model_dir ./models/GPT2-Small --repo_name GPT2-Small --org LMSeed
  
  # Upload only main files (skip checkpoints)
  python batch_upload_to_hf.py --model_dir ./models/GPT2-Small --repo_name GPT2-Small --org LMSeed --skip-checkpoints
  
  # Upload as private repository
  python batch_upload_to_hf.py --model_dir ./models/GPT2-Small --repo_name GPT2-Small --org LMSeed --private
        """
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the local model directory containing checkpoints and model files"
    )
    
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Name of the repository on HuggingFace Hub"
    )
    
    parser.add_argument(
        "--org",
        type=str,
        default=None,
        help="Organization name (e.g., 'LMSeed'). If not provided, uploads to personal account"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Only upload main files, skip checkpoint folders"
    )
    
    args = parser.parse_args()
    
    # Batch upload
    success = batch_upload(
        model_dir=args.model_dir,
        repo_name=args.repo_name,
        org=args.org,
        private=args.private,
        skip_checkpoints=args.skip_checkpoints
    )
    
    if success:
        print("\n🎉 All uploads completed successfully!")
    else:
        print("\n❌ Batch upload failed. Please check the error messages above.")
        exit(1)


if __name__ == "__main__":
    main()
