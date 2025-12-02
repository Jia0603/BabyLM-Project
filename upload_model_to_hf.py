#!/usr/bin/env python3
"""
Upload a local model to HuggingFace Hub

Usage:
    python upload_model_to_hf.py --model_path ./models/my-model --repo_name my-model --org LMSeed

Example:
    python upload_model_to_hf.py --model_path ./models/GPT2-Small-Distilled-100M --repo_name GPT2-Small-Distilled-100M --org LMSeed
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login
from transformers import AutoTokenizer, AutoModelForCausalLM


def upload_model(model_path: str, repo_name: str, org: str = None, private: bool = False):
    """
    Upload a model from local path to HuggingFace Hub
    
    Args:
        model_path: Path to the local model directory
        repo_name: Name of the repository on HuggingFace Hub
        org: Organization name (e.g., 'LMSeed'). If None, uploads to personal account
        private: Whether the repository should be private
    """
    
    # Validate model path
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    print(f"📦 Preparing to upload model from: {model_path}")
    
    # Construct full repo ID
    if org:
        repo_id = f"{org}/{repo_name}"
    else:
        repo_id = repo_name
    
    print(f"🎯 Target repository: {repo_id}")
    
    # Login to HuggingFace (will use cached token if available)
    print("🔑 Logging in to HuggingFace...")
    try:
        login()
    except Exception as e:
        print(f"⚠️  Login failed. Please run 'huggingface-cli login' first.")
        print(f"   Error: {e}")
        return False
    
    # Create repository (will skip if already exists)
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
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return False
    
    # Upload all files in the model directory
    print(f"⬆️  Uploading model files...")
    try:
        # Option 1: Upload the entire folder
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
        )
        
        print(f"✅ Successfully uploaded model to: https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload a local model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload to LMSeed organization
  python upload_model_to_hf.py --model_path ./models/GPT2-Small --repo_name GPT2-Small --org LMSeed
  
  # Upload to personal account
  python upload_model_to_hf.py --model_path ./models/my-model --repo_name my-model
  
  # Upload as private repository
  python upload_model_to_hf.py --model_path ./models/my-model --repo_name my-model --org LMSeed --private
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local model directory"
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
    
    args = parser.parse_args()
    
    # Upload the model
    success = upload_model(
        model_path=args.model_path,
        repo_name=args.repo_name,
        org=args.org,
        private=args.private
    )
    
    if success:
        print("\n🎉 Upload completed successfully!")
    else:
        print("\n❌ Upload failed. Please check the error messages above.")
        exit(1)


if __name__ == "__main__":
    main()
