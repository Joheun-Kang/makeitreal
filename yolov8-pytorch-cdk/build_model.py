"""
Build model.tar.gz and upload to the CDK-created S3 bucket.

Usage:
    python build_model.py
    python build_model.py --model /path/to/best.pt
    python build_model.py --bucket my-bucket
"""
import argparse
import os
import subprocess
import boto3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="yolov8-dev", help="S3 bucket name")
    parser.add_argument("--model", default="model/yolov8l.pt", help="Path to model weights")
    parser.add_argument("--prefix", default="yolov8", help="S3 key prefix")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        return

    # 1. Create model.tar.gz
    model_filename = os.path.basename(args.model)
    print(f"Creating model.tar.gz with {model_filename} and code/...")
    subprocess.run(
        ["tar", "-czf", "model.tar.gz",
         "-C", os.path.dirname(os.path.abspath(args.model)), model_filename,
         "-C", PROJECT_ROOT, "code/"],
        check=True,
    )

    # 2. Upload to S3
    s3_uri = f"s3://{args.bucket}/{args.prefix}/model.tar.gz"
    print(f"Uploading to {s3_uri}...")
    s3_client = boto3.client("s3")
    s3_client.upload_file("model.tar.gz", args.bucket, f"{args.prefix}/model.tar.gz")

    print(f"Done! Uploaded to {s3_uri}")


if __name__ == "__main__":
    main()
