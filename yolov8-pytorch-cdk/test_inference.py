"""
Test inference against the SageMaker endpoint.

Usage:
    python test_inference.py --image test.jpg
    python test_inference.py --image test.jpg --endpoint my-endpoint-name
"""
import argparse
import base64
import json
import boto3
import cv2


def get_endpoint_name():
    """Find the first InService SageMaker endpoint."""
    sm = boto3.client("sagemaker")
    endpoints = sm.list_endpoints(StatusEquals="InService")["Endpoints"]
    if not endpoints:
        return None
    return endpoints[0]["EndpointName"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--endpoint", default=None, help="SageMaker endpoint name (auto-detects if omitted)")
    parser.add_argument("--output", default="output.jpg", help="Path to save output image")
    args = parser.parse_args()

    endpoint = args.endpoint or get_endpoint_name()
    if not endpoint:
        print("ERROR: No InService endpoint found. Pass --endpoint or check your deployment.")
        return

    # 1. Read and resize image
    orig_image = cv2.imread(args.image)
    if orig_image is None:
        print(f"ERROR: Could not read image: {args.image}")
        return

    image_height, image_width, _ = orig_image.shape
    x_ratio = image_width / 640
    y_ratio = image_height / 640

    resized = cv2.resize(orig_image, (640, 640))
    _, encoded = cv2.imencode(".jpg", resized)
    payload = base64.b64encode(encoded).decode("utf-8")

    # 2. Invoke endpoint
    print(f"Invoking endpoint: {endpoint}")
    runtime = boto3.client("runtime.sagemaker")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="text/csv",
        Body=payload,
    )
    result = json.loads(response["Body"].read().decode("ascii"))
    print(json.dumps(result, indent=2))

    # 3. Draw results on image
    if "boxes" in result:
        for x1, y1, x2, y2, conf, lbl in result["boxes"]:
            x1, x2 = int(x_ratio * x1), int(x_ratio * x2)
            y1, y2 = int(y_ratio * y1), int(y_ratio * y2)
            cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_image, f"Class:{int(lbl)} Conf:{conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(args.output, orig_image)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
