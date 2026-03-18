#!/usr/bin/env python3
"""
Update a RunPod Serverless endpoint-bound template to a new container image.

Usage:
    export RUNPOD_API_KEY=...
    export RUNPOD_ENDPOINT_ID=...
    python scripts/update_runpod_serverless.py --image byoungj/sd:v59 --wait
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any

import requests

RUNPOD_REST_BASE = "https://rest.runpod.io/v1"


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def get_endpoint(api_key: str, endpoint_id: str, include_workers: bool = False) -> dict[str, Any]:
    params = {"includeTemplate": "true"}
    if include_workers:
        params["includeWorkers"] = "true"
    resp = requests.get(
        f"{RUNPOD_REST_BASE}/endpoints/{endpoint_id}",
        headers=build_headers(api_key),
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def get_template(api_key: str, template_id: str) -> dict[str, Any]:
    resp = requests.get(
        f"{RUNPOD_REST_BASE}/templates/{template_id}",
        headers=build_headers(api_key),
        params={"includeEndpointBoundTemplates": "true"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def update_template(api_key: str, template_id: str, image_name: str) -> dict[str, Any]:
    resp = requests.post(
        f"{RUNPOD_REST_BASE}/templates/{template_id}/update",
        headers=build_headers(api_key),
        json={"imageName": image_name},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def summarize_workers(endpoint: dict[str, Any]) -> dict[str, int]:
    return dict(Counter((worker.get("imageName") or "<unknown>") for worker in endpoint.get("workers", [])))


def wait_for_rollout(
    api_key: str,
    endpoint_id: str,
    target_image: str,
    previous_version: int | None,
    timeout_sec: int,
    poll_interval_sec: int,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        endpoint = get_endpoint(api_key, endpoint_id, include_workers=True)
        version = endpoint.get("version")
        worker_images = summarize_workers(endpoint)
        print(
            "[wait] endpoint_version={} worker_images={}".format(
                version,
                json.dumps(worker_images, ensure_ascii=False, sort_keys=True),
            )
        )
        version_ready = previous_version is None or (isinstance(version, int) and version > previous_version)
        workers = endpoint.get("workers") or []
        workers_ready = bool(workers) and set(worker_images.keys()) == {target_image}
        if version_ready and workers_ready:
            return endpoint
        time.sleep(poll_interval_sec)

    raise TimeoutError(f"Timed out waiting for endpoint {endpoint_id} to fully roll out {target_image}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update a RunPod Serverless endpoint template image")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"), help="RunPod API key")
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"), help="RunPod endpoint ID")
    parser.add_argument("--image", required=True, help="New container image, e.g. byoungj/sd:v59")
    parser.add_argument("--wait", action="store_true", help="Wait until all workers report the new image")
    parser.add_argument("--timeout", type=int, default=900, help="Rollout wait timeout in seconds")
    parser.add_argument("--poll-interval", type=int, default=10, help="Rollout polling interval in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without updating")
    args = parser.parse_args()

    if not args.api_key:
        print("RUNPOD_API_KEY is required", file=sys.stderr)
        return 2
    if not args.endpoint_id:
        print("RUNPOD_ENDPOINT_ID is required", file=sys.stderr)
        return 2

    endpoint = get_endpoint(args.api_key, args.endpoint_id, include_workers=True)
    template_id = endpoint.get("templateId")
    if not template_id:
        print(f"Endpoint {args.endpoint_id} has no templateId", file=sys.stderr)
        return 1

    template = get_template(args.api_key, template_id)
    current_image = template.get("imageName")
    current_version = endpoint.get("version")
    worker_images = summarize_workers(endpoint)

    print(
        json.dumps(
            {
                "endpointId": args.endpoint_id,
                "endpointName": endpoint.get("name"),
                "endpointVersion": current_version,
                "templateId": template_id,
                "currentTemplateImage": current_image,
                "targetImage": args.image,
                "workerImages": worker_images,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.dry_run:
        print("[dry-run] No changes applied.")
        return 0

    if current_image == args.image:
        print("[skip] Template already points to the requested image.")
    else:
        updated = update_template(args.api_key, template_id, args.image)
        print(
            json.dumps(
                {
                    "updatedTemplateId": updated.get("id", template_id),
                    "imageName": updated.get("imageName", args.image),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    if not args.wait:
        return 0

    endpoint = wait_for_rollout(
        args.api_key,
        args.endpoint_id,
        args.image,
        current_version if isinstance(current_version, int) else None,
        timeout_sec=args.timeout,
        poll_interval_sec=args.poll_interval,
    )
    print(
        json.dumps(
            {
                "endpointId": endpoint.get("id"),
                "endpointVersion": endpoint.get("version"),
                "workerImages": summarize_workers(endpoint),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
