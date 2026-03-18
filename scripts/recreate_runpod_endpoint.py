#!/usr/bin/env python3
"""
Clone a RunPod Serverless endpoint into a fresh endpoint, optionally delete the old one,
and update RUNPOD_ENDPOINT_ID in a local .env file.

Usage:
    python scripts/recreate_runpod_endpoint.py
    python scripts/recreate_runpod_endpoint.py --delete-old
    python scripts/recreate_runpod_endpoint.py --new-name "fresh short-hair endpoint -fb"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

RUNPOD_REST_BASE = "https://rest.runpod.io/v1"


def load_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def replace_env_value(env_path: Path, key: str, value: str) -> None:
    if not env_path.exists():
        env_path.write_text(f"{key}={value}\n")
        return

    lines = env_path.read_text().splitlines()
    replaced = False
    new_lines: list[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        if new_lines and new_lines[-1] != "":
            new_lines.append("")
        new_lines.append(f"{key}={value}")
    env_path.write_text("\n".join(new_lines) + "\n")


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def get_endpoint(api_key: str, endpoint_id: str) -> dict[str, Any]:
    resp = requests.get(
        f"{RUNPOD_REST_BASE}/endpoints/{endpoint_id}",
        headers=build_headers(api_key),
        params={"includeWorkers": "true", "includeTemplate": "true"},
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


def create_template(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    query = """
    mutation CreateTemplate($input: SaveTemplateInput!) {
      saveTemplate(input: $input) {
        id
        name
        imageName
        isServerless
        containerDiskInGb
        readme
      }
    }
    """
    resp = requests.post(
        "https://api.runpod.io/graphql",
        params={"api_key": api_key},
        headers={"content-type": "application/json"},
        json={"query": query, "variables": {"input": payload}},
        timeout=60,
    )
    resp.raise_for_status()
    obj = resp.json()
    if obj.get("errors"):
        raise RuntimeError(json.dumps(obj["errors"], ensure_ascii=False))
    return obj["data"]["saveTemplate"]


def create_endpoint(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        f"{RUNPOD_REST_BASE}/endpoints",
        headers=build_headers(api_key),
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def patch_endpoint(api_key: str, endpoint_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.patch(
        f"{RUNPOD_REST_BASE}/endpoints/{endpoint_id}",
        headers=build_headers(api_key),
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def delete_endpoint(api_key: str, endpoint_id: str) -> None:
    resp = requests.delete(
        f"{RUNPOD_REST_BASE}/endpoints/{endpoint_id}",
        headers=build_headers(api_key),
        timeout=60,
    )
    if resp.status_code not in (200, 202, 204):
        resp.raise_for_status()


def wait_for_workers(api_key: str, endpoint_id: str, timeout_sec: int = 300) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        endpoint = get_endpoint(api_key, endpoint_id)
        workers = endpoint.get("workers") or []
        if workers:
            return workers
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for workers on endpoint {endpoint_id}")


def wait_for_zero_workers(api_key: str, endpoint_id: str, timeout_sec: int = 180) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        endpoint = get_endpoint(api_key, endpoint_id)
        workers = endpoint.get("workers") or []
        if not workers:
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting to drain endpoint {endpoint_id}")


def clone_payload(endpoint: dict[str, Any], new_name: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "templateId": endpoint["templateId"],
        "name": new_name or endpoint["name"],
        "gpuTypeIds": endpoint.get("gpuTypeIds") or [],
        "gpuCount": endpoint.get("gpuCount", 1),
        "idleTimeout": endpoint.get("idleTimeout", 5),
        "scalerType": endpoint.get("scalerType", "QUEUE_DELAY"),
        "scalerValue": endpoint.get("scalerValue", 4),
        "workersMin": endpoint.get("workersMin", 0),
        "workersMax": endpoint.get("workersMax", 3),
        "flashboot": endpoint.get("flashboot", False),
        "executionTimeoutMs": endpoint.get("executionTimeoutMs", 600000),
    }

    optional_keys = [
        "allowedCudaVersions",
        "computeType",
        "dataCenterIds",
        "minCudaVersion",
        "networkVolumeId",
        "networkVolumeIds",
        "instanceIds",
    ]
    for key in optional_keys:
        value = endpoint.get(key)
        if value not in (None, "", []):
            payload[key] = value

    return payload


def clone_template_payload(template: dict[str, Any], template_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "imageName": template["imageName"],
        "name": template_name,
        "containerDiskInGb": template.get("containerDiskInGb") or 30,
        "dockerArgs": "",
        "env": [],
        "isServerless": True,
        "readme": template.get("readme") or "",
        "volumeInGb": 0,
    }
    return payload


def format_summary(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": endpoint.get("id"),
        "name": endpoint.get("name"),
        "templateId": endpoint.get("templateId"),
        "workersMin": endpoint.get("workersMin"),
        "workersMax": endpoint.get("workersMax"),
        "gpuTypeIds": endpoint.get("gpuTypeIds"),
        "gpuCount": endpoint.get("gpuCount"),
        "idleTimeout": endpoint.get("idleTimeout"),
        "scalerType": endpoint.get("scalerType"),
        "scalerValue": endpoint.get("scalerValue"),
        "workerImages": sorted({w.get("imageName") for w in endpoint.get("workers", [])}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recreate a RunPod Serverless endpoint and update .env")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--new-name", default=None, help="Optional replacement name for the recreated endpoint")
    parser.add_argument(
        "--name-suffix",
        default=None,
        help="Optional suffix appended to the current endpoint name (ignored if --new-name is set)",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Scale old endpoint down to zero, delete it, and keep only the new endpoint",
    )
    parser.add_argument(
        "--drain-source-first",
        action="store_true",
        help="Scale the source endpoint to workersMin=0/workersMax=0 before creating the new endpoint",
    )
    parser.add_argument(
        "--wait-workers",
        action="store_true",
        help="Wait until the recreated endpoint has at least one worker attached",
    )
    args = parser.parse_args()

    env_values = load_env_file(args.env_file)
    api_key = args.api_key or env_values.get("RUNPOD_API_KEY")
    endpoint_id = args.endpoint_id or env_values.get("RUNPOD_ENDPOINT_ID")

    if not api_key:
        print("RUNPOD_API_KEY is required", file=sys.stderr)
        return 2
    if not endpoint_id:
        print("RUNPOD_ENDPOINT_ID is required", file=sys.stderr)
        return 2

    source = get_endpoint(api_key, endpoint_id)
    source_template = get_template(api_key, source["templateId"])
    new_name = args.new_name
    if not new_name and args.name_suffix:
        new_name = f"{source['name']}{args.name_suffix}"

    if args.drain_source_first:
        patch_endpoint(api_key, endpoint_id, {"workersMin": 0, "workersMax": 0})
        wait_for_zero_workers(api_key, endpoint_id)

    template_name = f"{source_template['name']}__clone__{int(time.time())}"
    new_template = create_template(
        api_key,
        clone_template_payload(source_template, template_name=template_name),
    )

    payload = clone_payload(source, new_name=new_name)
    payload["templateId"] = new_template["id"]
    created = create_endpoint(api_key, payload)
    replace_env_value(args.env_file, "RUNPOD_ENDPOINT_ID", created["id"])

    print(
        json.dumps(
            {
                "source": format_summary(source),
                "created": {
                    "id": created.get("id"),
                    "name": created.get("name"),
                    "templateId": created.get("templateId"),
                },
                "clonedTemplate": {
                    "id": new_template.get("id"),
                    "name": new_template.get("name"),
                    "imageName": new_template.get("imageName"),
                },
                "envFile": str(args.env_file),
                "updatedEnvEndpointId": created.get("id"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.wait_workers:
        workers = wait_for_workers(api_key, created["id"])
        print(
            json.dumps(
                {
                    "newEndpointWorkers": [
                        {
                            "id": worker.get("id"),
                            "imageName": worker.get("imageName"),
                            "slsVersion": worker.get("slsVersion"),
                        }
                        for worker in workers
                    ]
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    if not args.delete_old:
        return 0

    patch_endpoint(api_key, endpoint_id, {"workersMin": 0, "workersMax": 0})
    wait_for_zero_workers(api_key, endpoint_id)
    delete_endpoint(api_key, endpoint_id)
    print(
        json.dumps(
            {
                "deletedOldEndpointId": endpoint_id,
                "activeEndpointId": created.get("id"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
