#!/usr/bin/env python3
"""Copy existing Azure cache objects to R2; never delete or switch build backends."""
import argparse
import json
import os
import subprocess


PREFIXES = (
    "deeplearning4j/releases/compiler-cache/v1",
    "deeplearning4j/releases/toolchain-cache/v1",
    "deeplearning4j/releases/dependency-cache/v2",
)


def environment():
    env = os.environ.copy()
    required = ("AZURE_SCCACHE_CONNECTION_STRING", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    for name in required:
        if not env.get(name, "").strip():
            raise ValueError(f"Required credential environment variable is missing: {name}")
    # Secret input may contain a trailing newline from a file or clipboard.
    # Normalize surrounding whitespace before constructing S3 authorization headers.
    for name in required[1:]:
        value = env[name].strip()
        if not value.isascii() or any(char.isspace() or ord(char) < 33 or ord(char) == 127 for char in value):
            raise ValueError(f"{name} contains invalid embedded whitespace or control characters")
        env[name] = value
        if env.get("GITHUB_ACTIONS") == "true":
            print(f"::add-mask::{value.replace('%', '%25')}", flush=True)
    fields = {}
    for entry in env[required[0]].strip().split(";"):
        if entry.strip():
            key, value = entry.strip().split("=", 1)
            fields[key.lower()] = value
    if env.get("GITHUB_ACTIONS") == "true":
        for name in ("sharedaccesssignature", "accountkey"):
            if fields.get(name):
                masked = fields[name].replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
                print(f"::add-mask::{masked}", flush=True)
    account = fields.get("accountname", "dl4jrel26302370c1eeb25")
    endpoint = fields.get("blobendpoint", f"https://{account}.blob.core.windows.net").rstrip("/")
    # This migration is intentionally restricted to the existing cache account.
    if account != "dl4jrel26302370c1eeb25" or endpoint != f"https://{account}.blob.core.windows.net":
        raise ValueError("Azure credential does not identify the expected cache account")
    # Do not inherit unrelated rclone remote definitions or options.
    env = {key: value for key, value in env.items() if not key.startswith("RCLONE_")}
    env.update({
        "RCLONE_CONFIG_AZURE_TYPE": "azureblob",
        "RCLONE_CONFIG_AZURE_ENCODING": "None",
        "RCLONE_CONFIG_R2_TYPE": "s3",
        "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": env["R2_ACCESS_KEY_ID"],
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": env["R2_SECRET_ACCESS_KEY"],
        "RCLONE_CONFIG_R2_ENDPOINT": "https://318204901782458555a243ad96f80e3f.r2.cloudflarestorage.com",
        "RCLONE_CONFIG_R2_REGION": "auto",
        "RCLONE_CONFIG_R2_NO_CHECK_BUCKET": "true",
        "RCLONE_CONFIG_R2_ENCODING": "None",
    })
    if fields.get("sharedaccesssignature"):
        env["RCLONE_CONFIG_AZURE_SAS_URL"] = endpoint + "/releases?" + fields["sharedaccesssignature"].lstrip("?")
    elif fields.get("accountkey"):
        env["RCLONE_CONFIG_AZURE_ACCOUNT"] = account
        env["RCLONE_CONFIG_AZURE_KEY"] = fields["accountkey"]
    else:
        raise ValueError("Azure connection string requires a SAS or account key")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preview", "copy", "verify"), default="preview")
    parser.add_argument("--prefix", choices=PREFIXES, action="append", help="Defaults to all three cache namespaces")
    args = parser.parse_args()
    env = environment()
    for prefix in args.prefix or PREFIXES:
        source = f"azure:releases/{prefix}"
        destination = f"r2:dl4j-cache/{prefix}"
        common = ["--config", os.devnull, "--checkers", "8", "--stats", "30s", "--stats-one-line",
                  "--stats-log-level", "NOTICE"]
        inventory = json.loads(subprocess.check_output(
            ["rclone", "size", source, "--json", "--config", os.devnull], env=env, text=True))
        print(f"Source inventory {prefix}: {inventory['count']} objects, {inventory['bytes']} bytes", flush=True)
        if inventory["count"] == 0:
            raise ValueError(f"Refusing to certify an empty source namespace: {prefix}")
        if args.mode != "verify":
            command = ["rclone", "copy", source, destination, "--transfers", "8",
                       "--s3-upload-concurrency", "2", "--s3-chunk-size", "16Mi", *common]
            if args.mode == "preview":
                command.append("--dry-run")
            print(f"{args.mode}: {source} -> {destination}", flush=True)
            subprocess.run(command, env=env, check=True)
        if args.mode != "preview":
            # Compare downloaded bytes even when Azure blobs have no stored MD5.
            # Extra destination objects are retained, never deleted.
            print(f"Verifying bytes: {prefix}", flush=True)
            subprocess.run(["rclone", "check", source, destination, "--download", "--one-way", *common],
                           env=env, check=True)
    print("Preview complete; no objects written." if args.mode == "preview" else
          "Selected cache namespaces verified. Build backend unchanged; Azure source retained.", flush=True)


if __name__ == "__main__":
    main()
