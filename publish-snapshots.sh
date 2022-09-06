#!/bin/bash
# See: https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow
# This just invokes the needed workflows with the default values (which mean snapshots)
# This script usually will be used within a cron job.

gh workflow run build-deploy-android-arm32.yml
gh workflow run build-deploy-android-arm64.yml
gh workflow run build-deploy-android-x86.yml
gh workflow run build-deploy-android-x86_64.yml
gh workflow run build-deploy-cross-platform.yml
gh workflow run build-deploy-linux-cuda-10.2-arm64.yml
gh workflow run build-deploy-linux-arm32.yml
gh workflow run build-deploy-linux-arm64.yml
gh workflow run build-deploy-linux-cuda-11.4.yml
gh workflow run build-deploy-linux-cuda-11.6.yml
gh workflow run build-deploy-linux-x86_64.yml
gh workflow run build-deploy-linux-x86_64-compat.yml
gh workflow run build-deploy-mac-arm64.yml
gh workflow run build-deploy-mac.yml
gh workflow run build-deploy-windows-cuda-11.4.yml
gh workflow run build-deploy-windows-cuda-11.6.yml
gh workflow run build-deploy-windows.yml