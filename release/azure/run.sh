#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
conda_exe="${DL4J_AZURE_CONDA_EXE:-${CONDA_EXE:-conda}}"
conda_env="${DL4J_AZURE_CONDA_ENV:-dl4j-azure-release}"
subscription="${AZURE_SUBSCRIPTION_ID:-}"
location="${AZURE_LOCATION:-${AZURE_DEFAULTS_LOCATION:-eastus2}}"

if [[ -z "$subscription" ]]; then
    subscription="$("$conda_exe" run --no-capture-output -n "$conda_env" \
        az account show --query id -o tsv)"
fi
if [[ -z "$subscription" ]]; then
    echo "Unable to resolve an Azure subscription; run az login or set AZURE_SUBSCRIPTION_ID." >&2
    exit 2
fi

exec "$conda_exe" run --no-capture-output -n "$conda_env" \
    python "$script_dir/release.py" \
    --subscription "$subscription" \
    --location "$location" \
    --no-wizard \
    "$@"
