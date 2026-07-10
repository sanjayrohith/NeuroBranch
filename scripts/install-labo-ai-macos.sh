#!/usr/bin/env bash

set -euo pipefail

REPOSITORY="Complexity-ML/labo-ai"
ASSET="LABO-AI-Setup-arm64-helper"
LATEST_URL="https://github.com/${REPOSITORY}/releases/latest/download"
INSTALL_DIR="${HOME}/Library/Application Support/LABO AI/setup"
INSTALL_PATH="${INSTALL_DIR}/labo-ai-setup"
LOG_PATH="${TMPDIR:-/tmp}/labo-ai-setup.log"

if [[ "$(uname -s)" != "Darwin" ]]; then
  printf 'LABO AI macOS Setup can only run on macOS.\n' >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  printf 'LABO AI currently requires an Apple silicon Mac.\n' >&2
  exit 1
fi

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/labo-ai-setup.XXXXXX")"
cleanup() {
  rm -rf "${temporary_dir}"
}
trap cleanup EXIT

printf 'Downloading the latest verified LABO AI Setup…\n'
curl --fail --location --silent --show-error \
  "${LATEST_URL}/${ASSET}" \
  --output "${temporary_dir}/${ASSET}"
curl --fail --location --silent --show-error \
  "${LATEST_URL}/${ASSET}.sha256" \
  --output "${temporary_dir}/${ASSET}.sha256"

expected_sha="$(awk 'NR == 1 { print $1 }' "${temporary_dir}/${ASSET}.sha256")"
actual_sha="$(shasum -a 256 "${temporary_dir}/${ASSET}" | awk '{ print $1 }')"
if [[ -z "${expected_sha}" || "${actual_sha}" != "${expected_sha}" ]]; then
  printf 'LABO AI Setup checksum verification failed. Nothing was installed.\n' >&2
  exit 1
fi

if [[ "${LABO_AI_SETUP_VERIFY_ONLY:-0}" == "1" ]]; then
  printf 'LABO AI Setup checksum verified: %s\n' "${actual_sha}"
  exit 0
fi

mkdir -p "${INSTALL_DIR}"
/usr/bin/install -m 755 "${temporary_dir}/${ASSET}" "${INSTALL_PATH}.next"
mv -f "${INSTALL_PATH}.next" "${INSTALL_PATH}"
xattr -dr com.apple.quarantine "${INSTALL_PATH}" 2>/dev/null || true

printf 'Verified. Opening LABO AI Setup…\n'
nohup "${INSTALL_PATH}" --auto-install >"${LOG_PATH}" 2>&1 &
printf 'The Setup window will open now. Log: %s\n' "${LOG_PATH}"
