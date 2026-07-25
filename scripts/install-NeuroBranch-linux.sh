#!/usr/bin/env bash

set -euo pipefail

REPOSITORY="Complexity-ML/NeuroBranch"
ASSET="NeuroBranch-Setup-x64.AppImage"
LATEST_URL="https://github.com/${REPOSITORY}/releases/latest/download"
CONFIG_ROOT="${XDG_CONFIG_HOME:-${HOME}/.config}"
INSTALL_DIR="${CONFIG_ROOT}/NeuroBranch/installer"
INSTALL_PATH="${INSTALL_DIR}/NeuroBranch-setup"
LOG_PATH="${TMPDIR:-/tmp}/NeuroBranch-setup.log"

if [[ "$(uname -s)" != "Linux" ]]; then
  printf 'NeuroBranch Linux Setup can only run on Linux.\n' >&2
  exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  printf 'NeuroBranch currently requires an x86_64 Linux system.\n' >&2
  exit 1
fi

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/NeuroBranch-setup.XXXXXX")"
cleanup() {
  rm -rf "${temporary_dir}"
}
trap cleanup EXIT

printf 'Downloading the latest verified NeuroBranch Setup…\n'
curl --fail --location --silent --show-error \
  "${LATEST_URL}/${ASSET}" \
  --output "${temporary_dir}/${ASSET}"
curl --fail --location --silent --show-error \
  "${LATEST_URL}/${ASSET}.sha256" \
  --output "${temporary_dir}/${ASSET}.sha256"

expected_sha="$(awk 'NR == 1 { print $1 }' "${temporary_dir}/${ASSET}.sha256")"
actual_sha="$(sha256sum "${temporary_dir}/${ASSET}" | awk '{ print $1 }')"
if [[ -z "${expected_sha}" || "${actual_sha}" != "${expected_sha}" ]]; then
  printf 'NeuroBranch Setup checksum verification failed. Nothing was installed.\n' >&2
  exit 1
fi

if [[ "${NEUROBRANCH_SETUP_VERIFY_ONLY:-0}" == "1" ]]; then
  printf 'NeuroBranch Setup checksum verified: %s\n' "${actual_sha}"
  exit 0
fi

mkdir -p "${INSTALL_DIR}"
install -m 755 "${temporary_dir}/${ASSET}" "${INSTALL_PATH}.next"
mv -f "${INSTALL_PATH}.next" "${INSTALL_PATH}"

printf 'Verified. Opening NeuroBranch Setup…\n'
APPIMAGE_EXTRACT_AND_RUN=1 nohup "${INSTALL_PATH}" --auto-install >"${LOG_PATH}" 2>&1 &
printf 'The Setup window will open now. Log: %s\n' "${LOG_PATH}"
