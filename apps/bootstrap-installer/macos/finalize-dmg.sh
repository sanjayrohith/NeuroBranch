#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
  printf 'Usage: %s path/to/LABO-AI-Setup.dmg\n' "$0" >&2
  exit 1
fi

input_dmg="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
script_dir="$(cd "$(dirname "$0")" && pwd)"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/labo-ai-dmg.XXXXXX")"
device=""

cleanup() {
  if [[ -n "${device}" ]]; then
    hdiutil detach "${device}" >/dev/null 2>&1 || true
  fi
  rm -rf "${temporary_dir}"
}
trap cleanup EXIT

hdiutil convert "${input_dmg}" -quiet -format UDRW -o "${temporary_dir}/writable.dmg"
attach_output="$(hdiutil attach -readwrite -noverify -noautoopen "${temporary_dir}/writable.dmg")"
device="$(printf '%s\n' "${attach_output}" | awk '/Apple_HFS/ { print $1; exit }')"
mount_point="$(printf '%s\n' "${attach_output}" | awk '/Apple_HFS/ { sub(/^.*Apple_HFS[[:space:]]+/, ""); print; exit }')"

if [[ -z "${device}" || -z "${mount_point}" || ! -d "${mount_point}" ]]; then
  printf 'Unable to mount the intermediate LABO AI DMG.\n' >&2
  exit 1
fi

base64 -D < "${script_dir}/dmg-ds-store.gz.b64" | gzip -dc > "${mount_point}/.DS_Store"
sync
hdiutil detach "${device}" -quiet
device=""

rm -f "${input_dmg}"
hdiutil convert "${temporary_dir}/writable.dmg" -quiet -format UDZO -imagekey zlib-level=9 -o "${input_dmg}"

