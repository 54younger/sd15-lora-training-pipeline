#!/usr/bin/env bash
# Reproduce the submitted diagrams locally with the pinned PlantUML renderer.
set -euo pipefail

diagram_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
diagram_root="$(cd -- "${diagram_script_dir}/.." && pwd)"
diagram_jar="${1:-}"
diagram_java="${JAVA_BIN:-java}"
diagram_expected_sha256="0f77e5f769836b3dee340e207fe497c3e4c43e973d559e3c306915da9c32e34c"

if [[ -z "${diagram_jar}" || ! -f "${diagram_jar}" ]]; then
  printf 'Usage: JAVA_BIN=/path/to/java bash scripts/render-diagrams.sh /path/to/plantuml-1.2026.8.jar\n' >&2
  exit 2
fi
if ! command -v "${diagram_java}" >/dev/null 2>&1; then
  printf 'Java is unavailable. Set JAVA_BIN to a Java 21 executable.\n' >&2
  exit 2
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  printf 'sha256sum is required to verify the pinned renderer.\n' >&2
  exit 2
fi
diagram_actual_sha256="$(sha256sum -- "${diagram_jar}")"
diagram_actual_sha256="${diagram_actual_sha256%% *}"
if [[ "${diagram_actual_sha256}" != "${diagram_expected_sha256}" ]]; then
  printf 'Renderer checksum mismatch. Use the PlantUML 1.2026.8 JAR linked in README.md.\n' >&2
  exit 2
fi

diagram_sources=(
  "${diagram_root}/diagrams/system-architecture.puml"
  "${diagram_root}/diagrams/job-lifecycle.puml"
  "${diagram_root}/diagrams/lease-recovery.puml"
)

"${diagram_java}" -XX:-UsePerfData -Djava.awt.headless=true -jar "${diagram_jar}" \
  --check-syntax "${diagram_sources[@]}"
for diagram_format in svg png; do
  "${diagram_java}" -XX:-UsePerfData -Djava.awt.headless=true -jar "${diagram_jar}" \
    -failfast2 "-t${diagram_format}" "${diagram_sources[@]}"
done
printf 'Rendered 3 diagrams as SVG and PNG in %s/diagrams\n' "${diagram_root}"
