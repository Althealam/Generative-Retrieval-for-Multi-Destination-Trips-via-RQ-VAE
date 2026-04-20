#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <embedding|rqvae|rqkmeans> [args...]"
  exit 1
fi

PIPELINE="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${PIPELINE}" in
  embedding)
    exec "${SCRIPT_DIR}/run_train_model_with_embedding.sh" "$@"
    ;;
  rqvae)
    exec "${SCRIPT_DIR}/run_train_model_with_rqvae.sh" "$@"
    ;;
  rqkmeans)
    exec "${SCRIPT_DIR}/run_train_model_with_rqkmeans.sh" "$@"
    ;;
  *)
    echo "Unknown pipeline: ${PIPELINE}"
    echo "Expected one of: embedding, rqvae, rqkmeans"
    exit 2
    ;;
esac
