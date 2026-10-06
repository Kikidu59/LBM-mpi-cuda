#!/usr/bin/env bash
# ===================================================================
# Correctness check: the MPI solver must reproduce the serial solver
# bit for bit.
#
# Builds ../serial and ../mpi, runs a small reference case with the
# serial code and with the MPI code on several rank counts (including
# a row count that does not divide evenly, to exercise the remainder
# rows), and compares the probe time series byte by byte.
#
# Usage (from the repository root):
#   tests/check_mpi_matches_serial.sh
# Environment:
#   HDF5_ROOT   HDF5 install for the serial build (see serial/README.md)
#   RANKS       rank counts to test        (default: "1 2 3 4")
#   MPIRUN      MPI launcher + flags       (default: "mpirun")
# ===================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RANKS="${RANKS:-1 2 3 4}"
read -r -a MPIRUN <<< "${MPIRUN:-mpirun}"   # may contain extra flags
CASE="nx=200 ny=101 re=100 steps=1000"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "=== building serial and MPI versions ==="
make -C "$ROOT/serial" -s
make -C "$ROOT/mpi" -s

echo "=== serial reference: $CASE ==="
(cd "$WORK" && "$ROOT/serial/lbm" $CASE every=0 probe=serial.csv > /dev/null)

status=0
for P in $RANKS; do
  (cd "$WORK" && "${MPIRUN[@]}" -np "$P" "$ROOT/mpi/lbm_mpi" $CASE probe="mpi_$P.csv" > /dev/null)
  if cmp -s "$WORK/serial.csv" "$WORK/mpi_$P.csv"; then
    echo "  P=$P : identical to serial"
  else
    echo "  P=$P : DIFFERS from serial" >&2
    status=1
  fi
done

exit $status
