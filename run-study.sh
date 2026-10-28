#!/usr/bin/env bash
#
# One MAST-U tearing-mode hazard study, end to end.
#
#   ./run-study.sh <folder> [coherence-gate] [threshold-gauss] [--amp-form centroid|perbin]
#
# EXAMPLE (a gate sensitivity scan, the analogue of DIII-D's POINT-A..F):
#   ./run-study.sh GATE-055 0.55
#   ./run-study.sh GATE-060 0.60
#   ./run-study.sh GATE-070 0.70
#
# Omit the threshold and it is taken from the knee that plot_ladder locates.
#
# Every stage is skipped if its output already exists, so a rerun resumes.
# The parameters are pinned to the folder on first run and checked on every
# later run: a mismatch is a hard error, so one folder cannot mix settings.
#
# Env:
#   SHOT_MIN, SHOT_MAX   shot range            (default 47000 47100)
#   WORKERS              extraction threads    (default 8)
#   N_DETECT             toroidal mode number  (default 1)
#   PARALLEL_TAGS        true -> harvest EPM and EPQ concurrently
#
set -euo pipefail

# --amp-form may sit anywhere among the arguments; the rest stay positional.
# "centroid" is Giovannozzi's Ba and the default; "perbin" is DIII-D's.
AMP_FORM=centroid
_pos=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --amp-form)    AMP_FORM=${2:?--amp-form needs a value}; shift 2 ;;
    --amp-form=*)  AMP_FORM=${1#*=}; shift ;;
    *)             _pos+=("$1"); shift ;;
  esac
done
set -- ${_pos[@]+"${_pos[@]}"}
case $AMP_FORM in
  centroid|perbin) ;;
  *) echo "ERROR: --amp-form must be centroid or perbin, not '$AMP_FORM'" >&2
     exit 1 ;;
esac

if [[ $# -lt 1 ]]; then
  sed -n '2,20p' "$0" | sed 's/^# \?//'
  exit 1
fi

folder=$1
gate=${2:-0.60}
threshold=${3:-}

SHOT_MIN=${SHOT_MIN:-47000}
SHOT_MAX=${SHOT_MAX:-47100}
WORKERS=${WORKERS:-8}
N_DETECT=${N_DETECT:-1}
PARALLEL_TAGS=${PARALLEL_TAGS:-false}

mkdir -p "$folder"
range="${SHOT_MIN}_$((SHOT_MAX - 1))"

# --- pin the parameters to the folder -------------------------------------
params="$folder/params.txt"
check_param () {                 # $1 = key, $2 = value asked for
  local have
  have=$(grep "^$1=" "$params" | head -1 | cut -d= -f2-)
  if [[ "$have" != "$2" ]]; then
    echo "ERROR: $folder was built with $1=$have, you asked for $2" >&2
    echo "       use a different folder, or delete $folder" >&2
    exit 1
  fi
}

if [[ -f $params ]]; then
  # folders pinned before --amp-form existed were all built with the default
  grep -q "^AMP_FORM=" "$params" || echo "AMP_FORM=centroid" >> "$params"
  check_param GATE     "$gate"
  check_param SHOT_MIN "$SHOT_MIN"
  check_param SHOT_MAX "$SHOT_MAX"
  check_param N_DETECT "$N_DETECT"
  check_param AMP_FORM "$AMP_FORM"
  echo "parameters match $params"
else
  { echo "GATE=$gate"
    echo "SHOT_MIN=$SHOT_MIN"
    echo "SHOT_MAX=$SHOT_MAX"
    echo "N_DETECT=$N_DETECT"
    echo "AMP_FORM=$AMP_FORM"; } > "$params"
  echo "pinned parameters to $params"
fi

echo "study folder : $folder"
echo "shots        : $SHOT_MIN .. $((SHOT_MAX - 1))"
echo "coherence    : $gate"
echo "n_detect     : $N_DETECT"
echo "amp form     : $AMP_FORM"

ladder="$folder/rm_ladder_g${gate}.csv"
joint="$folder/joint.csv"

# --- 1. detector: the threshold ladder ------------------------------------
if [[ -f $ladder ]]; then
  echo "skipping: $ladder already exists"
else
  echo "[1/5] detector"
  python run_master.py --shots $(seq "$SHOT_MIN" $((SHOT_MAX - 1))) \
    --build-detector --coherence-min "$gate" --n-detect "$N_DETECT" \
    --amp-form "$AMP_FORM" --out-csv "$ladder"
fi

# --- 2. base-rate curve and figures ---------------------------------------
echo "[2/5] base rate and figures"
python ladder_base_rate.py "$ladder" | tee "$folder/base_rate.txt"
python plot_ladder.py "$ladder" --labels "gate $gate" \
  --out "$folder/ladder_$range" | tee "$folder/knee.txt"

# --- 3. threshold -> survival table ---------------------------------------
if [[ -z $threshold ]]; then
  threshold=$(grep -oE "knee +[0-9.]+ G" "$folder/knee.txt" | head -1 \
              | grep -oE "[0-9.]+" || true)
  if [[ -z $threshold ]]; then
    echo "ERROR: no knee found; pass the threshold explicitly as argument 3" >&2
    exit 1
  fi
  echo "threshold from the knee: $threshold G"
fi
echo "THRESHOLD=$threshold" >> "$params"

# make_annotation appends _threshold<value> to the name it is given, so the
# actual file is found afterwards; a plain joint.csv is a legacy folder
found=$(ls "$folder"/joint_threshold*.csv 2>/dev/null | head -1 || true)
if [[ -n $found ]]; then
  joint=$found
  echo "skipping: $joint already exists"
elif [[ -f $joint ]]; then
  echo "skipping: $joint already exists (legacy name, no threshold in it)"
else
  echo "[3/5] annotation at $threshold G"
  python make_annotation.py "$ladder" --threshold "$threshold" --out "$joint" \
    | tee "$folder/annotation.txt"
  joint=$(grep -oE "wrote .*" "$folder/annotation.txt" | tail -1 | sed 's/^wrote //')
  [[ -f $joint ]] || { echo "ERROR: make_annotation output not found" >&2; exit 1; }
fi
echo "JOINT=$joint" >> "$params"

# --- 4. equilibrium features, one store per tree --------------------------
harvest () {                      # $1 = EPM | EPQ
  local tag=$1
  local store="$folder/harvest_$tag"
  local out="$folder/feat_$tag.pickle"
  if [[ -f $out ]]; then
    echo "skipping: $out already exists"
    return
  fi
  python extract_parallel.py --annotation "$joint" --tag "$tag" \
    --store "$store" --workers "$WORKERS"
  python extract_parallel.py --store "$store" --consolidate "$out"
}

echo "[4/5] feature harvest (EPM, EPQ)"
if [[ $PARALLEL_TAGS == true ]]; then
  echo "  running both trees concurrently"
  harvest EPM & pid_epm=$!
  harvest EPQ & pid_epq=$!
  wait $pid_epm; wait $pid_epq
else
  harvest EPM
  harvest EPQ
fi

# --- 5. where this stops --------------------------------------------------
echo "[5/5] done"
echo
echo "  $folder/"
ls -1 "$folder" | sed 's/^/    /'
echo
echo "  Next: python merge_pair.py --primary $folder/feat_EPM.pickle \\"
echo "            --secondary $folder/feat_EPQ.pickle --out $folder/merged.pickle"
echo "  (omit --secondary to work on EPM alone).  Still to be written: the model."
