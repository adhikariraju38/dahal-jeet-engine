#!/bin/zsh
# Progress of the current run. Read-only; safe to run any time.
cd "$(dirname "$0")"
setopt NULL_GLOB
B=$'\e[1m'; D=$'\e[2m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[0m'

echo "${B}=== $(date '+%F %H:%M:%S') ===${R}"

# --- running workers (elapsed time + the arguments that matter)
echo "\n${B}workers${R}"
ps -Ao etime=,args= \
  | grep -E "ablate_encoding\.py --block|best_response_search\.py --deals" \
  | grep -v grep \
  | sed -E 's@[^ ]*/Python @@; s@[^ ]*ablate_encoding\.py@ablate@; s@[^ ]*best_response_search\.py@best-response@' \
  | sed 's/^ */  /'

# --- encoding ablation: 18 runs (6 blocks x 3 seeds).
# Count with a glob array: `ls pattern` with no match lists the whole directory.
files=( ablate_*_s*.json )
echo "\n${B}encoding ablation${R}  ${G}${#files}${R}/18 runs complete"
# latest progress line PER BLOCK, so all six in-flight runs are visible at once
# rather than whichever happened to print most recently
if [[ -f run_fixes2.log ]]; then
  grep -E "^    \[.*\] ep " run_fixes2.log \
    | awk '{ blk=$1; line[blk]=$0 } END { for (b in line) print line[b] }' \
    | sort | sed 's/^ */  /'
fi

echo "\n${B}best-response search${R}"
if grep -qE "exploitability [-+]" run_fixes2.log 2>/dev/null; then
  grep -E "BR .*generic .*exploitability" run_fixes2.log | sed 's/^ */  /'
else
  echo "  ${D}running; first target not finished yet${R}"
fi

if (( ${#files} > 0 )); then
  echo "\n${B}ablation results so far${R}"
  .venv/bin/python - <<'PY' 2>/dev/null
import glob, json
for f in sorted(glob.glob("ablate_*_s*.json")):
    d = json.load(open(f)); v = d["final"]["TensThenTricks"]
    print(f"  {d['block']:15s} seed {d['seed']}  {v['win']:.4f}  "
          f"({d['episodes']:,} eps, {d['seconds']/60:.0f} min)")
PY
fi

if grep -q "^COMPLETE" run_fixes2.log 2>/dev/null; then
  echo "\n${G}${B}RUN COMPLETE${R} $(grep '^COMPLETE' run_fixes2.log)"
else
  echo "\n${Y}still running${R}"
fi
