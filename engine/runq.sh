#!/bin/zsh
# Job queue with a concurrency cap. On a 16 GB machine each torch worker sits
# around 500-700 MB, so 6 concurrent jobs (~4 GB) leaves comfortable headroom
# for the OS and avoids swapping, which would be far slower than queueing.
MAX=${MAX:-6}
throttle(){ while (( $(jobs -rp | wc -l) >= MAX )); do sleep 5; done }
run(){ throttle; echo "  [start $(date +%H:%M:%S)] $*" >> runq.log; "$@" >> runq.log 2>&1 & }
