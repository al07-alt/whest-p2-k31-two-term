#!/bin/bash
# Job runner: pops the first line of jobs.txt and runs it; strictly one job at a time.
# Edit the queue only under the lock: flock jobs.lock -c '...'
cd ${REPO:-.}
touch jobs.txt
while pgrep -f "[f]srun.py" > /dev/null; do sleep 20; done   # let an orphaned run finish first
while true; do
  line=$(flock jobs.lock -c 'l=$(head -n1 jobs.txt); [ -n "$l" ] && sed -i "1d" jobs.txt; echo "$l"')
  if [ -z "$line" ]; then sleep 30; continue; fi
  [ "$line" = "STOP" ] && break
  echo "$(date +%m-%d_%H:%M:%S) START $line" >> runner.log
  bash -c "$line"
  echo "$(date +%m-%d_%H:%M:%S) END" >> runner.log
done
