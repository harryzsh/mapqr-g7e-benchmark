#!/bin/bash
# gpu_monitor.sh - Continuous GPU monitoring (CSV output)
echo "timestamp,gpu,util%,mem_used_mb,mem_total_mb,temp_c,power_w,power_limit_w,sm_clock_mhz,mem_clock_mhz" > gpu_monitor.csv
while true; do
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader,nounits | awk -F"," -v ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{print ts","$1","$2","$3","$4","$5","$6","$7","$8","$9}'
  sleep 5
done >> gpu_monitor.csv
