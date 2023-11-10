#!/bin/bash

# Create or overwrite a log file
touch ./mem_check.log
echo "内存监测: " >> ./mem_check.log

while true
do
    # Get the process ID of gpt2
    pid=$(ps -ef | grep vicuna | grep -v grep | awk '{print $2}')

    # Check if the process ID is not empty
    if [[ -n $pid ]]; then
        # Get memory information of the process and append to the log
        cat /proc/$pid/status | grep -E 'VmSize|VmRSS|VmData|VmStk|VmExe|VmLib' >> ./mem_check.log
        
        # Append the current date and time to the log
        date >> ./mem_check.log
        
        # Append an empty line for readability
        echo " " >> ./mem_check.log
    fi

    # Sleep for a short duration
    sleep 0.1
done
