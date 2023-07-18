#!/bin/bash

log=result.txt
# create log file or overwrite if already present
printf "Log File - " > $log
# append date to log file
date >> $log

for i in 1

do
  echo $i
  python main_tva_1.py --seed $i --data_path './data' >> $log

done

