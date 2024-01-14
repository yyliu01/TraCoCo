#!/bin/bash

if [ "$#" -eq 0 ] || ! [ "$1" -gt 0 ] 2> /dev/null; then
  echo "please enter a valid labelled number for training"
  exit 1

  else
	  if (($1 != 6 && $1 != 12 && $1 != 24)); then
             echo "we support the experimental setup as follows:"
             echo "
+----------------+-----------------+--------------+
| # labelled num |  max iterations | unsup weight |
+----------------+-----------------+--------------+
| 6              | 10000            | 0.3          |
+----------------+-----------------+--------------+
| 12             | 10000            | 1.0          |
+----------------+-----------------+--------------+
| 24             | 15000            | 1.0          |
+----------------+-----------------+--------------+"
          exit 1
  fi
fi

if [ "$1" == 6 ]; then
  unsup_weight=0.3
  max_iterations=10000
elif [ "$1" == 12 ]; then
  unsup_weight=1.0
  max_iterations=10000
else
  unsup_weight=1.0
  max_iterations=15000
fi

nohup python3 Code/VnetPancreas/main.py --architecture="vnet" --labeled_num="$1" --unsup_weight=${unsup_weight} --max_iterations=${max_iterations} > pan_hist_"$1"_"${max_iterations}".out &
