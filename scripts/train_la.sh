#!/bin/bash

if [ "$#" -eq 0 ] || ! [ "$1" -gt 0 ] 2> /dev/null; then
  echo "please enter a valid labelled number for training"
  exit 1

  else
	  if (($1 != 8 && $1 != 16 && $1 != 32)); then
             echo "we support the experimental setup as follows:"
             echo "
+----------------+-----------------+--------------+
| # labelled num |  max iterations | unsup weight |
+----------------+-----------------+--------------+
| 8              | 9000            | 0.3          |
+----------------+-----------------+--------------+
| 16             | 9000            | 1.0          |
+----------------+-----------------+--------------+
| 32             | 12000           | 1.0          |
+----------------+-----------------+--------------+"
          exit 1
  fi
fi

if [ "$1" == 8 ]; then
  unsup_weight=0.3
  max_iterations=9000
elif [ "$1" == 16 ]; then
  unsup_weight=1.0
  max_iterations=9000
else
  unsup_weight=1.0
  max_iterations=12000
fi

nohup python3 Code/VnetLA/main.py --architecture="vnet" --labeled_num="$1" --unsup_weight=${unsup_weight} --max_iterations=${max_iterations} > la_hist_"$1"_"${max_iterations}".out &
