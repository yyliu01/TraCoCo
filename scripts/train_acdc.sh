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
| 3              | 60000            | 0.5          |
+----------------+-----------------+--------------+
| 7              | 60000            | 1.0          |
+----------------+-----------------+--------------+
| 14             | 90000            | 1.0          |
+----------------+-----------------+--------------+"
          exit 1
  fi
fi

if [ "$1" == 3 ]; then
  unsup_weight=0.5
  max_iterations=60000
elif [ "$1" == 12 ]; then
  unsup_weight=1.0
  max_iterations=60000
else
  unsup_weight=1.0
  max_iterations=90000
fi

nohup python3 Code/UnetACDC/main.py --labeled_num="$1" --unsup_weight=${unsup_weight} --max_iterations=${max_iterations} > acdc_hist_"$1"_"${max_iterations}".out &
