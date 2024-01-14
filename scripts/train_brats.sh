#!/bin/bash

if [ "$#" -eq 0 ] || ! [ "$1" -gt 0 ] 2> /dev/null; then
  echo "please enter a valid labelled number for training"
  exit 1

  else
	  if (($1 != 25 && $1 != 50 && $1 != 100)); then
             echo "we support the experimental setup as follows:"
             echo "
+----------------+-----------------+--------------+
| # labelled num |  max iterations | unsup weight |
+----------------+-----------------+--------------+
| 25             | 30000           | 1.0          |
+----------------+-----------------+--------------+
| 50             | 30000           | 1.0          |
+----------------+-----------------+--------------+
| 100            | 30000           | 1.0          |
+----------------+-----------------+--------------+"
          exit 1
  fi
fi

if [ "$1" == 25 ]; then
  unsup_weight=1.
  max_iterations=30000
elif [ "$1" == 50 ]; then
  unsup_weight=1.0
  max_iterations=30000
else
  unsup_weight=1.0
  max_iterations=30000
fi

nohup python3 Code/UnetBRATS/main.py --architecture="unet" --labeled_num="$1" --unsup_weight=${unsup_weight} --max_iterations=${max_iterations} > brats_hist_"$1"_"${max_iterations}".out &
