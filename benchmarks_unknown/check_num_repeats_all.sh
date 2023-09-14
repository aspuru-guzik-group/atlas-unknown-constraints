#!/bin/bash

for d in $( ls -d ./*/ ); do
  echo -n "$d  --->  "
  cd $d
  if [ -f results.pkl ]; then
    python ../../check_num_repeats.py
  else
    echo "MISSING"
  fi
  cd ../
done
