#!/usr/bin/env bash

if [[ -z "$1" ]]; then
  echo "Usage: $0 <simulation_number>"
  exit 1
fi

i=$1
echo -e "\n ## New run $(date)" >> output/prof.txt
python scripts/profile/prof_csa.py line \
	--file "data/fastdid/sim$i.csv" --fn fastdid \
	| tee -a output/prof.txt
echo "--- Finished run $(date)" >> output/prof.txt
