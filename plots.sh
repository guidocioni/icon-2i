#!/bin/bash

echo "-----------------------------------------------------------------------------------------"
echo "Starting icon-2i plots - `date`"
echo "-----------------------------------------------------------------------------------------"

. /home/cioni/.bashrc
conda activate models

echo "Using python binary located in `which python`"

cd "$(dirname "$0")";
current_dir=$(pwd)

# Source the config file
if [ -f ./plots.conf ]; then
    source ./plots.conf
else
    echo "Missing plots.conf! Please create it (see README)."
    exit 1
fi


# There seems to be some issues when running more than 1 script in parallel...need to investigate why
#for proj in "${projections[@]}"; do
#    echo "Running scripts for projection: $proj"
#    printf "%s\n" "${scripts[@]}" | parallel -j 1 python {} --projection "$proj"
#done

for projection in "${projections[@]}"; do
    for script in "${scripts[@]}"; do
        echo "Running: python $script --projection $projection"
        python "$script" --projection "$projection"
    done
done

