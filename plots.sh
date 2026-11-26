#!/bin/bash

echo "-----------------------------------------------------------------------------------------"
echo "Starting icon-2i plots - `date`"
echo "-----------------------------------------------------------------------------------------"

. /home/meteonetwork/.cron_jobs_default_load
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

# Clean up the fsspec cache: every run downloads new files, so old ones are never reused.
# Without this CACHE_DIR grows by several GB per day.
# CACHE_KEEP_DAYS can be set in plots.conf (default: 2 days).
if [ -n "$CACHE_DIR" ] && [ -d "$CACHE_DIR" ]; then
    keep_days="${CACHE_KEEP_DAYS:-2}"
    echo "Cleaning cache files older than ${keep_days} days in $CACHE_DIR (before: $(du -sh "$CACHE_DIR" | cut -f1))"
    find "$CACHE_DIR" -maxdepth 1 -type f -mmin +$((keep_days * 1440)) -delete
    echo "Cache cleaned (after: $(du -sh "$CACHE_DIR" | cut -f1))"
fi

echo "-----------------------------------------------------------------------------------------"
echo "Finished icon-2i plots - `date`"
echo "-----------------------------------------------------------------------------------------"
