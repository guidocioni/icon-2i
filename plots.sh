cd "$(dirname "$0")";
current_dir=$(pwd)
export ECCODES_DEFINITION_PATH="$current_dir/grib_tables/dwd"

# Source the config file
if [ -f ./plots.conf ]; then
    source ./plots.conf
else
    echo "Missing plots.conf! Please create it (see README)."
    exit 1
fi

for proj in "${projections[@]}"; do
    echo "Running scripts for projection: $proj"
    printf "%s\n" "${scripts[@]}" | parallel -j 2 python {} --projection "$proj"
done
