cd "$(dirname "$0")";
current_dir=$(pwd)
export ECCODES_DEFINITION_PATH="$current_dir/grib_tables/dwd"
export IMAGES_DIR=/data/icon_2i/ 

# scripts=("plot_cape_cin.py" "plot_gusts.py" "plot_lpi.py" "plot_pres_t2m_winds10m.py" "plot_rh.py" \
#     "plot_snowlmt.py" "plot_winter.py" "plot_clct.py" "plot_hsnow.py" "plot_prec_clouds.py" \
#     "plot_pres_td2m_winds10m.py" "plot_sdi.py" "plot_t2m_values.py" "plot_gph_t.py" "plot_hsnow_change.py" \
#     "plot_precip_acc.py" "plot_pwat.py" "plot_shear.py" "plot_uh_max.py")

scripts=("plot_gusts.py" \
    "plot_snowlmt.py" "plot_winter.py" "plot_clct.py"
    "plot_hsnow.py" "plot_prec_clouds.py" \
    "plot_t2m_values.py" "plot_hsnow_change.py" \
    "plot_precip_acc.py" "plot_shear.py")

#projections=("nord" "sud" "centro" "it")
projections=("nord" "sud" "centro")


run_script() {
  script=$1
  proj=$2
  python "$script" --projection "$proj"
}

export -f run_script


parallel -j 3 --delay 1 run_script ::: "${scripts[@]}" ::: "${projections[@]}"
