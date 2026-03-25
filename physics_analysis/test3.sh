# runai bash intintermask
source /mnt/vita/scratch/vita-staff/users/rh/miniconda3/etc/profile.d/conda.sh
cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/default_intermask/InterMask/
conda activate newton_env

# Clear stale Warp kernel cache
rm -rf /home/runai-home/.cache/warp/1.11.1/
# python physics_analysis/newton_force_analysis.py --clip 1000


# python physics_analysis/newton_force_analysis.py --clip 1000
python physics_analysis/newton_force_analysis.py \
    --clips 1000 