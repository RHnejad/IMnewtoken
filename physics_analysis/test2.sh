# runai bash intintermask
source /mnt/vita/scratch/vita-staff/users/rh/miniconda3/etc/profile.d/conda.sh
cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/default_intermask/InterMask/
conda activate newton_env
# python physics_analysis/newton_force_analysis.py --clip 1000


# python physics_analysis/newton_force_analysis.py --clip 1000
python physics_analysis/newton_force_analysis.py --clip 2000 --methods 1 
python physics_analysis/newton_force_analysis.py --clip 2000 --methods 2
python physics_analysis/newton_force_analysis.py --clip 2000 --methods 3

python physics_analysis/newton_force_analysis.py --clip 2000 --mass-uncertainty --mc-samples 10
