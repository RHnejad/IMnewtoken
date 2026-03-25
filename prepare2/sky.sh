/data2/rh/conda_envs/mimickit/bin/python prepare2/compute_skyhook_metrics.py \
  --dataset interhuman --clip 10  \
  --input retargeted \
  --output-dir data/test/skyhook \
  --gpu cuda:0 --force \
  --ground-fix --ground-clearance 0.01 \
  --trim-edges 2 \
  --balance-hold-frames 20 \
  --balance-transition-frames 40 \
  --balance-knee-bend-deg 25


/data2/rh/conda_envs/mimickit/bin/python prepare2/visualize_skyhook_newton.py \
  --dataset interhuman --clip 2000 --person 0 \
  --metrics-dir data/test/skyhook \
  --mode playback --device cuda:0 --viewer gl


/data2/rh/conda_envs/mimickit/bin/python prepare2/visualize_skyhook_newton.py \
  --dataset interhuman --clip 10 \
  --metrics-dir data/test/skyhook \
  --mode playback --device cuda:0 --viewer gl