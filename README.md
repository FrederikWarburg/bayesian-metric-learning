# bayesian-metric-learning

```bash
# HPC, GPU
export CUDA_VISIBLE_DEVICES=0  # If on an Nvidia GPU machine
module load python3/3.9.6 cuda/11.4.2  # If on an HPC machine
source metric_learning/bin/activate
python src/run.py --config configs/fashionmnist/deterministic.yaml
```

```bash
# Local, CPU
source metric_learning/bin/activate
python src/run.py --config configs/fashionmnist/deterministic.yaml
```
