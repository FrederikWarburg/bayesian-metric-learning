# bayesian-metric-learning

# Setup

```bash
module load python3/3.9.6 cuda/11.4.2  # If on an HPC machine
python3 -m pip install -r requirements.txt
cp -n .env.example .env
```

# Running

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

Frequently run
```bash
isort src/
black src/
flake8 src/
```
