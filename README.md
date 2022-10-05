# bayesian-metric-learning


# Place-Recognition-Pytorch-Lightning

## Training:

### Train on MSLS

```
CUDA_VISIBLE_DEVICES=0 python train.py --directory '.' --data_dir /scratch/frwa/MSLS/ --train_cities "zurich" --workers 8 --val_cities "" --test_cities "" --batch_size 16 --arch resnet50
```

### Train on Aerial-Street 

```
CUDA_VISIBLE_DEVICES=0 python train.py --directory '.' --data_dir /work3/frwa/air-ground-project/data/ --training-dataset aerialstreet --workers 8 --batch_size 16 --arch resnet152
```

## Validation on MSLS
