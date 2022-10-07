import wandb
from run import main, parse_args

# Example sweep configuration
sweep_configuration = {
    'method': 'bayes',
    'name': 'margin_lr_sweep',
    'metric': {"name": "val_map/map@5", "goal": "maximize"},
    'parameters': {
        'lr': {'max': 1e-5, 'min': 1e-7},
        'margin': {'max': 1.0, 'min': 0.0}
     }
}



def my_train_func():
    
    wandb.init()
    margin = wandb.config.margin
    lr = wandb.config.lr

    config, args = parse_args()

    main(config, args, margin=margin, lr=lr)



sweep_id = wandb.sweep(sweep_configuration)

# run the sweep
wandb.agent(sweep_id, function=my_train_func)
