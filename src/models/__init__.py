from models.networks.imageretrievalnet import init_network
from models.networks.mnistmodel import MnistLinearNet


def configure_model(args):

    ######
    # initialize model dict
    ######

    if args.dataset in ("mnist", "fashionmnist"):
        model = MnistLinearNet(args.latent_dim)
    else:
        if args.pretrained:
            print(">> Using pre-trained model '{}'".format(args.arch))
        else:
            print(">> Using model from scratch (random weights) '{}'".format(args.arch))
        model_params = {}
        model_params["architecture"] = args.arch
        model_params["pooling"] = args.pool
        model_params["regional"] = args.regional
        model_params["whitening"] = args.whitening
        model_params["pretrained"] = args.pretrained

        model = init_network(model_params)

    return model


def get_model_parameters(model, args):

    if hasattr(model, "fc_log_var"):
        parameters = [{"params": model.fc_log_var.parameters()}]
        
    elif hasattr(model, "pool"):
        parameters = []

        # parameters split into features, pool, whitening
        # IMPORTANT: no weight decay for pooling parameter p in GeM or regional-GeM

        # add feature parameters
        parameters.append({"params": model.backbone.parameters()})

        # global, only pooling parameter p weight decay should be 0
        parameters.append(
            {
                "params": model.pool.parameters(),
                "lr": args.lr * 10,
                "weight_decay": 0,
            }
        )

        parameters.append({"params": model.linear.parameters()})
    else:
        parameters = [{"params": model.parameters()}]

    return parameters
