import torch


def compute_rank(embed, ref_embed=None):

    if ref_embed is None:
        same_source = True
        ref_embed = embed
    else:
        same_source = False

    dist = embed @ ref_embed.T
    ranks = torch.argsort(-dist, dim=1).numpy()

    # remove itself
    if same_source:
        ranks = ranks[:, 1:]

    return ranks
