import torch

def angular_loss(anchors, positives, negatives, angle_bound=1.):
    """
    Calculates angular loss
    :param anchors: A torch.Tensor, (n, embedding_size)
    :param positives: A torch.Tensor, (n, embedding_size)
    :param negatives: A torch.Tensor, (n, embedding_size)
    :param angle_bound: tan^2 angle
    :return: A scalar
    """
    anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
    positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)
    negatives = torch.unsqueeze(negatives, dim=0).expand(len(positives), -1, -1)

    x = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) \
        - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (n, 1, n-1)

    # Preventing overflow
    with torch.no_grad():
        t = torch.max(x, dim=2)[0]

    x = torch.exp(x - t.unsqueeze(dim=1))
    x = torch.log(torch.exp(-t) + torch.sum(x, 2))
    loss = torch.mean(t + x)

    return loss

def bias_loss(anchors, positives, negatives):
    raise NotImplementedError()