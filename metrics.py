import torch

def kl_div(logits, targets):
    """

    Args:
        logits: (b, seq_len, vocab_size)
        targets (b, seq_len)
    """
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # to one-hot
    targets = torch.nn.functional.one_hot(targets, num_classes=logits.shape[-1])
    return torch.nn.functional.kl_div(log_probs, targets.float(), reduction='batchmean')

def perplexity(nlls):
    """

    Args:
        nlls: (b, ) negative log likelihoods 
    """
    return torch.exp(nlls.mean()).item()
    