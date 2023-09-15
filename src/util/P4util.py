import torch


def sbs_avg_cost(performances):
    return torch.min(torch.mean(performances, dim=0))


def vbs_avg_cost(performances):
    return torch.mean(torch.min(performances, dim=1).values)


def avg_cost(pred_chosen, performances):
    return torch.mean(performances[torch.arange(len(pred_chosen)), pred_chosen])


def sbs_vbs_gap(pred_chosen, performances):
    sbs_cost = sbs_avg_cost(performances)
    vbs_cost = vbs_avg_cost(performances)
    f_cost = avg_cost(pred_chosen, performances)
    return (f_cost - vbs_cost) / (sbs_cost - vbs_cost)
