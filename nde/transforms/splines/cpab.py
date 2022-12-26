import torch
# from nflows.transforms.base import InputOutsideDomain
# from nflows.transforms.cpab_transform import cpab_transform
from nde.transforms import InputOutsideDomain
from nde.transforms.cpab_transform import cpab_transform

# backend = 'pytorch'
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
# N = 1



def _share_across_batch(params, batch_size):
    return params[None, ...].expand(batch_size, *params.shape)

def unconstrained_cpab_spline(
    inputs,
    unnormalized_theta,
    T,
    inverse=False,
    tails="linear",
    tail_bound=1.0,

):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        outputs[outside_interval_mask] = inputs[outside_interval_mask] # outside range as equal value as input
        logabsdet[outside_interval_mask] = 0 #outside as 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
        (
            outputs[inside_interval_mask],
            logabsdet[inside_interval_mask],
        ) = cpab_spline(
            inputs=inputs[inside_interval_mask], # input the inside masks
            unnormalized_theta = unnormalized_theta[inside_interval_mask],
            T = T,
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
        )

    return outputs, logabsdet


def cpab_spline(
    inputs,
    unnormalized_theta,
    T,
    inverse=False,
    left= 0.0,
    right=1.0,
):

    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()
    # transfer inputs into [0.1,0.9] from [-1,1]
    bounds_range = abs(left)+abs(right)
    inputs = (inputs/bounds_range)*0.8 + 0.5
    num_bins = unnormalized_theta.shape[-1] + 1

    divide = (torch.max(torch.abs(unnormalized_theta),1,True))[0]/3
    theta = (unnormalized_theta/(divide+1)).to(device)    # restrct theta
    theta = theta.reshape(theta.shape[0],1,theta.shape[1])
    inputs = inputs.reshape(inputs.shape[0],1,1)

    if inverse: # read the paper with definition of a, b, c
      outputs, logabsdet = cpab_transform(T,inputs,-theta)
      outputs = (outputs - 0.5)*bounds_range/0.8
      return outputs.flatten(), -logabsdet.flatten()
    else:
      outputs, logabsdet = cpab_transform(T,inputs,theta)
      outputs = (outputs - 0.5)*bounds_range/0.8
      return outputs.flatten(), logabsdet.flatten()