import torch
from torch import nn
from .distortion import DistortionLoss


class NeRFLoss(nn.Module):

    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion

    def forward(self, results, target, **kwargs):
        d = {}
        d['rgb'] = (results['rgb'] - target['rgb'])**2

        o = results['opacity'] + 1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opacity * (-o * torch.log(o))

        if self.lambda_distortion > 0:
            d['distortion'] =self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])

        return d
