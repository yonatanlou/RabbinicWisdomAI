"""
Clip loss from https://github.com/mlfoundations/open_clip/blob/24ddefb37fc4892f6a0c975b732226fe8a9a8613/src/open_clip/loss.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# https://github.com/ssnl/moco_align_uniform/blob/7698785a105f174fc765f3442e32c4b385008f03/main_moco.py#L87C3-L87C7
# https://github.com/ssnl/align_uniform/blob/master/align_uniform/__init__.py
def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


# https://github.com/ssnl/align_uniform/blob/master/align_uniform/__init__.py
def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class MainLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.simclr = InfoNceLoss(config.nce_t)

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, img_features, ppi_features, logit_scale):
        logits_per_image = logit_scale * img_features @ ppi_features.T
        logits_per_text = logit_scale * ppi_features @ img_features.T
        return logits_per_image, logits_per_text

    def clip_loss(self, img_features, ppi_features, logit_scale):
        device = img_features.device
        logits_per_image, logits_per_text = self.get_logits(
            img_features, ppi_features, logit_scale
        )
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        clip_loss_val = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return clip_loss_val

    def combine(
        self,
        clip_w: float = 0,
        align_w: float = 0,
        unif_w: float = 0,
        nce_w: float = 1,
    ) -> torch.Tensor:
        assert not clip_w == align_w == unif_w == nce_w == 0
        l = 0
        if clip_w != 0:
            assert self.loss_clip is not None
            l += clip_w * self.loss_clip
        if nce_w != 0:
            assert self.loss_nce is not None
            l += nce_w * self.loss_nce
        if align_w != 0:
            assert self.loss_align is not None
            l += align_w * self.loss_align
        if unif_w != 0:
            assert self.loss_unif is not None
            l += unif_w * self.loss_unif

        return l

    def forward(
        self,
        batch_ppi_input,
        batch_img_input,
        ppi_features,
        img_features,
        logit_scale,
    ):
        self.loss_align = align_loss(img_features, ppi_features, alpha=2)
        self.loss_unif = (
            uniform_loss(img_features, t=2) + uniform_loss(ppi_features, t=2)
        ) / 2
        self.loss_clip = self.clip_loss(img_features, ppi_features, logit_scale)
        self.loss_nce = self.simclr.forward(
            img_features,
            ppi_features,
        )
        return {
            "Clip": self.loss_clip,
            "NCE": self.loss_nce,
            "Alignment": self.loss_align,
            "Uniform": self.loss_unif,
            "Temperatue": 1 / logit_scale,
        }


def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)


class InfoNceLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

    def forward(
        self,
        proj_1,
        proj_2,
    ):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(
            similarity_matrix / self.temperature
        )

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss
