import torch
import torch.nn as nn

import tdml.model.detail as detail


class ContrastiveLoss(nn.Module):
    def __init__(self, directed, args):
        super().__init__()

        self.margin_positive = args.margin_positive
        self.margin_negative = args.margin_negative

        self.directed = directed

    def forward(self, _, embeddings, **kwargs):
        distances, _, _ = detail.calculate_similarities_groups_pt(embeddings, 3, self.directed)

        if self.directed:
            d_01, d_02 = distances
            losses = (detail.hinge(self.margin_positive, d_01) + detail.hinge(d_02, self.margin_negative)) / 2.0
        else:
            d_01, d_02, d_12 = distances
            losses = (
                detail.hinge(self.margin_positive, d_01)
                + detail.hinge(d_02, self.margin_negative)
                + detail.hinge(d_12, self.margin_negative)
            ) / 3.0

        return torch.mean(losses)


class TripletMarginLoss(nn.Module):
    def __init__(self, directed, args):
        super().__init__()

        self.margin = args.margin

        self.directed = directed

    def forward(self, _, embeddings, **kwargs):
        distances, _, _ = detail.calculate_similarities_groups_pt(embeddings, 3, self.directed)

        if self.directed:
            d_01, d_02 = distances
            losses = detail.hinge(d_02 - d_01, self.margin)
        else:
            d_01, d_02, d_12 = distances
            losses = (detail.hinge(d_02 - d_01, self.margin) + detail.hinge(d_12 - d_01, self.margin)) / 2.0

        return torch.mean(losses)


class DoubleMarginLoss(nn.Module):
    def __init__(self, directed, args):
        super().__init__()

        self.margin = args.margin
        self.beta = args.beta

        self.directed = directed

    def forward(self, _, embeddings, **kwargs):
        distances, _, _ = detail.calculate_similarities_groups_pt(embeddings, 3, self.directed)

        if self.directed:
            d_01, d_02 = distances
            losses = self._calculate_margin_losses(d_01, d_02)
        else:
            d_01, d_02, d_12 = distances
            losses = (self._calculate_margin_losses(d_01, d_02) + self._calculate_margin_losses(d_01, d_12)) / 2.0

        return torch.mean(losses)

    def _calculate_margin_losses(self, d_ap, d_an):
        positive_losses = detail.hinge(self.beta - self.margin, d_ap)
        negative_losses = detail.hinge(d_an, self.beta + self.margin)

        return positive_losses + negative_losses


class InfoNCELoss(nn.Module):
    def __init__(self, directed, args):
        super().__init__()

        self.tau = args.tau

        self.directed = directed

    def forward(self, _, embeddings, **kwargs):
        _, _, dot_products = detail.calculate_similarities_groups_pt(embeddings, 3, self.directed)

        if self.directed:
            s_01, s_02 = dot_products

            s_max = torch.maximum(s_01, s_02)
            s_01n = torch.exp((s_01 - s_max) / self.tau)
            s_02n = torch.exp((s_02 - s_max) / self.tau)

            probabilities = detail.safe_division(s_01n, s_01n + s_02n)
            losses = -detail.safe_log(probabilities)

        else:
            s_01, s_02, s_12 = dot_products

            s_max = torch.maximum(torch.maximum(s_02, s_12), s_01)
            s_02n = torch.exp((s_02 - s_max) / self.tau)
            s_12n = torch.exp((s_12 - s_max) / self.tau)
            s_01n = torch.exp((s_01 - s_max) / self.tau)

            probabilities = detail.safe_division(s_01n, s_01n + s_02n + s_12n)
            losses = -detail.safe_log(probabilities)

        return torch.mean(losses)


class MultiSimilarityLoss(nn.Module):
    def __init__(self, directed, args):
        super().__init__()

        self.margin = args.margin
        self.alpha = args.alpha
        self.beta = args.beta

        self.directed = directed

    def forward(self, _, embeddings, **kwargs):
        _, _, dot_products = detail.calculate_similarities_groups_pt(embeddings, 3, self.directed)

        if self.directed:
            s_01, s_02 = dot_products
            losses = self._calculate_ms_losses(s_01, s_02)
        else:
            s_01, s_02, s_12 = dot_products
            losses = (self._calculate_ms_losses(s_01, s_02) + self._calculate_ms_losses(s_01, s_12)) / 2.0

        return torch.mean(losses)

    def _calculate_ms_losses(self, s_ap, s_an):
        scaled_positive_scores = torch.exp(self.alpha * (self.margin - s_ap))
        scaled_negative_scores = torch.exp(self.beta * (s_an - self.margin))

        pull_losses = detail.safe_log(1.0 + scaled_positive_scores) / self.alpha
        push_losses = detail.safe_log(1.0 + scaled_negative_scores) / self.beta

        return torch.mean(pull_losses + push_losses)
