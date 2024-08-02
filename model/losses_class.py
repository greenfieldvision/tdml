import torch
import torch.nn as nn

import tdml.model.detail as detail
import tdml.model.batch_miners as batch_miners


class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.margin_positive = args.margin_positive
        self.margin_negative = args.margin_negative

    def forward(self, classes, embeddings, **kwargs):
        same_group, different_group = detail.calculate_groups_pt(classes.detach())

        distances, _, _ = detail.calculate_similarities_pt(embeddings)

        contrastive_losses = (
            detail.hinge(self.margin_positive, distances) * same_group
            + detail.hinge(distances, self.margin_negative) * different_group
        )
        contrastive_loss = detail.safe_division(
            torch.sum(contrastive_losses), torch.sum(same_group) + torch.sum(different_group)
        )

        return contrastive_loss


class SamplingMattersLoss(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()

        self.margin = args.margin
        self.nu = 0.0
        self.beta_constant = False
        self.beta_val = args.beta

        if self.beta_constant:
            self.beta = self.beta_val
        else:
            self.beta = torch.nn.Parameter(torch.ones(num_classes) * self.beta_val)

        self.batchminer = batch_miners.SamplingMattersBatchMiner()

    def forward(self, classes, embeddings, **kwargs):
        sampled_triplets = self.batchminer(embeddings, classes)

        if len(sampled_triplets):
            d_ap, d_an = [], []
            for triplet in sampled_triplets:
                train_triplet = {
                    "Anchor": embeddings[triplet[0], :],
                    "Positive": embeddings[triplet[1], :],
                    "Negative": embeddings[triplet[2]],
                }

                pos_dist = ((train_triplet["Anchor"] - train_triplet["Positive"]).pow(2).sum() + 1e-8).pow(1 / 2)
                neg_dist = ((train_triplet["Anchor"] - train_triplet["Negative"]).pow(2).sum() + 1e-8).pow(1 / 2)

                d_ap.append(pos_dist)
                d_an.append(neg_dist)
            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

            if self.beta_constant:
                beta = self.beta
            else:
                beta = (
                    torch.stack([self.beta[classes[triplet[0]]] for triplet in sampled_triplets])
                    .to(torch.float)
                    .to(d_ap.device)
                )

            pos_loss = torch.nn.functional.relu(d_ap - beta + self.margin)
            neg_loss = torch.nn.functional.relu(beta - d_an + self.margin)

            pair_count = torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)).to(torch.float).to(d_ap.device)
            if pair_count == 0.0:
                loss = torch.sum(pos_loss + neg_loss)
            else:
                loss = torch.sum(pos_loss + neg_loss) / pair_count

            if self.nu:
                beta_regularization_loss = torch.sum(beta)
                loss += self.nu * beta_regularization_loss.to(torch.float).to(d_ap.device)
        else:
            loss = torch.tensor(0.0).to(torch.float).to(embeddings.device)

        return loss


class FacenetLoss(nn.Module):
    RHO = 10.0

    def __init__(self, args):
        super().__init__()

        self.margin = args.margin

    def forward(self, classes, embeddings, **kwargs):
        same_group, different_group = detail.calculate_groups_pt(classes.detach())
        valid_triplet = detail.calculate_triplets(same_group, different_group)

        distances, _, _ = detail.calculate_similarities_pt(embeddings)

        # Make 3 dimensional matrix of triplet losses.
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)
        triplet_losses = detail.hinge(triplet_distance_differences, self.margin)

        # Make positive pair semihard losses.
        semihard_triplet = (
            torch.logical_and(triplet_losses > 0.0, triplet_losses <= self.margin).to(torch.float) * valid_triplet
        )
        positive_pair_semihard_losses = torch.amax(triplet_losses * semihard_triplet, dim=2)

        # Make positive pair hard losses.
        hard_triplet = (triplet_losses > self.margin).to(torch.float) * valid_triplet
        shifted_losses = torch.amin(triplet_losses - self.RHO * hard_triplet, dim=2)
        positive_pair_hard_losses = (shifted_losses + self.RHO) * (shifted_losses < 0.0).to(torch.float)

        # Calculate Facenet loss from positive pair losses.
        positive_pair_losses = (
            positive_pair_semihard_losses
            + (positive_pair_semihard_losses == 0.0).to(torch.float) * positive_pair_hard_losses
        )
        facenet_loss = detail.safe_division(torch.sum(positive_pair_losses), torch.sum(same_group))

        return facenet_loss


class InfoNCELoss(nn.Module):
    RHO = 10.0

    def __init__(self, args):
        super().__init__()

        self.tau = args.tau

    def forward(self, classes, embeddings, **kwargs):
        same_group, different_group = detail.calculate_groups_pt(classes.detach())
        _, _, dot_products = detail.calculate_similarities_pt(embeddings)

        raw_scores = ((same_group + different_group) * (1.0 + self.RHO) - self.RHO) * dot_products / self.tau
        max_raw_score = torch.amax(raw_scores, dim=1, keepdim=True)
        shifted_scores = torch.exp(raw_scores - max_raw_score)
        shifted_positive_scores = same_group * shifted_scores
        sum_shifted_negative_scores = torch.sum(different_group * shifted_scores, dim=1, keepdim=True)

        infonce_losses = (
            -detail.safe_log(
                detail.safe_division(shifted_positive_scores, shifted_positive_scores + sum_shifted_negative_scores)
            )
            * same_group
        )
        infonce_loss = detail.safe_division(torch.sum(infonce_losses), torch.sum(same_group))

        return infonce_loss


class MultiSimilarityLossOriginal(nn.Module):
    def __init__(self, args):
        super(MultiSimilarityLossOriginal, self).__init__()
        self.thresh = args.margin
        self.margin = args.margin2

        self.scale_pos = args.alpha
        self.scale_neg = args.beta

    def forward(self, labels, feats):
        assert feats.size(0) == labels.size(
            0
        ), f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue

            # neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            # pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]
            neg_pair = neg_pair_
            pos_pair = pos_pair_

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = (
                1.0 / self.scale_pos * torch.log(1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            )
            neg_loss = (
                1.0 / self.scale_neg * torch.log(1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            )
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss


class MultiSimilarityLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.margin = args.margin
        self.margin2 = args.margin2
        self.alpha = args.alpha
        self.beta = args.beta

    def forward(self, classes, embeddings, **kwargs):
        # Calculate grouping information according to classes and dot products according to the learned embeddings.
        same_group, different_group = detail.calculate_groups_pt(classes.detach())
        _, _, dot_products = detail.calculate_similarities_pt(embeddings)

        # Calculate losses for positive and negative pairs.
        scaled_positive_scores = same_group * torch.exp(self.alpha * (self.margin - dot_products))
        scaled_negative_scores = different_group * torch.exp(self.beta * (dot_products - self.margin))
        pull_losses = detail.safe_log(1.0 + torch.sum(scaled_positive_scores, dim=1)) / self.alpha
        push_losses = detail.safe_log(1.0 + torch.sum(scaled_negative_scores, dim=1)) / self.beta

        # Calculate final loss as the sum of losses for positive and negative pairs.
        loss = torch.mean(pull_losses + push_losses)

        return loss
