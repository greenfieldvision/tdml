import torch
import torch.nn as nn

import tdml.model.detail as detail

from tdml.model.batch_miners import RelaxedSamplingMattersBatchMiner


class RKDLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.batch_mining is None:
            self.huber_loss = nn.HuberLoss(reduction="mean")
            self.batch_miner = None
        else:
            self.huber_loss = nn.HuberLoss(reduction="none")
            if args.batch_mining == "relaxed_sampling_matters":
                self.batch_miner = RelaxedSamplingMattersBatchMiner(args)
            else:
                raise Exception("unknown batch mining method {}".format(args.batch_mining))
        self.zero = None

    def forward(self, source_embeddings, embeddings, **kwargs):
        rkdd_loss = self._compute_rkdd_loss(source_embeddings, embeddings)
        rkda_loss = self._compute_rkda_loss(source_embeddings, embeddings)
        return 1.0 * rkdd_loss + 2.0 * rkda_loss

    def _compute_rkdd_loss(self, source_embeddings, embeddings):
        # Calculate distances according to the averaged source and learned embeddings.
        _, source_avg_distances, _, _ = detail.calculate_source_similarities(source_embeddings.detach())
        distances, _, _ = detail.calculate_similarities_pt(embeddings)

        # Calculate mean distances according to the averaged source and learned embeddings.
        n = source_embeddings.shape[0]
        source_mean_avg_distances = torch.sum(source_avg_distances, dim=1, keepdims=True) / (n - 1)
        mean_distances = torch.sum(distances, dim=1, keepdims=True) / (n - 1)

        # Calculate the Huber loss for the distances normalized by the mean.
        losses = self.huber_loss(
            detail.safe_division(distances, mean_distances),
            detail.safe_division(source_avg_distances, source_mean_avg_distances),
        )
        loss = torch.mean(losses)

        return loss

    def _compute_rkda_loss(self, source_embeddings, embeddings):
        se = source_embeddings[:, 0, :]
        is_one_hot = (
            torch.amax(torch.abs(torch.amax(se, dim=1) - torch.ones((len(se),), device=se.device))) <= detail.EPSILON
        )

        source_angles = []
        for i in range(source_embeddings.shape[1]):
            se = source_embeddings[:, i, :].detach()
            if is_one_hot:
                a = detail.calculate_triplet_angles_one_hot(se)
            else:
                a = detail.calculate_triplet_angles(se)
            source_angles.append(a)
        source_angles = torch.mean(torch.stack(source_angles), dim=0)

        if is_one_hot:
            angles = detail.calculate_triplet_angles_one_hot(embeddings)
        else:
            angles = detail.calculate_triplet_angles(embeddings)

        if self.batch_miner is None:
            # The Huber loss module is set to return a scalar (reduction=mean).
            loss = self.huber_loss(angles, source_angles)
        else:
            losses = self.huber_loss(angles, source_angles)
            flat_indexes = self.batch_miner(source_embeddings, embeddings, losses)
            if len(flat_indexes) > 0:
                loss = torch.mean(torch.take(losses, flat_indexes))
            else:
                if self.zero is None:
                    self.zero = torch.tensor(0.0).to(torch.float).to(embeddings.device)
                loss = self.zero

        return loss


class RelaxedContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.margin = args.margin
        self.tau = args.tau

    def forward(self, source_embeddings, embeddings, **kwargs):
        # Get number of instances.
        n = source_embeddings.shape[0]

        # Calculate distances according to the source embeddings.
        _, source_avg_distances, _, _ = detail.calculate_source_similarities(source_embeddings.detach())

        # Calculate pair weights according to the source embeddings.
        source_pair_weights = torch.exp(-(source_avg_distances**2) / self.tau)
        source_pair_weights_adjusted = source_pair_weights * (
            torch.ones_like(source_avg_distances, device=source_pair_weights.device)
            - torch.eye(n, device=source_pair_weights.device)
        )

        # Calculate distances according to the embeddings.
        distances, squared_distances, _ = detail.calculate_similarities_pt(embeddings)

        # Normalize the distances and squared distances.
        mean_distances = torch.mean(distances, dim=1, keepdim=True)
        normalized_distances = detail.safe_division(distances, mean_distances)
        normalized_squared_distances = detail.safe_division(squared_distances, mean_distances**2)

        # Calculate the loss based on the contrastive losses weighted by the source embedding derived pair weights.
        losses = source_pair_weights_adjusted * normalized_squared_distances + (1.0 - source_pair_weights) * (
            detail.hinge(normalized_distances, self.margin) ** 2
        )
        loss = detail.safe_division(torch.sum(losses), n * (n - 1))

        return loss


class RelaxedTripletMarginLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.triplet_weight_type = args.triplet_weight_type
        self.margin = args.margin
        self.margin2 = args.margin2
        self.tau = args.tau

    def forward(self, source_embeddings, embeddings, **kwargs):
        # Calculate distances according to the source embeddings.
        _, source_avg_distances, _, _ = detail.calculate_source_similarities(source_embeddings.detach())

        # Assign weights to triplets according to the source embeddings.
        if self.triplet_weight_type == "from_source_triplets":
            _, _, source_triplet_weights = detail.calculate_weights_from_distances(
                source_avg_distances, self.tau, self.margin2, pair_type=1, triplet_type=2
            )
        elif self.triplet_weight_type == "from_source_pairs":
            _, _, source_triplet_weights = detail.calculate_weights_from_distances(
                source_avg_distances, self.tau, pair_type=1, triplet_type=1
            )
        else:
            source_triplet_weights = None

        # Calculate distances according to the embeddings.
        distances, _, _ = detail.calculate_similarities_pt(embeddings)

        # Calculate triplet losses based on distances.
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)
        triplet_losses = detail.hinge(triplet_distance_differences, self.margin)

        # Calculate the loss based on the triplet losses weighted by source embedding derived weights.
        loss = detail.safe_division(
            torch.sum(triplet_losses * source_triplet_weights), torch.sum(source_triplet_weights)
        )

        return loss


class RelaxedFacenetLoss(nn.Module):
    RHO = 10.0

    def __init__(self, args):
        super().__init__()

        self.triplet_weight_type = args.triplet_weight_type
        self.margin = args.margin
        self.margin2 = args.margin2
        self.tau = args.tau
        self.semihard_loss_threshold = args.semihard_loss_threshold

    def forward(self, source_embeddings, embeddings, **kwargs):
        # Calculate distances according to the source embeddings.
        _, source_avg_distances, _, _ = detail.calculate_source_similarities(source_embeddings.detach())

        # Assign weights to pairs and triplets according to the source embeddings.
        if self.triplet_weight_type == "from_source_triplets":
            source_pair_weights, _, source_triplet_weights = detail.calculate_weights_from_distances(
                source_avg_distances, self.tau, self.margin2, pair_type=1, triplet_type=2
            )
        elif self.triplet_weight_type == "from_source_triplets2":
            source_pair_weights, _, source_triplet_weights = detail.calculate_weights_from_distances(
                source_avg_distances, self.tau, self.margin2, pair_type=2, triplet_type=2
            )
        elif self.triplet_weight_type == "from_source_pairs":
            source_pair_weights, _, source_triplet_weights = detail.calculate_weights_from_distances(
                source_avg_distances, self.tau, pair_type=1, triplet_type=1
            )
        elif self.triplet_weight_type == "from_source_pairs2":
            source_pair_weights, _, source_triplet_weights = detail.calculate_weights_from_distances(
                source_avg_distances, self.tau, pair_type=2, triplet_type=1
            )
        else:
            source_triplet_weights = None

        # Calculate distances according to the embeddings.
        distances, _, _ = detail.calculate_similarities_pt(embeddings)

        # Calculate triplet losses according to the embeddings.
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)
        triplet_losses = detail.hinge(triplet_distance_differences, self.margin)

        # Calculate positive pair semihard losses based on distances.
        semihard_triplet = (
            torch.logical_and(triplet_losses > 0.0, triplet_losses <= self.margin) * source_triplet_weights
        )
        positive_pair_semihard_losses = torch.amax(triplet_losses * semihard_triplet, dim=2)

        # If required, calculate positive pair hard losses based on distances.
        if self.semihard_loss_threshold > 0.0:
            hard_triplet = (triplet_losses > self.margin) * source_triplet_weights
            normalized_losses = torch.amin(triplet_losses - self.RHO * hard_triplet, dim=2)
            positive_pair_hard_losses = (normalized_losses + self.RHO) * (normalized_losses < 0.0)
        else:
            positive_pair_hard_losses = 0.0

        # Calculate final loss based on the positive pair losses weighted by source embedding derived weights.
        positive_pair_losses = (
            positive_pair_semihard_losses
            + (positive_pair_semihard_losses <= self.semihard_loss_threshold) * positive_pair_hard_losses
        )
        loss = detail.safe_division(
            torch.sum(positive_pair_losses * source_pair_weights), torch.sum(source_pair_weights)
        )

        return loss


class RelaxedInfoNCELoss(nn.Module):
    RHO = 10.0

    def __init__(self, args):
        super().__init__()

        self.tau = args.tau

        self.I = None

    def forward(self, source_embeddings, embeddings, **kwargs):
        if (self.I is None) or (self.I.shape[0] != embeddings.shape[0]):
            self.I = torch.eye(embeddings.shape[0], device=embeddings.device)

        # Calculate distances according to the source embeddings.
        _, source_avg_distances, _, _ = detail.calculate_source_similarities(source_embeddings.detach())

        # Calculate pair weights according to the source embeddings.
        relaxed_source_same_group, relaxed_source_different_group, _ = detail.calculate_weights_from_distances(
            source_avg_distances, self.tau
        )

        # Calculate dot products according to the learned embeddings.
        _, _, dot_products = detail.calculate_similarities_pt(embeddings)

        # Calculate normalized scores.
        raw_scores = dot_products / self.tau - self.RHO * self.I
        max_raw_score = torch.amax(raw_scores, dim=1, keepdim=True)
        normalized_scores = torch.exp(raw_scores - max_raw_score)
        normalized_positive_scores = relaxed_source_same_group * normalized_scores
        sum_normalized_negative_scores = torch.sum(
            relaxed_source_different_group * normalized_scores, dim=1, keepdim=True
        )

        # Calculate final loss based on same group pair losses.
        losses = (
            -detail.safe_log(
                detail.safe_division(
                    normalized_positive_scores, normalized_positive_scores + sum_normalized_negative_scores
                )
            )
            * relaxed_source_same_group
        )
        loss = detail.safe_division(torch.sum(losses), torch.sum(relaxed_source_same_group))

        return loss


class RelaxedMultiSimilarityLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.margin = args.margin
        self.tau = args.tau
        self.alpha = args.alpha
        self.beta = args.beta

    def forward(self, source_embeddings, embeddings, **kwargs):
        # Calculate distances according to the source embeddings.
        _, source_avg_distances, _, _ = detail.calculate_source_similarities(source_embeddings.detach())

        # Assign weights to pairs according to the source embeddings.
        source_pair_weights, _, _ = detail.calculate_weights_from_distances(
            source_avg_distances, self.tau, pair_type=2, triplet_type=1
        )

        # Calculate dot products according to the learned embeddings.
        _, _, dot_products = detail.calculate_similarities_pt(embeddings)

        # Calculate losses for positive and negative pairs.
        scaled_positive_scores = source_pair_weights * torch.exp(self.alpha * (self.margin - dot_products))
        scaled_negative_scores = (1.0 - source_pair_weights) * torch.exp(self.beta * (dot_products - self.margin))
        pull_losses = detail.safe_log(1.0 + torch.sum(scaled_positive_scores, dim=1)) / self.alpha
        push_losses = detail.safe_log(1.0 + torch.sum(scaled_negative_scores, dim=1)) / self.beta

        # Calculate final loss.
        loss = torch.mean(pull_losses + push_losses)

        return loss


class SoftTripletMarginRegressionLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tau = args.tau
        self.beta = args.beta

        if args.batch_mining is None:
            self.huber_loss = nn.HuberLoss(reduction="mean")
            self.batch_miner = None
        else:
            self.huber_loss = nn.HuberLoss(reduction="none")
            if args.batch_mining == "relaxed_sampling_matters":
                self.batch_miner = RelaxedSamplingMattersBatchMiner(args)
            else:
                raise Exception("unknown batch mining method {}".format(args.batch_mining))
        self.zero = None

    def forward(self, source_embeddings, embeddings, **kwargs):
        # Calculate distances according to the averaged source and learned embeddings.
        _, source_avg_distances, _, _ = detail.calculate_source_similarities(source_embeddings.detach())
        distances, _, _ = detail.calculate_similarities_pt(embeddings)

        # Calculate triplet losses according to the averaged source and learned embeddings.
        source_triplet_distance_differences = torch.unsqueeze(source_avg_distances, 1) - torch.unsqueeze(
            source_avg_distances, 2
        )
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)

        alpha = torch.sigmoid(source_triplet_distance_differences / self.beta)
        gamma = detail.safe_log(1.0 / alpha - 1.0) / self.tau
        triplet_losses = (
            alpha
            * detail.safe_log(
                1.0 + torch.exp(self.tau * (source_triplet_distance_differences - triplet_distance_differences + gamma))
            )
            + (1.0 - alpha)
            * detail.safe_log(
                1.0 + torch.exp(self.tau * (triplet_distance_differences - source_triplet_distance_differences - gamma))
            )
        ) / self.tau

        if self.batch_miner is None:
            loss = torch.mean(triplet_losses)
        else:
            flat_indexes = self.batch_miner(source_embeddings, embeddings, triplet_losses)
            if len(flat_indexes) > 0:
                loss = torch.mean(torch.take(triplet_losses, flat_indexes))
            else:
                if self.zero is None:
                    self.zero = torch.tensor(0.0).to(torch.float).to(embeddings.device)
                loss = self.zero

        return loss


class MultipleTripletTeachersLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.teacher_margin = args.teacher_margin
        self.margin = args.margin

    def forward(self, source_embeddings, embeddings, **kwargs):
        # Compute triplet weights according to the source embeddings.
        source_triplet_weights = detail.compute_multi_teacher_triplet_weights(source_embeddings, self.teacher_margin)

        # Calculate distances according to the learned embeddings.
        distances, _, _ = detail.calculate_similarities_pt(embeddings)

        # Calculate triplet losses based on distances.
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)
        triplet_losses = detail.hinge(triplet_distance_differences, self.margin)

        # Calculate the loss based on the triplet losses weighted by source embedding derived weights.
        loss = detail.safe_division(
            torch.sum(triplet_losses * source_triplet_weights), torch.sum(source_triplet_weights)
        )

        return loss
