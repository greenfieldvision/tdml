import torch

import tdml.data.dataset_builders as dataset_builders
import tdml.data.preprocessing as preprocessing
import tdml.model.metrics as metrics
import tdml.model.losses_class as losses_class
import tdml.model.losses_source_embedding as losses_source_embedding
import tdml.model.losses_triplet as losses_triplet
import tdml.model.models as models


def make_dataset_builder(args):
    dataset_builder_cls = {
        "ThingsTImg": dataset_builders.ThingsTImg,
        "ThingsTInd": dataset_builders.ThingsTInd,
        "ThingsSE": dataset_builders.ThingsSE,
        "IHSJCTImg": dataset_builders.IHSJCTImg,
        "IHSJCTInd": dataset_builders.IHSJCTInd,
        "IHSJCSE": dataset_builders.IHSJCSE,
        "YummlyTImg": dataset_builders.YummlyTImg,
        "YummlyTInd": dataset_builders.YummlyTInd,
        "YummlySE": dataset_builders.YummlySE,
        "CUBSEC": dataset_builders.CUBSEC,
        "CARSSEC": dataset_builders.CARSSEC,
        "SOPSEC": dataset_builders.SOPSEC,
    }[args.name]

    args.source_embeddings_id = args.get("source_embeddings_id", None)

    return dataset_builder_cls(args)


def make_preprocessing(dataset_builder, args):
    preprocessing_cls = {
        "ResNetImagePreprocessing": preprocessing.ResNetImagePreprocessing,
        "ViTImagePreprocessing": preprocessing.ViTImagePreprocessing,
        "IndexRemapPreprocessing": preprocessing.IndexRemapPreprocessing,
    }[args.name]

    return preprocessing_cls(dataset_builder, args)


def make_model(dataset_builder, args):
    model_cls = {
        "ResNet50": models.ResNet50,
        "Inception": models.InceptionV3,
        "ViT": models.ViT,
        "ViTS": models.ViTS,
        "Psycho": models.Psycho,
    }[args.name]

    if args.name == "Psycho":
        args.num_instances = dataset_builder.num_instances

    args.batch_mining = args.get("batch_mining", None)

    return model_cls(args)


def make_loss_and_metric(dataset_builder, args):
    loss_computer, metric_computer = None, None

    directed = dataset_builder.get_triplet_directedness()

    # original metric learning losses
    if args.loss_type in {
        "contrastive",
        "facenet",
        "infonce",
        "multi_similarity",
        "multi_similarity_original",
        "sampling_matters",
    }:
        loss_cls = {
            "contrastive": losses_class.ContrastiveLoss,
            "facenet": losses_class.FacenetLoss,
            "infonce": losses_class.InfoNCELoss,
            "multi_similarity": losses_class.MultiSimilarityLoss,
            "multi_similarity_original": losses_class.MultiSimilarityLossOriginal,
            "sampling_matters": losses_class.SamplingMattersLoss,
        }[args.loss_type]
        loss_computer = loss_cls(args)

    # reduced versions of metric learning losses
    if args.loss_type in {"double_contrastive", "double_margin", "double_multisimilarity", "ste", "triplet"}:
        loss_cls = {
            "double_contrastive": losses_triplet.ContrastiveLoss,
            "double_margin": losses_triplet.DoubleMarginLoss,
            "double_multisimilarity": losses_triplet.MultiSimilarityLoss,
            "ste": losses_triplet.InfoNCELoss,
            "triplet": losses_triplet.TripletMarginLoss,
        }[args.loss_type]
        loss_computer = loss_cls(directed, args)

    # knowledge distillation losses
    if args.loss_type in {
        "rkd",
        "relaxed_contrastive",
        "relaxed_infonce",
        "relaxed_triplet",
        "relaxed_facenet",
        "relaxed_multisimilarity",
        "stmr",
        "mtt",
    }:
        loss_cls = {
            "rkd": losses_source_embedding.RKDLoss,
            "relaxed_contrastive": losses_source_embedding.RelaxedContrastiveLoss,
            "relaxed_infonce": losses_source_embedding.RelaxedInfoNCELoss,
            "relaxed_triplet": losses_source_embedding.RelaxedTripletMarginLoss,
            "relaxed_facenet": losses_source_embedding.RelaxedFacenetLoss,
            "relaxed_multisimilarity": losses_source_embedding.RelaxedMultiSimilarityLoss,
            "stmr": losses_source_embedding.SoftTripletMarginRegressionLoss,
            "mtt": losses_source_embedding.MultipleTripletTeachersLoss,
        }[args.loss_type]
        loss_computer = loss_cls(args)

    metric_name, _ = args.metric_mode.split("/")
    if metric_name == "mP@1":
        metric_computer = metrics.MetricsClassSupervision()
    elif metric_name == "FCT":
        metric_computer = metrics.MetricsTripletSupervision(directed)
    elif metric_name == "fraction_same_triplet_relation_thresholded":
        metric_computer = metrics.MetricsSourceEmbeddingSupervision()

    return loss_computer, metric_computer


def make_optimizer(model, args):
    regular_regularized_parameters = [
        p for p in model.parameters() if p not in set(model.custom_regularized_parameters())
    ]
    optimizer_parameters = dict(
        params=[
            {"params": regular_regularized_parameters},
            {"params": model.custom_regularized_parameters(), "weight_decay": 0.0},
        ],
        weight_decay=args.weight_decay,
        lr=args.initial_learning_rate,
    )

    optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[args.optimizer_type]

    return optimizer_cls(**optimizer_parameters)
