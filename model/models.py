import itertools

import pretrainedmodels
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from tdml.model.model import Model


class ResNet50(Model):
    def __init__(self, args):
        super().__init__(args)

        self.model = pretrainedmodels.resnet50(num_classes=1000, pretrained="imagenet")

        if args.frozen_variables == "bn":
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        elif args.frozen_variables == "all":
            for module in self.model.modules():
                module.eval()
                module.train = lambda _: None

            for p in self.model.parameters():
                p.requires_grad = False

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
        self.final_nonlinearity_layers = nn.ModuleList(
            itertools.chain.from_iterable(
                [
                    [nn.Linear(args.embedding_size, args.embedding_size, device=torch.device("cuda:0")), nn.ReLU()]
                    for _ in range(args.num_nonlinearity_layers)
                ]
            )
        )
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, args.embedding_size)

    def forward(self, x, **kwargs):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for lb in self.layer_blocks:
            x = lb(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.model.last_linear(x)

        for nl in self.final_nonlinearity_layers:
            x = nl(x)

        x = F.normalize(x, dim=-1)

        return x


class InceptionV3(Model):
    def __init__(self, args):
        super().__init__(args)

        self.model = pretrainedmodels.bninception(num_classes=1000, pretrained="imagenet")

        if args.frozen_variables == "bn":
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        elif args.frozen_variables == "all":
            for module in self.model.modules():
                module.eval()
                module.train = lambda _: None

            for p in self.model.parameters():
                p.requires_grad = False

        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, args.embedding_size)

    def forward(self, x, **kwargs):
        x = self.model(x)

        x = F.normalize(x, dim=-1)

        return x


class ViT(Model):
    def __init__(self, args):
        super().__init__(args)

        self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        if args.frozen_variables == "all":
            for module in self.model.modules():
                module.eval()
                module.train = lambda _: None

            for p in self.model.parameters():
                p.requires_grad = False

        self.model.last_linear = nn.Linear(self.model.norm.normalized_shape[0], args.embedding_size)

    def forward(self, x, **kwargs):
        x = self.model(x)

        x = self.model.last_linear(x)

        x = F.normalize(x, dim=-1)

        return x


class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class HeadSwitch(nn.Module):
    def __init__(self, body, last_layer):
        super(HeadSwitch, self).__init__()
        self.body = body
        self.last_layer = last_layer
        self.norm = NormLayer()

    def forward(self, x, skip_head=False):
        x = self.body(x)
        if type(x) == tuple:
            x = x[0]
        if not skip_head:
            x = self.last_layer(x)
        else:
            x = self.norm(x)
        return x


class ViTS(Model):
    def __init__(self, args):
        super().__init__(args)

        body = timm.create_model("vit_small_patch16_224", pretrained=True)

        def freeze(model, num_block):
            def fr(m):
                for param in m.parameters():
                    param.requires_grad = False

            fr(model.patch_embed)
            fr(model.pos_drop)
            for i in range(num_block):
                fr(model.blocks[i])

        freeze(body, 0)
        bdim = 384

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        last_norm = nn.LayerNorm(bdim, elementwise_affine=False).cuda()
        last_layer = nn.Sequential(last_norm, nn.Linear(bdim, args.embedding_size), NormLayer())
        last_layer.apply(_init_weights)

        body.reset_classifier(0, "token")
        self.model = HeadSwitch(body, last_layer)
        self.model.cuda()

    def forward(self, x, **kwargs):
        return self.model(x)


class Psycho(Model):
    def __init__(self, args):
        super().__init__(args)

        initial_weights = torch.normal(mean=0.5, std=0.2, size=(args.num_instances, args.embedding_size))
        self.embedding_layer = nn.Embedding.from_pretrained(initial_weights, freeze=False)
        self.beta = args.beta

    def forward(self, x, **kwargs):
        x = self.embedding_layer(x)
        x = F.relu(x)

        x = F.normalize(x, dim=-1)

        return x

    def custom_regularized_parameters(self):
        return self.embedding_layer.parameters()

    def get_custom_regularization_loss(self):
        regularization_losses = [self.beta * torch.abs(p.data) for p in self.embedding_layer.parameters()]

        return torch.sum(torch.stack(regularization_losses))
