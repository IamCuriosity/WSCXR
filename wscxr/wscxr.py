"""detection methods."""
import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from wscxr import common
from wscxr import metrics
from wscxr import utils
import numpy as np
import sklearn


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):

        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes

        self.layers = torch.nn.Sequential()
        _in = None
        _out = None

        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):

        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x



class WSCXR(torch.nn.Module):
    def __init__(self, device):
        """anomaly detection class."""
        super(WSCXR, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            layers_to_extract_from,

            device,
            input_shape,

            pretrain_embed_dimension,  # 1536
            target_embed_dimension,  # 1536

            prototypes=None,
            dsc_save_path='',

            patchsize=3,  # 3
            patchstride=1,

            meta_epochs=1,  # 40
            gan_epochs=1,  # 4

            dsc_layers=2,  # 2
            dsc_hidden=None,  # 1024
            dsc_margin=.8,  # .5
            dsc_lr=0.0002,
            lr=1e-3,
            logger=None,

            **kwargs,
    ):

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from

        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )

        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )

        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension

        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.meta_epochs = meta_epochs
        self.lr = lr

        # Discriminator

        self.dsc_lr = dsc_lr  # 0.0002
        self.gan_epochs = gan_epochs  # 4

        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)

        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_margin = dsc_margin  # 0.5

        self.logger = logger

        if prototypes is not None:
            self.prototypes = torch.from_numpy(prototypes).to(self.device)

        self.dsc_save_path = dsc_save_path


    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)


    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        _ = self.forward_modules["feature_aggregator"].eval()

        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](
            features)  # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features)  # further pooling
        return features, patch_shapes



    def train_(self, training_data, test_data):

        best_record = None

        for i_mepoch in range(self.meta_epochs):

            self._train_discriminator(training_data, i_mepoch)

            torch.cuda.empty_cache()

            scores, segmentations, labels_gt, image_paths = self.predict(test_data)

            scores = np.array(scores)
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores)).tolist()

            results = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt)

            if best_record is None:
                best_record = [results["auroc"], results['acc'], results['f1']]
                torch.save(self.discriminator.state_dict(), self.dsc_save_path)
            else:
                if np.mean([results["auroc"], results['acc'], results['f1']]) > np.mean(best_record):
                    best_record = [results["auroc"], results['acc'], results['f1']]
                    torch.save(self.discriminator.state_dict(), self.dsc_save_path)

            if self.logger is not None:
                self.logger.info(
                    "image auroc: {:.4f}, acc: {:.4f}, f1: {:.4f},".format(results["auroc"], results['acc'],
                                                                           results['f1']))
                self.logger.info("best image auroc: {:.4f}, best acc: {:.4f}, best f1: {:.4f},".format(best_record[0],
                                                                                                       best_record[1],
                                                                                                       best_record[2]))
        return best_record


    def _train_discriminator(self, input_data, epoch):

        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        self.discriminator.train()

        i_iter = 0

        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):

                all_loss = []

                for data_item in input_data:
                    self.dsc_opt.zero_grad()

                    i_iter += 1
                    img = data_item["image"]

                    img = img.to(torch.float).to(self.device)

                    true_feats = self._embed(img)[0]
                    fake_feats = self.create_fake_feats(true_feats)

                    scores = self.discriminator(torch.cat([true_feats, fake_feats]))

                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(fake_feats):]

                    th = self.dsc_margin

                    true_loss = torch.clip(-true_scores + th, min=0)  # torch.clamp
                    fake_loss = torch.clip(fake_scores + th, min=0)

                    loss = true_loss.mean() + fake_loss.mean()

                    loss.backward()

                    self.dsc_opt.step()

                    loss = loss.detach().cpu()
                    all_loss.append(loss.item())

                all_loss = sum(all_loss) / len(input_data)

                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)


    def create_fake_feats(self, true_feats):
        normal_Len = true_feats.shape[0]
        abnormal_Len = self.prototypes.shape[0]
        index = np.random.choice(list(range(abnormal_Len)), size=(normal_Len), replace=True)
        weights = (torch.rand(size=(normal_Len, 1)) * 0.9 + 0.1).to(self.device)  # [0.1,1.0]
        return true_feats * (1.0 - weights) + self.prototypes[index] * weights


    def create_fake_feats_overlapping(self, true_feats):
        normal_Len = true_feats.shape[0]
        abnormal_Len = self.prototypes.shape[0]
        k = math.ceil(normal_Len / abnormal_Len)
        fake_feats = self.prototypes.repeat((k, 1))
        return fake_feats


    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        img_paths = []
        scores = []
        masks = []
        labels_gt = []

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:

            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])

                _scores, _masks = self._predict(image)

                for score, mask, is_anomaly in zip(_scores, _masks, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)

        return scores, masks, labels_gt, img_paths

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        self.discriminator.eval()

        with torch.no_grad():
            features, patch_shapes = self._embed(images,
                                                 provide_patch_shapes=True,
                                                 )
            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            patch_scores = image_scores = -self.discriminator(features)

            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )

            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )

            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return list(image_scores), list(masks)


    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")


    def save_to_path(self, save_path: str, prepend: str = ""):
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                                s + 2 * padding - 1 * (self.patchsize - 1) - 1
                        ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
