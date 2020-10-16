# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
import numpy as np
from typing import List
import torch
from fvcore.nn import giou_loss, sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from ..anchor_generator import build_anchor_generator
from ..backbone import build_backbone
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..postprocessing import detector_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["ClassifierML"]


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def one_hot_embedding(eye ,labels):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    one_hot_list = []
    for label in labels:
        one_hot_list.append(eye[labels].sum(dim=0))
    return torch.stack(one_hot_list)


@META_ARCH_REGISTRY.register()
class ClassifierML(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        backbone,
        head,
        head_in_features,
        num_classes,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            head_in_features (Tuple[str]): Names of the input feature maps to be used in head
            num_classes (int): number of classes. Used to label background proposals.

            # Loss parameters:
            focal_loss_alpha (float): focal_loss_alpha
            focal_loss_gamma (float): focal_loss_gamma
            
            # Input parameters
            pixel_mean (Tuple[float]):
                Values to be used for image normalization (BGR order).
                To train on images of different number of channels, set different mean & std.
                Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
            pixel_std (Tuple[float]):
                When using pre-trained models in Detectron1 or any MSRA models,
                std has been absorbed into its conv1 weights, so the std needs to be set 1.
                Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
            vis_period (int):
                The period (in terms of steps) for minibatch visualization at train time.
                Set to 0 to disable.
            input_format (str): Whether the model needs RGB, YUV, HSV etc.
        """
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features
        self.num_classes = num_classes

        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        self.eye = torch.eye(num_classes, device=self.device)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.CLASSIFIER_ML.IN_FEATURES]
        head = ClassifierMLHead(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.CLASSIFIER_ML.NUM_CLASSES,
            "head_in_features": cfg.MODEL.CLASSIFIER_ML.IN_FEATURES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.CLASSIFIER_ML.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.CLASSIFIER_ML.FOCAL_LOSS_GAMMA,
            # Vis parameters
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        show image and CAM result

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)

        #change here to show CAM result
        """
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)
        """

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        pred_logits = self.head(features)
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_labels = [torch.unique(inst.gt_classes) for inst in gt_instances]

            #change here to get gt_label from gt_instances and loss function
            losses = self.losses(pred_logits, gt_labels)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(pred_logits)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                processed_results.append({"classifier": results_per_image})
            return processed_results

    def losses(self, pred_logits, gt_labels):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_labels)
        gt_labels_target = one_hot_embedding(self.eye, gt_labels)  # (N, R)
        gt_labels_target = gt_labels_target.to(pred_logits[0].device)

        # classification and regression loss
        loss_cls = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1).squeeze(1),
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        return {
            "loss_cls": loss_cls ,
        }

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps (sum(Hi * Wi * A)).
                Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.
            list[Tensor]:
                i-th element is a Rx4 tensor, where R is the total number of anchors across
                feature maps. The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as foreground.
        """
        anchors = Boxes.cat(anchors)  # Rx4

        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def inference(self, anchors, pred_logits, pred_anchor_deltas, image_sizes):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, pred_logits_per_image, deltas_per_image, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, anchors, box_cls, box_delta, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = torch.nonzero(keep_idxs, as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_thresh)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class ClassifierMLHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
        self,
        *,
        input_shape: List[ShapeSpec],
        num_classes,
        conv_dims: List[int],
        prior_prob,
        norm
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
        """
        if norm == "BN" or norm == "SyncBN":
            logger = logging.getLogger(__name__)
            logger.warn("Shared norm does not work well for BN, SyncBN, expect poor results")

        super().__init__()

        cls_subnet = []
        for in_channels, out_channels in zip([input_shape[0].channels] + conv_dims, conv_dims):
            cls_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                cls_subnet.append(get_norm(norm, out_channels))
            cls_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.cls_score = nn.Conv2d(
            conv_dims[-1], num_classes, kernel_size=1, stride=1, padding=0
        )

        # Initialization
        for modules in [self.cls_score]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,
            "prior_prob": cfg.MODEL.RETINANET.PRIOR_PROB,
            "conv_dims": [input_shape[0].channels] * cfg.MODEL.RETINANET.NUM_CONVS,
            "norm": cfg.MODEL.RETINANET.NORM,
        }

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
        """
        logits = []
        for feature in features:
            feature = self.cls_subnet(feature)
            feature = F.adaptive_avg_pool2d(feature, (1, 1))
            logits.append(self.cls_score(feature))
        return logits
