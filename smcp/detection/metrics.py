# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchmetrics
from torchvision.ops import nms, box_iou

from smcp.detection.bounding_box import cxcy_to_xy, gcxgcy_to_cxcy

# Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the SSD paper's results and other repos
def detect_objects2(
    predicted_locs: torch.Tensor, predicted_scores: torch.Tensor,
    priors_cxcy: torch.Tensor, min_score: float = 0.01, max_overlap: float = 0.45, top_k: int = 200
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decipher the 8732/24564 locations and class scores (output of ths SSD300/SSD512) to detect objects.

    For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

    :param predicted_locs: predicted locations/boxes w.r.t the 8732/24564 prior boxes, a tensor of dimensions (8732/24564, 4)
    :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (8732/24564, n_classes)
    :param min_score: minimum threshold for a box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :return: detections (boxes, labels, and scores)
    """
    n_priors = priors_cxcy.size(0)
    n_classes = predicted_scores.size(-1)
    device = predicted_scores.device

    assert n_priors == predicted_locs.size(0) == predicted_scores.size(0)

    # Decode object coordinates from the form we regressed predicted boxes to
    decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs, priors_cxcy))  # (8732/24564, 4), these are fractional pt. coordinates

    # Lists to store boxes and scores for this image
    image_boxes = list()
    image_labels = list()
    image_scores = list()

    # Check for each class
    for c in range(1, n_classes):
        # Keep only predicted boxes and scores where scores for this class are above the minimum score
        class_scores = predicted_scores[:, c]  # (8732)
        score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
        n_above_min_score = score_above_min_score.sum().item()
        if n_above_min_score == 0:
            continue
        class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732/24564
        class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

        kept_indices = nms(class_decoded_locs, class_scores, max_overlap)

        image_boxes.append(class_decoded_locs[kept_indices])
        image_labels.append(torch.tensor(torch.numel(kept_indices) * [c], dtype=torch.long, device=device))
        image_scores.append(class_scores[kept_indices])

    # If no object in any class is found, store a placeholder for 'background'
    if len(image_boxes) == 0:
        image_boxes.append(torch.tensor([[0., 0., 1., 1.]], dtype=torch.float32, device=device))
        image_labels.append(torch.tensor([0], dtype=torch.long, device=device))
        image_scores.append(torch.tensor([0.], dtype=torch.float32, device=device))

    # Concatenate into single tensors
    image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
    image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
    image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
    n_objects = image_scores.size(0)

    # Keep only the top k objects
    if n_objects > top_k:
        image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
        image_scores = image_scores[:top_k]  # (top_k)
        image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
        image_labels = image_labels[sort_ind][:top_k]  # (top_k)

    return image_boxes, image_labels, image_scores

def calculate_mAP(
    preds: Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]],
    target: Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]],
    num_classes: int, iou_threshold: float, recall_thresholds: torch.Tensor,
) -> Tuple[float, List[float]]:
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param preds: tuple of
        det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
        det_labels: list of tensors, one tensor for each image containing detected objects' labels
        det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param target: tuple of
        true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
        true_labels: list of tensors, one tensor for each image containing actual objects' labels
        true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: mean average precision (mAP), list of average precisions for all classes
    """
    det_boxes, det_labels, det_scores = preds
    true_boxes, true_labels, true_difficulties = target

    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) \
        == len(true_labels) == len(true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images

    device = det_boxes[0].device

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))

    true_images = torch.tensor(true_images, dtype=torch.long, device=device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))

    det_images = torch.tensor(det_images, dtype=torch.long, device=device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((num_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, num_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (~true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        cntr = torch.arange(len(true_class_boxes), dtype=torch.long, device=device)
        true_class_boxes_detected = torch.zeros_like(true_class_difficulties, dtype=torch.uint8)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)

        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros_like(det_class_images, dtype=torch.float)  # (n_class_detections)
        false_positives = torch.zeros_like(det_class_images, dtype=torch.float)  # (n_class_detections)

        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)

            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = box_iou(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = cntr[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > iou_threshold:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        precisions = torch.zeros_like(recall_thresholds, dtype=torch.float)  # (11)

        # Remove zigzags
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.

        # Calculate average precision for each class
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean()

    return mean_average_precision.item(), average_precisions.tolist()

class SSDDetectionMAP(torchmetrics.Metric):
    priors_cxcy: torch.Tensor
    map_thresholds: torch.Tensor
    det_boxes: List[torch.Tensor]
    det_labels: List[torch.Tensor]
    det_scores: List[torch.Tensor]
    true_boxes: List[torch.Tensor]
    true_labels: List[torch.Tensor]
    true_difficulties: List[torch.Tensor]

    def __init__(self, label_map: Dict[str, int], priors_cxcy: torch.Tensor, iou_threshold: float = 0.5, **kwargs):
        super().__init__(compute_on_step=False, dist_sync_on_step=False, **kwargs)

        self.rev_label_map = { v: k for k, v in label_map.items() }
        self.register_buffer("priors_cxcy", priors_cxcy)
        self.iou_threshold = iou_threshold

        self.register_buffer("map_thresholds", torch.linspace(0, 1, 11))  # VOC2007 style

        self.add_state("det_boxes", default=[], dist_reduce_fx=None)
        self.add_state("det_labels", default=[], dist_reduce_fx=None)
        self.add_state("det_scores", default=[], dist_reduce_fx=None)
        self.add_state("true_boxes", default=[], dist_reduce_fx=None)
        self.add_state("true_labels", default=[], dist_reduce_fx=None)
        self.add_state("true_difficulties", default=[], dist_reduce_fx=None)

    def update(self, preds: Tuple[torch.Tensor, torch.Tensor], target: Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]) -> None:
        """
        :param preds: tuple of
            predicted_locs: predicted locations/boxes w.r.t the 8732/24564 prior boxes, a tensor of dimensions (N, 8732/24564, 4)
            predicted_logits: class logits for each of the encoded locations/boxes, a tensor of dimensions (N, 8732/24564, n_classes)
        :param target: tuple of
            boxes: true object bounding boxes in boundary coordinates, a list of N tensors
            labels: true object labels, a list of N tensors
            difficulties: true object difficulties, a list of N tensors
        """
        predicted_locs_batch, predicted_logits_batch = preds

        batch_size, num_prior_boxes, num_classes = predicted_logits_batch.shape
        predicted_scores_batch = F.softmax(predicted_logits_batch, dim=2)  # (N, num_prior_boxes, n_classes)

        for i in range(batch_size):
            pred_locs = predicted_locs_batch[i, ...]  # (num_prior_boxes, 4)
            pred_scores = predicted_scores_batch[i, ...]  # (num_prior_boxes, num_classes)

            # Detect objects in SSD output
            det_boxes, det_labels, det_scores = detect_objects2(pred_locs, pred_scores, self.priors_cxcy)  # (n_detections, 4), (n_detections), (n_detections)

            self.det_boxes.append(det_boxes)
            self.det_labels.append(det_labels)
            self.det_scores.append(det_scores)

        # Save target information
        boxes_batch, labels_batch, difficulties_batch = target
        self.true_boxes.extend(boxes_batch)
        self.true_labels.extend(labels_batch)
        self.true_difficulties.extend(difficulties_batch)

    def compute(self) -> Dict[str, torch.Tensor]:
        preds = (self.det_boxes, self.det_labels, self.det_scores)
        targets = (self.true_boxes, self.true_labels, self.true_difficulties)

        mAP, class_mAPs = calculate_mAP(preds, targets, len(self.rev_label_map), self.iou_threshold, self.map_thresholds)

        return {
            "map": mAP,
            **{ f"map_{self.rev_label_map[c + 1]}" : v for c, v in enumerate(class_mAPs) }
        }
