import cv2
import numpy as np
from vlm.coco_classes import COCO_CLASSES
from vlm.detector.yolov7 import YOLOv7Client
from vlm.segmentor.sam import MobileSAMClient
from vlm.detector.grounding_dino import GroundingDINOClient
from vlm.itm.blip2itm import BLIP2ITMClient
from vlm.utils.get_itm_message import get_itm_message

yolov7_detector = YOLOv7Client(port=12184)
blip2_itm = BLIP2ITMClient(port=12182)
sam_segmentor = MobileSAMClient(port=12183)
dino_detector = GroundingDINOClient(port=12181)


def _norm_label(text: str) -> str:
    t = text.strip().lower().replace("_", " ").replace("-", " ")
    t = " ".join(t.split())
    alias = {
        "sofa": "couch",
        "settee": "couch",
        "tv": "television",
        "tv monitor": "television",
    }
    if t in alias:
        t = alias[t]
    if t.endswith("s") and len(t) > 3:
        t = t[:-1]
    return t


def _is_label_match(label_detected: str, candidates: list) -> bool:
    ld = _norm_label(label_detected)
    for c in candidates:
        lc = _norm_label(c)
        if ld == lc or ld in lc or lc in ld:
            return True
    return False
# def _is_label_match(label_detected: str, candidates: list) -> bool:
#     ld = _norm_label(label_detected)
#     for c in candidates:
#         lc = _norm_label(c)
#         if ld == lc:
#             return True
#     return False


def get_segmentation(segmented_img, idx, detections, img, label, score, color):
    object_mask = np.zeros((480, 640), dtype=np.uint8)
    bbox_denorm = detections.boxes[idx] * np.array(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    )
    x1, y1, x2, y2 = [int(v) for v in bbox_denorm]
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img.shape[0] * img.shape[1]

    if bbox_area / img_area < 0.99:
        object_mask = sam_segmentor.segment_bbox(img, bbox_denorm.tolist())
        mask_pixels = int(np.count_nonzero(object_mask))
        contours, _ = cv2.findContours(
            object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            cv2.drawContours(segmented_img, [contour], 0, color, 4)

        cv2.rectangle(
            segmented_img,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label_text = f"{label} ({score:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
        )
        label_x = x1
        label_y = y1 - text_height
        cv2.rectangle(
            segmented_img,
            (label_x, label_y - 30),
            (label_x + text_width, label_y + text_height),
            color,
            2,
        )
        cv2.putText(
            segmented_img,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            1,
        )
    else:
        pass

    return segmented_img, object_mask

def get_object(right_label, img, cfg, similar_answer, use_label_filter: bool = True):
    score_list = []
    object_masks_list = []
    segmented_img = img.copy()
    label_list = []
    coco_label = []
    dino_label = []
    right_label_list = []
    all_answer = []
    if use_label_filter:
        right_label_list = list(map(str.strip, right_label.split('|')))
        # print(f"right_label_list: {right_label_list}")
        all_answer = right_label_list + similar_answer
        for label in all_answer:
            if label in COCO_CLASSES:
                coco_label.append(label)
            else:
                dino_label.append(label)

        if any(item in dino_label for item in right_label_list):
            dino_label = all_answer
            coco_label = []
            for label in right_label_list:
                if label in COCO_CLASSES:
                    coco_label.append(label)
    else:
        coco_label = list(COCO_CLASSES)

    if coco_label:
        detections = yolov7_detector.predict(img, agnostic_nms=cfg.yolo.agnostic_nms, 
                                            conf_thres=cfg.yolo.confidence_threshold_yolo, iou_thres=cfg.yolo.iou_threshold_yolo)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if not use_label_filter:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(0, 255, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(1)
            elif _is_label_match(label_detected, right_label_list):
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(0)
            elif _is_label_match(label_detected, coco_label):
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(0, 255, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(1)

        # for idx in range(len(detections.logits)):
        #     label_detected = detections.phrases[idx]
        #     score = detections.logits[idx].item()
            
        #     # 强制让所有 YOLO 发现的物体都进入 CLIP 的视野
        #     # 不再使用 if _is_label_match(...) 过滤掉物体
        #     segmented_img, object_mask = get_segmentation(
        #         segmented_img, idx, detections, img, label_detected, score, 
        #         color=(255, 0, 0) if _is_label_match(label_detected, right_label_list) else (0, 255, 0)
        #     )
        #     score_list.append(score)
        #     object_masks_list.append(object_mask)
            
        #     # 记录它是“目标类”还是“普通物体”
        #     is_target = 0 if _is_label_match(label_detected, right_label_list) else 1
        #     label_list.append(is_target)

    if dino_label and use_label_filter:
        caption = ' '.join(f'{item}.  ' for item in dino_label)
        detections = dino_detector.predict(img, caption=caption, 
                                        box_threshold=cfg.groundingDINO.confidence_threshold_dino, text_threshold=cfg.groundingDINO.text_threshold)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if _is_label_match(label_detected, right_label_list):
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(0)

            elif _is_label_match(label_detected, dino_label):
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(0, 255, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(1)

    return segmented_img, score_list, object_masks_list, label_list

def get_object_with_itm(label, img, cfg):
    score_list = []
    object_masks_list = []
    cosine_list = []
    itm_score_list = []
    segmented_img = img.copy()
    if label in COCO_CLASSES:
        detections = yolov7_detector.predict(img, agnostic_nms=cfg.yolo.agnostic_nms,
                                             conf_thres=cfg.yolo.confidence_threshold_yolo, iou_thres=cfg.yolo.iou_threshold_yolo)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if detections.phrases[idx] == label:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                img_detected = crop_and_expand_box(img, detections, idx)
                # cv2.imshow(f"img_detected{idx}", img_detected)
                cosine, itm_score = get_itm_message(img_detected, label)
                pass
                score_list.append(score)
                object_masks_list.append(object_mask)
                cosine_list.append(cosine)
                itm_score_list.append(itm_score)

    else:
        detections = dino_detector.predict(img, caption=label, 
                                           box_threshold=cfg.groundingDINO.confidence_threshold_dino, text_threshold=cfg.groundingDINO.text_threshold)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if score > cfg.groundingDINO.confidence_threshold_dino:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                img_detected = crop_and_expand_box(img, detections, idx)
                # cv2.imshow(f"img_detected{idx}", img_detected)
                cosine, itm_score = get_itm_message(img_detected, label)
                pass
                cosine_list.append(cosine)
                itm_score_list.append(itm_score)
    
    return segmented_img, score_list, object_masks_list, cosine_list, itm_score_list


def crop_and_expand_box(img, detections, idx, expand_pixels=0.4):
    # Get bounding box coordinates in [x_min, y_min, x_max, y_max] format
    x_min, y_min, x_max, y_max = detections.boxes[idx]
    x_min = int(x_min * img.shape[1])
    y_min = int(y_min * img.shape[0])
    x_max = int(x_max * img.shape[1])
    y_max = int(y_max * img.shape[0])

    # Expand the box outward; clamp to image boundaries
    x_min = max(int(x_min*(1-expand_pixels)), 0)
    y_min = max(int(y_min*(1-expand_pixels)), 0)
    x_max = min(int(x_max*(1+expand_pixels)), img.shape[1] - 1)
    y_max = min(int(y_max*(1+expand_pixels)), img.shape[0] - 1)

    # Crop the image to keep only the box region
    img_detected = img[y_min:y_max+1, x_min:x_max+1]

    return img_detected

