from __future__ import annotations

import dataclasses
import threading
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from utils.face_pipeline import FaceObservation


DEFAULT_SAM_KEYWORDS: Tuple[str, ...] = (
    "sam",
    "\uc815\ubc00",
    "bangs",
    "fringe",
    "sideburns",
    "undercut",
    "fade",
    "baby hair",
    "flyaway",
    "hairline",
    "temple",
    "edge",
    "\uc55e\uba38\ub9ac",
    "\uad6c\ub808\ub098\ub8e3",
    "\uc794\uba38\ub9ac",
    "\ud5e4\uc5b4\ub77c\uc778",
    "\uc5e3\uc9c0",
    "\uad00\uc790\ub180\uc774",
)

SAM2_PROMPT_ALIASES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("bangs", "fringe", "\uc55e\uba38\ub9ac"), "hair bangs on forehead"),
    (("sideburns", "\uad6c\ub808\ub098\ub8e3"), "sideburn hair near ear"),
    (("undercut",), "hair undercut short sides"),
    (("fade",), "hair fade short sides"),
    (
        ("baby hair", "flyaway", "wispy", "\uc794\uba38\ub9ac"),
        "baby hair near hairline",
    ),
    (("temple", "\uad00\uc790\ub180\uc774"), "hair near temple"),
    (("hairline", "edge", "\ud5e4\uc5b4\ub77c\uc778", "\uc5e3\uc9c0"), "hairline edge"),
)


def build_sam2_prompt_hint(text_prompt: str) -> str:
    prompt_lower = str(text_prompt or "").lower()
    matched_prompts: List[str] = [
        "hair",
        "hair bangs on forehead",
        "baby hair near hairline",
        "sideburn hair near ear",
        "hairline edge",
    ]
    for aliases, prompt_hint in SAM2_PROMPT_ALIASES:
        if any(alias in prompt_lower for alias in aliases):
            matched_prompts.append(prompt_hint)
    return ", ".join(dict.fromkeys(matched_prompts))


@dataclasses.dataclass
class SegmentationResult:
    parsing_mask: Any
    hair_mask: Any
    source: str
    sam_refined: bool
    sam_prompt: Optional[str] = None


class SAM2Refiner:
    def __init__(self, predictor_factory: Optional[Callable[[], Any]]) -> None:
        self.predictor_factory = predictor_factory
        self._predictor = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.predictor_factory is not None


    def refine_mask_with_sam2(
        self,
        image_rgb,
        face_observation: FaceObservation,
        base_hair_mask,
        prompt: str = "",
    ):
        import numpy as np
        import torch
        import torch.nn.functional as F

        if not self.available:
            return base_hair_mask

        prompt_hint = build_sam2_prompt_hint(prompt)

        with self._lock:
            predictor = self._get_predictor()
            bbox = np.array(
                self._expanded_box(face_observation.bbox, image_rgb.shape[1], image_rgb.shape[0]),
                dtype=np.float32,
            )
            positive_points = np.array(
                [
                    face_observation.landmarks_px[10],
                    face_observation.landmarks_px[67],
                    face_observation.landmarks_px[297],
                ],
                dtype=np.float32,
            )
            negative_points = np.array(
                [
                    face_observation.landmarks_px[234],
                    face_observation.landmarks_px[454],
                    face_observation.landmarks_px[152],
                ],
                dtype=np.float32,
            )

            if callable(predictor) and not (hasattr(predictor, "set_image") and hasattr(predictor, "predict")):
                refined_np = self._predict_with_callable(
                    predictor,
                    image_rgb=image_rgb,
                    bbox=bbox,
                    positive_points=positive_points,
                    negative_points=negative_points,
                    prompt_hint=prompt_hint,
                )
            else:
                refined_np = self._predict_with_object(
                    predictor,
                    image_rgb=image_rgb,
                    bbox=bbox,
                    positive_points=positive_points,
                    negative_points=negative_points,
                    prompt_hint=prompt_hint,
                )
            refined_mask = torch.from_numpy(self._to_numpy_mask(refined_np))

        refined_mask = refined_mask.to(base_hair_mask.device).unsqueeze(0).unsqueeze(0)
        if tuple(refined_mask.shape[-2:]) != tuple(base_hair_mask.shape[-2:]):
            refined_mask = F.interpolate(
                refined_mask.float(),
                size=base_hair_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            refined_mask = refined_mask.to(base_hair_mask.dtype)
        return refined_mask.clamp_(0.0, 1.0)

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            self._predictor = self.predictor_factory()
        return self._predictor

    @staticmethod
    def _to_numpy_mask(mask):
        import numpy as np

        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        return np.asarray(mask, dtype="float32")

    def _predict_with_callable(
        self,
        predictor: Callable[..., Any],
        *,
        image_rgb,
        bbox,
        positive_points,
        negative_points,
        prompt_hint: str,
    ):
        kwargs = {
            "image_rgb": image_rgb,
            "box": bbox,
            "positive_points": positive_points,
            "negative_points": negative_points,
        }
        if prompt_hint:
            kwargs["text_prompt"] = prompt_hint
        try:
            return predictor(**kwargs)
        except TypeError:
            kwargs.pop("text_prompt", None)
            return predictor(**kwargs)

    def _predict_with_object(
        self,
        predictor: Any,
        *,
        image_rgb,
        bbox,
        positive_points,
        negative_points,
        prompt_hint: str,
    ):
        prompt_mask = self._predict_text_mask(predictor, image_rgb=image_rgb, prompt_hint=prompt_hint)
        if prompt_mask is not None:
            return prompt_mask
        if hasattr(predictor, "set_image") and hasattr(predictor, "predict"):
            return self._predict_point_mask(
                predictor,
                image_rgb=image_rgb,
                bbox=bbox,
                positive_points=positive_points,
                negative_points=negative_points,
                prompt_hint=prompt_hint,
            )
        raise TypeError("SAM3 predictor API is not supported.")

    def _predict_text_mask(self, predictor: Any, *, image_rgb, prompt_hint: str):
        import numpy as np
        import PIL.Image

        if not prompt_hint or not hasattr(predictor, "set_image") or not hasattr(predictor, "set_text_prompt"):
            return None
        image_input = PIL.Image.fromarray(image_rgb)
        try:
            state = predictor.set_image(image_input)
        except Exception:
            state = predictor.set_image(image_rgb)
        output = None
        for invocation in (
            lambda: predictor.set_text_prompt(state=state, prompt=prompt_hint),
            lambda: predictor.set_text_prompt(prompt=prompt_hint, state=state),
            lambda: predictor.set_text_prompt(state, prompt_hint),
            lambda: predictor.set_text_prompt(prompt_hint),
        ):
            try:
                output = invocation()
                break
            except TypeError:
                continue
        if not isinstance(output, Mapping):
            return None
        masks = output.get("masks")
        if masks is None or len(masks) == 0:
            return None
        scores = output.get("scores")
        best_index = 0 if scores is None or len(scores) == 0 else int(np.asarray(self._to_numpy_mask(scores)).argmax())
        return self._to_numpy_mask(masks[best_index])

    def _predict_point_mask(
        self,
        predictor: Any,
        *,
        image_rgb,
        bbox,
        positive_points,
        negative_points,
        prompt_hint: str,
    ):
        import numpy as np

        predictor.set_image(image_rgb)
        point_coords = np.concatenate([positive_points, negative_points], axis=0)
        point_labels = np.concatenate(
            [
                np.ones(len(positive_points), dtype=np.int32),
                np.zeros(len(negative_points), dtype=np.int32),
            ]
        )
        prediction = None
        for extra_kwargs in (
            {"text_prompt": prompt_hint} if prompt_hint else {},
            {"prompt": prompt_hint} if prompt_hint else {},
            {},
        ):
            try:
                prediction = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=bbox[None, :],
                    multimask_output=False,
                    **extra_kwargs,
                )
                break
            except TypeError:
                continue
        if prediction is None:
            raise TypeError("SAM3 predictor.predict() does not support the configured inputs.")
        if isinstance(prediction, Mapping):
            masks = prediction.get("masks")
        else:
            masks = prediction[0]
        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM3 predictor returned no masks.")
        return self._to_numpy_mask(masks[0])

    @staticmethod
    def _expanded_box(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
        left, top, right, bottom = bbox
        box_width = right - left
        box_height = bottom - top
        return (
            max(int(left - box_width * 0.15), 0),
            max(int(top - box_height * 0.25), 0),
            min(int(right + box_width * 0.15), width - 1),
            # 하단을 60%까지 확장: 긴 머리카락이 어깨까지 내려오는 경우 포착
            min(int(bottom + box_height * 0.60), height - 1),
        )


class SAMMaskingPipeline:
    """Two-stage masking pipeline: BiSeNet base → SAM3 refine.

    Stage 1 (BiSeNet): Fast per-class hair segmentation on the aligned face crop.
    Stage 2 (SAM3):    Point- and text-prompted refinement using MediaPipe landmarks.
    """

    def __init__(
        self,
        seg: Any,
        device: Any,
        use_autocast: bool,
        predictor_factory: Optional[Callable[[], Any]],
        hair_class_index: int,
        autocast_factory: Callable[[Any, bool], Any],
    ) -> None:
        self.seg = seg
        self.device = device
        self.use_autocast = use_autocast
        self.sam2_refiner = SAM2Refiner(predictor_factory)
        self.hair_class_index = hair_class_index
        self.autocast_factory = autocast_factory

    def segment(
        self,
        image_tensor,
        image_rgb,
        face_observation: FaceObservation,
        prompt: str = "",
        alignment_transform: Optional[Any] = None,
    ) -> SegmentationResult:
        """Run the full SAM2 → MediaPipe → HairCLIPv2-ready segmentation pipe.

        alignment_transform: AlignmentTransform from face_pipeline, used to remap
        MediaPipe landmarks from original-image coordinates into aligned-crop
        coordinates before they are passed as SAM2 point prompts.
        If None (skip_alignment mode), landmarks are already in image_rgb space.

        Returns a single SegmentationResult that is ready to feed into
        HairCLIPv2 (text_editor.optimize + hairstyle_feature_blending).
        If SAM2 is not available, returns the BiSeNet-only result.
        """
        base = self._base_segment(image_tensor)
        if not self.sam2_refiner.available:
            return base
        aligned_observation = self._remap_landmarks(face_observation, alignment_transform, image_rgb)
        return self._refine_with_sam2(base, image_rgb, aligned_observation, prompt)

    # ── internal stages ────────────────────────────────────────────────────────

    @staticmethod
    def _remap_landmarks(
        face_observation: FaceObservation,
        alignment_transform: Optional[Any],
        image_rgb,
    ) -> FaceObservation:
        """Return a copy of face_observation with landmarks_px transformed from
        original-image coordinates into the aligned-crop coordinate space.

        MediaPipe runs on the original (unaligned) image, so its landmark pixel
        coordinates live in that space. SAM3 is fed the aligned 1024×1024 crop,
        so we must apply the same homography that produced the crop before we use
        the landmarks as SAM3 point prompts.

        When alignment_transform is None (skip_alignment mode) the landmarks
        are already in the correct space and are returned unchanged.
        """
        if alignment_transform is None:
            return face_observation

        import cv2
        import numpy as np

        lm = np.asarray(face_observation.landmarks_px, dtype=np.float32)  # (N, 2)
        # cv2.perspectiveTransform expects shape (1, N, 2)
        lm_transformed = cv2.perspectiveTransform(lm.reshape(1, -1, 2), alignment_transform.original_to_aligned)
        lm_transformed = lm_transformed.reshape(-1, 2)

        # Clip to image bounds
        h, w = image_rgb.shape[:2]
        lm_transformed[:, 0] = np.clip(lm_transformed[:, 0], 0, w - 1)
        lm_transformed[:, 1] = np.clip(lm_transformed[:, 1], 0, h - 1)

        # Also transform the bounding box corners
        left, top, right, bottom = face_observation.bbox
        corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
        corners_t = cv2.perspectiveTransform(corners.reshape(1, -1, 2), alignment_transform.original_to_aligned).reshape(-1, 2)
        new_bbox = (
            int(np.clip(corners_t[:, 0].min(), 0, w - 1)),
            int(np.clip(corners_t[:, 1].min(), 0, h - 1)),
            int(np.clip(corners_t[:, 0].max(), 0, w - 1)),
            int(np.clip(corners_t[:, 1].max(), 0, h - 1)),
        )

        return FaceObservation(
            face_shape=face_observation.face_shape,
            expression=face_observation.expression,
            confidence=face_observation.confidence,
            metrics=face_observation.metrics,
            bbox=new_bbox,
            landmarks_px=lm_transformed.tolist(),
        )

    def _base_segment(self, image_tensor) -> SegmentationResult:
        """Stage 1: BiSeNet hair segmentation."""
        import torch

        with torch.no_grad():
            with self.autocast_factory(self.device, self.use_autocast):
                logits_1024, _ = self.seg(image_tensor)
            parsing_mask = torch.argmax(logits_1024.float(), dim=1, keepdim=True).long()
            hair_mask = (parsing_mask == self.hair_class_index).float()
        return SegmentationResult(
            parsing_mask=parsing_mask,
            hair_mask=hair_mask,
            source="bisenet",
            sam_refined=False,
            sam_prompt=None,
        )

    def _refine_with_sam2(
        self,
        base: SegmentationResult,
        image_rgb,
        face_observation: FaceObservation,
        prompt: str,
    ) -> SegmentationResult:
        """Stage 2: SAM2 refinement using MediaPipe face landmarks."""
        sam_prompt = build_sam2_prompt_hint(prompt)
        refined_hair_mask = self.sam2_refiner.refine_mask_with_sam2(
            image_rgb=image_rgb,
            face_observation=face_observation,
            base_hair_mask=base.hair_mask,
            prompt=prompt,
        )
        # SAM2가 놓친 영역도 유지: BiSeNet + SAM2 합집합(union) 사용
        # SAM2만 쓰면 SAM2가 못잡은 가는 머리카락, 경계 등이 사라짐
        refined_hair_union = ((refined_hair_mask > 0.5) | (base.hair_mask > 0.5)).float()
        refined_parsing_mask = base.parsing_mask.clone()
        refined_parsing_mask[base.parsing_mask == self.hair_class_index] = 0
        refined_parsing_mask[refined_hair_union > 0.5] = self.hair_class_index
        return SegmentationResult(
            parsing_mask=refined_parsing_mask.long(),
            hair_mask=refined_hair_union,
            source="sam2",
            sam_refined=True,
            sam_prompt=sam_prompt,
        )


__all__ = [
    "DEFAULT_SAM_KEYWORDS",
    "SAMMaskingPipeline",
    "SAM2Refiner",
    "SegmentationResult",
    "build_sam2_prompt_hint",
]
