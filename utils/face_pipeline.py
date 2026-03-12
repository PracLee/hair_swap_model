from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple

from runtime_spec import GENERATOR_OUTPUT_SIZE
from utils.runtime_compat import build_mediapipe_face_mesh


@dataclasses.dataclass
class FaceObservation:
    face_shape: str
    expression: str
    confidence: float
    metrics: Dict[str, float]
    bbox: Tuple[int, int, int, int]
    landmarks_px: Any = dataclasses.field(repr=False)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "face_shape": self.face_shape,
            "expression": self.expression,
            "confidence": round(self.confidence, 2),
            "metrics": self.metrics,
            "bbox": {
                "left": self.bbox[0],
                "top": self.bbox[1],
                "right": self.bbox[2],
                "bottom": self.bbox[3],
            },
        }


@dataclasses.dataclass(frozen=True)
class AlignmentTransform:
    original_to_aligned: Any = dataclasses.field(repr=False)
    aligned_to_original: Any = dataclasses.field(repr=False)
    original_size: Tuple[int, int]
    output_size: int


@dataclasses.dataclass
class AlignmentResult:
    aligned_image: Any = dataclasses.field(repr=False)
    aligned_rgb: Any = dataclasses.field(repr=False)
    transform: AlignmentTransform = dataclasses.field(repr=False)


class FaceAnalyzer:
    _LM = {
        "forehead_top": 10,
        "chin_bottom": 152,
        "left_cheek": 234,
        "right_cheek": 454,
        "left_jaw": 172,
        "right_jaw": 397,
        "left_temple": 21,
        "right_temple": 251,
        "mouth_left": 61,
        "mouth_right": 291,
        "upper_lip": 13,
        "lower_lip": 14,
        "left_eye_top": 159,
        "left_eye_bottom": 145,
        "right_eye_top": 386,
        "right_eye_bottom": 374,
        "left_brow": 105,
        "right_brow": 334,
    }
    LEFT_EYE_INDICES = (33, 133, 159, 145, 160, 144)
    RIGHT_EYE_INDICES = (362, 263, 386, 374, 387, 373)

    def __init__(self) -> None:
        self._mesh = None

    def inspect(self, image) -> Optional[FaceObservation]:
        import cv2
        import numpy as np

        rgb = self._ensure_rgb(image)
        results = self._face_mesh().process(rgb)
        if not results.multi_face_landmarks:
            return None

        height, width = rgb.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        points = np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float32)

        def pt(index: int) -> np.ndarray:
            return points[index]

        forehead = pt(self._LM["forehead_top"])
        chin = pt(self._LM["chin_bottom"])
        left_cheek = pt(self._LM["left_cheek"])
        right_cheek = pt(self._LM["right_cheek"])
        left_jaw = pt(self._LM["left_jaw"])
        right_jaw = pt(self._LM["right_jaw"])
        left_temple = pt(self._LM["left_temple"])
        right_temple = pt(self._LM["right_temple"])

        face_length = float(np.linalg.norm(forehead - chin))
        face_width = float(np.linalg.norm(left_cheek - right_cheek))
        jaw_width = float(np.linalg.norm(left_jaw - right_jaw))
        forehead_width = float(np.linalg.norm(left_temple - right_temple))

        aspect_ratio = face_length / max(face_width, 1e-6)
        jaw_ratio = jaw_width / max(face_width, 1e-6)
        forehead_ratio = forehead_width / max(face_width, 1e-6)
        face_shape, confidence = self._classify_face_shape(aspect_ratio, jaw_ratio, forehead_ratio)
        expression, smile_ratio, eye_openness = self._classify_expression(points)

        min_xy = np.clip(points.min(axis=0), a_min=0.0, a_max=None).astype(int)
        max_xy = np.clip(points.max(axis=0), a_min=0.0, a_max=None).astype(int)
        return FaceObservation(
            face_shape=face_shape,
            expression=expression,
            confidence=confidence,
            metrics={
                "face_length": round(face_length, 1),
                "face_width": round(face_width, 1),
                "jaw_width": round(jaw_width, 1),
                "aspect_ratio": round(aspect_ratio, 3),
                "jaw_ratio": round(jaw_ratio, 3),
                "forehead_ratio": round(forehead_ratio, 3),
                "smile_ratio": round(float(smile_ratio), 3),
                "eye_openness": round(float(eye_openness), 3),
            },
            bbox=(int(min_xy[0]), int(min_xy[1]), int(max_xy[0]), int(max_xy[1])),
            landmarks_px=points,
        )

    def analyze(self, image) -> Optional[Dict[str, Any]]:
        observation = self.inspect(image)
        return observation.to_payload() if observation else None

    def close(self) -> None:
        if self._mesh is None:
            return
        close = getattr(self._mesh, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        self._mesh = None

    def _face_mesh(self):
        if self._mesh is not None:
            return self._mesh
        self._mesh = build_mediapipe_face_mesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        return self._mesh

    @staticmethod
    def _ensure_rgb(image):
        import cv2

        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @classmethod
    def _classify_expression(cls, points) -> Tuple[str, float, float]:
        import numpy as np

        mouth_w = np.linalg.norm(points[cls._LM["mouth_left"]] - points[cls._LM["mouth_right"]])
        mouth_h = np.linalg.norm(points[cls._LM["upper_lip"]] - points[cls._LM["lower_lip"]])
        smile_ratio = mouth_w / max(mouth_h, 1e-6)
        left_eye_h = np.linalg.norm(points[cls._LM["left_eye_top"]] - points[cls._LM["left_eye_bottom"]])
        right_eye_h = np.linalg.norm(points[cls._LM["right_eye_top"]] - points[cls._LM["right_eye_bottom"]])
        avg_eye_h = (left_eye_h + right_eye_h) / 2.0
        left_brow_dist = np.linalg.norm(points[cls._LM["left_brow"]] - points[cls._LM["left_eye_top"]])
        right_brow_dist = np.linalg.norm(points[cls._LM["right_brow"]] - points[cls._LM["right_eye_top"]])
        eye_openness = avg_eye_h + ((left_brow_dist + right_brow_dist) / 2.0) * 0.5

        if smile_ratio > 4.5:
            expression = "smile"
        elif eye_openness > 35 and mouth_h > 15:
            expression = "surprise"
        else:
            expression = "neutral"
        return expression, float(smile_ratio), float(eye_openness)

    @staticmethod
    def _classify_face_shape(aspect_ratio: float, jaw_ratio: float, forehead_ratio: float) -> Tuple[str, float]:
        scores: Dict[str, float] = {}
        scores["oval"] = (
            (1.0 if 1.2 <= aspect_ratio <= 1.5 else 0.5)
            * (1.0 if 0.65 <= jaw_ratio <= 0.82 else 0.5)
            * (1.0 if 0.75 <= forehead_ratio <= 0.95 else 0.6)
        )
        scores["round"] = (
            (1.0 if aspect_ratio < 1.25 else 0.4)
            * (1.0 if jaw_ratio > 0.78 else 0.5)
        )
        scores["square"] = (
            (1.0 if aspect_ratio < 1.3 else 0.4)
            * (1.0 if jaw_ratio > 0.85 else 0.4)
            * (1.0 if forehead_ratio > 0.85 else 0.6)
        )
        scores["heart"] = (
            (1.0 if forehead_ratio > 0.85 else 0.4)
            * (1.0 if jaw_ratio < 0.72 else 0.4)
        )
        scores["oblong"] = (
            (1.0 if aspect_ratio > 1.45 else 0.3)
            * (1.0 if 0.7 <= jaw_ratio <= 0.85 else 0.5)
        )
        best_shape = max(scores, key=scores.get)
        total = sum(scores.values()) or 1.0
        return best_shape, scores[best_shape] / total

    @staticmethod
    def get_recommendations(
        face_shape: str,
        trend_kb: Any,
        expression: str = "neutral",
        prompt: Optional[str] = None,
        locale: str = "global",
        limit: int = 5,
    ) -> Dict[str, Any]:
        return trend_kb.recommend(
            face_shape=face_shape,
            expression=expression,
            prompt=prompt,
            locale=locale,
            limit=limit,
        )


class MediaPipeFFHQAligner:
    LEFT_EYE_INDICES = FaceAnalyzer.LEFT_EYE_INDICES
    RIGHT_EYE_INDICES = FaceAnalyzer.RIGHT_EYE_INDICES

    def __init__(
        self,
        output_size: int = GENERATOR_OUTPUT_SIZE,
        intermediate_size: int = GENERATOR_OUTPUT_SIZE * 2,
        min_source_size: int = GENERATOR_OUTPUT_SIZE,
        max_upscale_factor: float = 2.0,
    ) -> None:
        self.output_size = output_size
        self.intermediate_size = max(int(intermediate_size), int(output_size))
        self.min_source_size = max(int(min_source_size), 1)
        self.max_upscale_factor = max(float(max_upscale_factor), 1.0)

    def build_transform(self, image_bgr, observation: FaceObservation) -> AlignmentTransform:
        import cv2
        import numpy as np

        points = observation.landmarks_px
        eye_left = points[list(self.LEFT_EYE_INDICES)].mean(axis=0)
        eye_right = points[list(self.RIGHT_EYE_INDICES)].mean(axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = points[61]
        mouth_right = points[291]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x_axis = eye_to_eye - np.flipud(eye_to_mouth) * np.array([-1.0, 1.0], dtype=np.float32)
        x_axis /= max(np.hypot(*x_axis), 1e-6)
        x_axis *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y_axis = np.flipud(x_axis) * np.array([-1.0, 1.0], dtype=np.float32)
        center = eye_avg + eye_to_mouth * 0.1

        quad = np.stack(
            [
                center - x_axis - y_axis,
                center - x_axis + y_axis,
                center + x_axis + y_axis,
                center + x_axis - y_axis,
            ]
        ).astype(np.float32)
        destination = np.array(
            [
                [0.0, 0.0],
                [0.0, float(self.output_size - 1)],
                [float(self.output_size - 1), float(self.output_size - 1)],
                [float(self.output_size - 1), 0.0],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(quad, destination)
        inverse_matrix = cv2.getPerspectiveTransform(destination, quad)
        height, width = image_bgr.shape[:2]
        return AlignmentTransform(
            original_to_aligned=matrix,
            aligned_to_original=inverse_matrix,
            original_size=(width, height),
            output_size=self.output_size,
        )

    def align(
        self,
        image_bgr,
        observation: FaceObservation,
        transform: Optional[AlignmentTransform] = None,
    ) -> AlignmentResult:
        import cv2
        import numpy as np
        import PIL.Image

        transform = transform or self.build_transform(image_bgr, observation)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        prepared_rgb, source_scale_matrix = self._prepare_alignment_source(rgb)
        render_size = max(self.output_size, self.intermediate_size)
        render_scale = float(render_size) / float(self.output_size)
        render_scale_matrix = np.array(
            [[render_scale, 0.0, 0.0], [0.0, render_scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        prepared_to_render = render_scale_matrix @ transform.original_to_aligned @ np.linalg.inv(source_scale_matrix)
        aligned_render = cv2.warpPerspective(
            prepared_rgb,
            prepared_to_render,
            (render_size, render_size),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if render_size != self.output_size:
            aligned = cv2.resize(
                aligned_render,
                (self.output_size, self.output_size),
                interpolation=cv2.INTER_LANCZOS4,
            )
        else:
            aligned = aligned_render
        return AlignmentResult(
            aligned_image=PIL.Image.fromarray(aligned),
            aligned_rgb=aligned,
            transform=transform,
        )

    def _compute_pre_alignment_scale(self, width: int, height: int) -> float:
        min_dimension = min(int(width), int(height))
        if min_dimension <= 0:
            return 1.0
        target_min_dimension = max(self.output_size, self.min_source_size)
        if min_dimension >= target_min_dimension:
            return 1.0
        return min(self.max_upscale_factor, float(target_min_dimension) / float(min_dimension))

    def _prepare_alignment_source(self, rgb):
        import cv2
        import numpy as np

        height, width = rgb.shape[:2]
        scale = self._compute_pre_alignment_scale(width, height)
        if scale <= 1.0 + 1e-6:
            return rgb, np.eye(3, dtype=np.float32)
        resized = cv2.resize(
            rgb,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_LANCZOS4 if scale > 1.25 else cv2.INTER_CUBIC,
        )
        scale_matrix = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return resized, scale_matrix


__all__ = [
    "AlignmentResult",
    "AlignmentTransform",
    "FaceAnalyzer",
    "FaceObservation",
    "MediaPipeFFHQAligner",
]
