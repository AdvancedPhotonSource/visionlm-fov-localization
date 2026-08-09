from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DirectPromptSettings:
    strategy: str
    demonstrations: int = 3
    repeats: int = 3


@dataclass(frozen=True)
class MetadataPolicySettings:
    primary_demonstrations: int
    primary_repeats: int
    primary_metric: str
    offset_fraction: float = 0.0
    supplemental_demonstrations: int | None = None
    supplemental_repeats: int = 0
    supplemental_min_overlap_iou: float = 0.0
    fallback_metric: str | None = None
    fallback_template_area: int = 0


@dataclass(frozen=True)
class ZoomPolicySettings:
    direct_proposals: int = 3
    zoom_seed_count: int = 1
    zoom_repeats: int = 3
    zoom_margin_scale: float = 0.75
    zoom_min_scale: float = 2.5
    rerank_metric: str = "nmi_raw"


@dataclass(frozen=True)
class PaperSettings:
    xrf_channel: str
    inputs_pre_rotated: bool
    main_model_label: str
    controlled_box_size_source: str
    challenging_box_size_source: str
    unconstrained_direct: DirectPromptSettings
    metadata_direct: DirectPromptSettings
    controlled_metadata: MetadataPolicySettings
    challenging_metadata: MetadataPolicySettings
    unconstrained_zoom: ZoomPolicySettings
    ablation_strategies: tuple[str, ...]
    ablation_demonstration_counts: tuple[int, ...]


PAPER_SETTINGS = PaperSettings(
    xrf_channel="P",
    inputs_pre_rotated=True,
    main_model_label="GPT-5",
    controlled_box_size_source="ground_truth_box_size",
    challenging_box_size_source="probe_metadata",
    unconstrained_direct=DirectPromptSettings(strategy="bbox_xywh"),
    metadata_direct=DirectPromptSettings(strategy="frac_xy"),
    controlled_metadata=MetadataPolicySettings(
        primary_demonstrations=3,
        primary_repeats=3,
        primary_metric="nmi_raw",
        offset_fraction=0.1,
    ),
    challenging_metadata=MetadataPolicySettings(
        primary_demonstrations=3,
        primary_repeats=3,
        primary_metric="edge_chamfer",
        supplemental_demonstrations=0,
        supplemental_repeats=3,
        supplemental_min_overlap_iou=0.5,
        fallback_metric="nmi_raw",
        fallback_template_area=60_000,
    ),
    unconstrained_zoom=ZoomPolicySettings(),
    ablation_strategies=("top_left_xy", "center_xy", "frac_xy"),
    ablation_demonstration_counts=(0, 1, 2, 3),
)
