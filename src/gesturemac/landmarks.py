"""Hand landmark types (spec §5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HandJoint(str, Enum):
    WRIST = "wrist"
    THUMB_TIP = "thumb_tip"
    INDEX_TIP = "index_tip"
    INDEX_PIP = "index_pip"
    INDEX_MCP = "index_mcp"
    MIDDLE_TIP = "middle_tip"
    MIDDLE_PIP = "middle_pip"
    MIDDLE_MCP = "middle_mcp"
    RING_TIP = "ring_tip"
    RING_PIP = "ring_pip"
    PINKY_TIP = "pinky_tip"
    PINKY_PIP = "pinky_pip"


@dataclass(frozen=True)
class HandPoint:
    """Normalized image coordinates (0…1), origin top-left."""

    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class HandFrame:
    timestamp: float
    joints: dict[HandJoint, HandPoint]
    aspect: float = 1.0  # source frame width / height

    def get(self, joint: HandJoint) -> HandPoint | None:
        return self.joints.get(joint)

    def axis_scale(self) -> tuple[float, float]:
        """Factors converting normalized x/y into equal physical units (short side = 1)."""
        if self.aspect >= 1.0:
            return self.aspect, 1.0
        return 1.0, 1.0 / self.aspect


# MediaPipe hand landmark indices (21 points).
_MEDIAPIPE_INDEX: dict[HandJoint, int] = {
    HandJoint.WRIST: 0,
    HandJoint.THUMB_TIP: 4,
    HandJoint.INDEX_MCP: 5,
    HandJoint.INDEX_PIP: 6,
    HandJoint.INDEX_TIP: 8,
    HandJoint.MIDDLE_MCP: 9,
    HandJoint.MIDDLE_PIP: 10,
    HandJoint.MIDDLE_TIP: 12,
    HandJoint.RING_PIP: 14,
    HandJoint.RING_TIP: 16,
    HandJoint.PINKY_PIP: 18,
    HandJoint.PINKY_TIP: 20,
}


def mediapipe_joint_map() -> dict[HandJoint, int]:
    return dict(_MEDIAPIPE_INDEX)
