import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class PoseEstimator(nn.Module):
    """
    Predicts camera-frame xyz position AND orientation (6D rotation
    representation) for a detected object, given its full agentview frame,
    bbox, class, and a tight crop.

    Two streams, since xyz and rotation need fundamentally different visual
    information:
    - xyz stream: full agentview frame + bbox + class -> xyz_head. A tight
      crop alone carries no scale/position information to regress absolute
      xyz from, so the full frame (for spatial/perspective context) plus the
      bbox (which object, where) are used instead.
    - rotation stream: tight object crop, with class run through its own
      small MLP branch (with dropout) before fusing -> rot_head. Rotation
      depends on fine visual detail (which face/edges are visible), which is
      largely lost once the object is a small part of the downscaled full
      frame.

    Both streams share nothing (separate ResNet18 backbones) -- only the
    training loop and checkpoint are shared. rot_head output is the 6D
    rotation representation (Zhou et al., 2019); use rot6d_to_matrix() to
    turn it into an orthonormal 3x3 rotation matrix.
    """

    BBOX_FEATURES = 7
    CLASS_HIDDEN_FEATURES = 32
    HIDDEN_FEATURES = 256

    def __init__(self, num_classes: int, pretrained: bool = True, rotation_dropout_prob: float = 0.3) -> None:
        super().__init__()

        self.num_classes = num_classes

        # ===== xyz stream =====
        self.xyz_backbone = self._build_backbone(pretrained)
        xyz_backbone_out_features = self.xyz_backbone.fc.in_features
        self.xyz_backbone.fc = nn.Identity()

        xyz_head_in_features = xyz_backbone_out_features + self.BBOX_FEATURES + num_classes
        self.xyz_hidden = nn.Sequential(
            nn.Linear(xyz_head_in_features, self.HIDDEN_FEATURES),
            nn.ReLU(inplace=True),
        )
        self.xyz_head = nn.Linear(self.HIDDEN_FEATURES, 3)

        # ===== rotation stream =====
        self.rot_backbone = self._build_backbone(pretrained)
        rot_backbone_out_features = self.rot_backbone.fc.in_features
        self.rot_backbone.fc = nn.Identity()

        self.class_processor = nn.Sequential(
            nn.Linear(num_classes, self.CLASS_HIDDEN_FEATURES),
            nn.ReLU(inplace=True),
            nn.Dropout(rotation_dropout_prob),
            nn.Linear(self.CLASS_HIDDEN_FEATURES, self.CLASS_HIDDEN_FEATURES),
            nn.ReLU(inplace=True),
        )

        rot_head_in_features = rot_backbone_out_features + self.CLASS_HIDDEN_FEATURES
        self.rot_hidden = nn.Sequential(
            nn.Linear(rot_head_in_features, self.HIDDEN_FEATURES),
            nn.ReLU(inplace=True),
            nn.Dropout(rotation_dropout_prob),
        )
        self.rot_head = nn.Linear(self.HIDDEN_FEATURES, 6)

    @staticmethod
    def _build_backbone(pretrained: bool) -> nn.Module:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)

    def forward(self, image: torch.Tensor, bbox: torch.Tensor, class_onehot: torch.Tensor, crop: torch.Tensor):
        """
        Args:
            image (torch.Tensor): (B, 3, H, W) normalized RGB full agentview frames.
            bbox (torch.Tensor): (B, 7) bbox (x1, y1, x2, y2, area, cx, cy),
                normalized to [0, 1] by the frame's width/height.
            class_onehot (torch.Tensor): (B, num_classes) one-hot class vector.
            crop (torch.Tensor): (B, 3, H, W) normalized RGB tight object crops.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (xyz, rot6d), shapes (B, 3) and (B, 6).
        """

        xyz_features = self.xyz_backbone(image)
        xyz_features = torch.cat([xyz_features, bbox, class_onehot], dim=1)
        xyz = self.xyz_head(self.xyz_hidden(xyz_features))

        rot_image_features = self.rot_backbone(crop)
        rot_class_features = self.class_processor(class_onehot)
        rot_features = torch.cat([rot_image_features, rot_class_features], dim=1)
        rot6d = self.rot_head(self.rot_hidden(rot_features))

        return xyz, rot6d

    @staticmethod
    def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
        """
        Converts the raw 6D rotation output into an orthonormal 3x3 rotation
        matrix via Gram-Schmidt orthogonalization.

        Args:
            rot6d (torch.Tensor): (B, 6).

        Returns:
            torch.Tensor: (B, 3, 3) rotation matrices.
        """

        a1 = rot6d[:, 0:3]
        a2 = rot6d[:, 3:6]

        b1 = F.normalize(a1, dim=1)
        b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=2)
