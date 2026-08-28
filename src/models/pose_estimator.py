import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class PoseEstimator(nn.Module):
    """
    Predicts an object's camera-frame xyz position and its orientation (as a
    6D rotation) from its full agentview frame, bounding box, class, and a
    tight crop.

    Position and rotation need very different visual information, so the
    model uses two separate streams:
    - Position stream: full agentview frame + bbox + class -> xyz_head. A
      tight crop on its own cannot tell the model how big or how far away
      the object really is, because any crop gets resized to a fixed size
      before it reaches the network. The full frame gives the model the bin
      edges and the other objects to measure against, and the bbox tells it
      exactly which object to look at and where it is in the frame.
    - Rotation stream: a tight crop of just the object, with its class
      passed through a small MLP branch (with dropout) before the two are
      combined -> rot_head. Rotation depends on fine detail, like which face
      or edge is facing the camera. That detail is mostly lost once the
      object is only a small part of the resized full frame, so this stream
      instead gets a tight crop that spends all its pixels on the object.

    The two streams share nothing: each has its own ResNet18 backbone. Only
    the training loop and the checkpoint file are shared. The rot_head
    output is a 6D rotation representation (Zhou et al., 2019). Use
    rot6d_to_matrix() to turn it into a proper, orthonormal 3x3 rotation
    matrix.
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
                normalized to [0, 1] by the frame's width and height.
            class_onehot (torch.Tensor): (B, num_classes) one-hot class vector.
            crop (torch.Tensor): (B, 3, H, W) normalized RGB tight object crops.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (xyz, rot6d), with shapes (B, 3) and (B, 6).
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
        Converts the raw 6D rotation output into a proper, orthonormal 3x3
        rotation matrix, using Gram-Schmidt orthogonalization.

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
