import logging
from typing import Optional

import onnxruntime as ort
import torch

from src.util.types import DeviceConfig


class DeviceConfigurator:
    """
    Detects the available compute device and sets up torch and
    onnxruntime to use it.

    Picks the best torch device available (cuda, then mps, then cpu) and
    the matching ONNX Runtime execution provider. This way the rest of
    the pipeline does not need to repeat this detection logic.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Sets up the DeviceConfigurator.

        Args:
            logger (Optional[logging.Logger]): Logger instance. If given,
                `resolve()` logs the chosen torch device and ONNX
                providers.

        Returns:
            None
        """

        self.logger = logger

        # The ONNX Runtime execution provider to prefer for each torch
        # device, if it is available on this machine. CPUExecutionProvider
        # is always added separately as a fallback in
        # `_resolve_onnx_providers`, no matter what is in this map.
        self._onnx_provider_by_torch_device = {
            "cuda": "CUDAExecutionProvider",
            "mps": "CoreMLExecutionProvider",
        }

    def resolve(self) -> DeviceConfig:
        """
        Works out which compute device to use for this run.

        Returns:
            DeviceConfig: The chosen torch device, plus the ONNX Runtime
                execution providers to use, in priority order.
        """

        torch_device = self._resolve_torch_device()
        onnx_providers = self._resolve_onnx_providers(torch_device)

        if self.logger:
            self.logger.info("Using torch device: %s", torch_device)
            self.logger.info("Using ONNX providers: %s", onnx_providers)

        return DeviceConfig(
            torch_device=torch_device,
            onnx_providers=onnx_providers,
        )

    def _resolve_torch_device(self) -> str:
        """
        Picks the best available torch device.

        Returns:
            str: "cuda" if a GPU is available, else "mps" on Apple
                Silicon, else "cpu".
        """

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _resolve_onnx_providers(self, torch_device: str) -> list[str]:
        """
        Picks the ONNX Runtime execution providers to use, matching the
        chosen torch device where possible.

        Args:
            torch_device (str): The torch device from `_resolve_torch_device`.

        Returns:
            list[str]: Execution providers in priority order. The
                hardware-accelerated provider for `torch_device` comes
                first, but only if it is actually installed and available
                on this machine. For example, `onnxruntime-gpu` might not
                be installed on a CPU-only setup, even if torch itself
                reports "cuda". "CPUExecutionProvider" is always included
                last, as a fallback.
        """

        available_providers = ort.get_available_providers()

        providers = []

        preferred_provider = self._onnx_provider_by_torch_device.get(torch_device)

        if preferred_provider and preferred_provider in available_providers:
            providers.append(preferred_provider)

        providers.append("CPUExecutionProvider")

        return providers
