"""Fix for PyTorch compatibility with Streamlit."""

import os
import sys


def apply_torch_fix():
    """
    Apply fixes for PyTorch compatibility with Streamlit.

    This prevents errors related to torch.classes.__path__ when using
    sentence-transformers with Streamlit.
    """
    # Set environment variable to suppress PyTorch/Streamlit compatibility issues
    os.environ["PYTORCH_JIT"] = "0"

    # Try to filter out problematic modules from sys.modules
    if "torch._classes" in sys.modules:
        del sys.modules["torch._classes"]

    # Optionally, suppress PyTorch warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    # Additional workaround to prevent Streamlit from examining torch.classes
    if "torch.classes" in sys.modules:
        sys.modules["torch.classes"].__path__ = []