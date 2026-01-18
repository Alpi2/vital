"""
Python-Rust FFI Integration for ECG Processing
High-performance ECG analysis with Rust backend and Python frontend
"""

__version__ = "0.1.0"
__author__ = "VitalStream Team"
__email__ = "dev@vitalstream.com"

# Try to import the Rust FFI module
try:
    import ecg_processor_py as _rust_ffi
    FFI_AVAILABLE = True
    FFI_VERSION = _rust_ffi.get_version()
    import logging
    logging.getLogger(__name__).info(
        f"✅ Rust FFI module loaded successfully: {FFI_VERSION}"
    )
except ImportError as e:
    FFI_AVAILABLE = False
    FFI_VERSION = None
    import logging
    logging.getLogger(__name__).warning(
        f"⚠️ Rust FFI module could not be loaded: {e}. "
        "Falling back to Python implementation (slower performance)."
    )
except Exception as e:
    FFI_AVAILABLE = False
    FFI_VERSION = None
    import logging
    logging.getLogger(__name__).error(
        f"💥 Unexpected error loading Rust FFI module: {e}. "
        "Falling back to Python implementation."
    )

# Export main classes and functions
from .analyzer import ECGAnalyzerFFI
from .config import ECGAnalysisConfig

__all__ = [
    "ECGAnalyzerFFI",
    "ECGAnalysisConfig", 
    "FFI_AVAILABLE",
    "FFI_VERSION",
    "__version__",
]
