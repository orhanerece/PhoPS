"""Project specific exceptions."""


class PhopsError(Exception):
    """Base exception for the package."""


class ConfigurationError(PhopsError):
    """Raised when configuration is invalid."""


class DependencyError(PhopsError):
    """Raised when required external tools are missing."""


class PipelineError(PhopsError):
    """Raised when the pipeline cannot complete."""


class AstrometrySolveError(PipelineError):
    """Raised when plate solving fails."""


class TargetResolutionError(PipelineError):
    """Raised when the target coordinates cannot be resolved."""
