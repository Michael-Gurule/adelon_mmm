"""Domain-specific exceptions for Adelon MMM."""


class MMMError(Exception):
    """Base exception for Adelon MMM errors."""


class ModelNotFittedError(MMMError):
    """Raised when accessing results before model fitting."""


class DataValidationError(MMMError):
    """Raised when input data fails validation checks."""


class ConfigurationError(MMMError):
    """Raised when configuration is invalid or missing."""


class OptimizationInfeasibleError(MMMError):
    """Raised when the constrained optimization problem has no feasible solution.

    Common causes:
      - Total budget is less than the sum of per-channel minimum spend bounds.
      - Per-channel minimum spend exceeds per-channel maximum spend.
    """
