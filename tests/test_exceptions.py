"""Tests for src/exceptions.py"""

import pytest

from src.exceptions import (
    ConfigurationError,
    DataValidationError,
    MMMError,
    ModelNotFittedError,
)


class TestExceptionHierarchy:
    def test_mmm_error_is_base_exception(self):
        """MMMError should inherit from Exception."""
        assert issubclass(MMMError, Exception)

    def test_model_not_fitted_inherits_mmm_error(self):
        """ModelNotFittedError should be a subclass of MMMError."""
        assert issubclass(ModelNotFittedError, MMMError)

    def test_data_validation_inherits_mmm_error(self):
        """DataValidationError should be a subclass of MMMError."""
        assert issubclass(DataValidationError, MMMError)

    def test_configuration_error_inherits_mmm_error(self):
        """ConfigurationError should be a subclass of MMMError."""
        assert issubclass(ConfigurationError, MMMError)

    def test_catch_all_with_mmm_error(self):
        """All domain exceptions should be caught by MMMError."""
        for exc_class in (
            ModelNotFittedError,
            DataValidationError,
            ConfigurationError,
        ):
            with pytest.raises(MMMError):
                raise exc_class("test message")

    def test_exception_preserves_message(self):
        """Exception message should be accessible via str()."""
        msg = "Model has not been fitted yet"
        exc = ModelNotFittedError(msg)
        assert str(exc) == msg

    def test_exception_preserves_args(self):
        """Exception args tuple should contain the message."""
        msg = "Invalid data format"
        exc = DataValidationError(msg)
        assert exc.args == (msg,)
