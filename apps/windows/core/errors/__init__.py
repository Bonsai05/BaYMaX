"""Errors returned by the Windows Core."""


class CoreError(Exception):
    """Base class for expected Core operation failures."""


class ValidationError(CoreError, ValueError):
    """Input does not satisfy a Core contract."""


class ConflictError(CoreError):
    """An optimistic version or ownership check failed."""


class NotFoundError(CoreError):
    """A requested state record, event, or task does not exist."""


__all__ = ["CoreError", "ConflictError", "NotFoundError", "ValidationError"]
