"""Custom exceptions for DVRPTW benchmark."""


class BenchError(Exception):
    """Base benchmark exception."""


class DatasetError(BenchError):
    """Raised when dataset loading/parsing fails."""


class SolverUnavailableError(BenchError):
    """Raised when optional solver dependency is not available."""


class InfeasibleSolutionError(BenchError):
    """Raised when a solution is infeasible and strict mode is enabled."""
