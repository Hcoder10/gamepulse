"""Compatibility layer: use weave if available, otherwise provide no-op shims."""

try:
    import weave
    Scorer = weave.Scorer
    op = weave.op
except ImportError:
    class Scorer:
        """No-op base class when weave is not installed."""
        pass

    def op(*args, **kwargs):
        """No-op decorator when weave is not installed."""
        if args and callable(args[0]):
            return args[0]
        def decorator(fn):
            return fn
        return decorator
