import warnings
from dataclasses import dataclass

import casadi as ca


class FunctionDescriptor:
    """Wraps a ca.Function with metadata and validated calling."""

    def __init__(self, function: ca.Function, description: str = ''):
        self.function = function
        self.description = description
        self.name = function.name()
        self.input_names  = [function.name_in(i)  for i in range(function.n_in())]
        self.output_names = [function.name_out(i) for i in range(function.n_out())]
        self.input_shapes  = [function.size_in(i)  for i in range(function.n_in())]
        self.output_shapes = [function.size_out(i) for i in range(function.n_out())]

    def __call__(self, *args, **kwargs):
        """
        Call the underlying ca.Function with shape validation.

        Supports positional or keyword calling, but not mixed.
        Raises ValueError (not a raw CasADi error) on:
          - wrong argument count
          - unknown keyword
          - missing keyword
          - shape mismatch for MX arguments
        """
        if args and kwargs:
            raise ValueError(
                f"Function '{self.name}': mixed positional and keyword arguments "
                "are not supported. Use one or the other."
            )

        if args:
            if len(args) != len(self.input_names):
                raise ValueError(
                    f"Function '{self.name}': expected {len(self.input_names)} "
                    f"positional argument(s), got {len(args)}."
                )
            ordered = list(args)
        else:
            unknown = set(kwargs) - set(self.input_names)
            if unknown:
                raise ValueError(
                    f"Function '{self.name}': unknown keyword argument(s): "
                    f"{sorted(unknown)}. Valid inputs: {self.input_names}."
                )
            missing = set(self.input_names) - set(kwargs)
            if missing:
                raise ValueError(
                    f"Function '{self.name}': missing required argument(s): "
                    f"{sorted(missing)}."
                )
            ordered = [kwargs[name] for name in self.input_names]

        # Shape validation — only for MX; numeric values are promoted by CasADi.
        for arg, expected, name in zip(ordered, self.input_shapes, self.input_names):
            if isinstance(arg, ca.MX) and arg.shape != expected:
                raise ValueError(
                    f"Function '{self.name}': input '{name}' expected shape "
                    f"{expected}, got {arg.shape}."
                )

        return self.function(*ordered)

    def __repr__(self) -> str:
        inputs  = ', '.join(f'{n}{s}' for n, s in zip(self.input_names,  self.input_shapes))
        outputs = ', '.join(f'{n}{s}' for n, s in zip(self.output_names, self.output_shapes))
        return f"FunctionDescriptor('{self.name}', [{inputs}] -> [{outputs}])"


_VALID_SEMANTIC_TYPES = {'scalar', 'vector', 'trajectory', 'indexed_set', 'time_grid'}


@dataclass
class SymbolDescriptor:
    symbol:        ca.MX
    name:          str
    shape:         tuple
    semantic_type: str
    frame:         str    = None
    time_grid:     object = None
    units:         str    = None

    def __post_init__(self):
        if self.semantic_type not in _VALID_SEMANTIC_TYPES:
            raise ValueError(
                f"SymbolDescriptor '{self.name}': invalid semantic_type "
                f"'{self.semantic_type}'. Must be one of "
                f"{sorted(_VALID_SEMANTIC_TYPES)}."
            )
        if self.symbol.shape != self.shape:
            raise ValueError(
                f"SymbolDescriptor '{self.name}': shape mismatch — "
                f"symbol.shape is {self.symbol.shape}, declared shape is {self.shape}."
            )
        if self.frame is not None and self.semantic_type != 'vector':
            warnings.warn(
                f"SymbolDescriptor '{self.name}': 'frame' is set but "
                f"semantic_type is '{self.semantic_type}', not 'vector'. "
                "This may be unintentional.",
                UserWarning,
                stacklevel=2,
            )
