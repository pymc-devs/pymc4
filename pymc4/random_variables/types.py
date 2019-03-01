"""
Stores the type-hint Types.

TensorLike is for float-like tensors (scalars, vectors, matrices, tensors)
IntTensorLike like TensorLike, just for ints.
"""

from typing import NewType, Union, Sequence


TensorLike = NewType("TensorLike", Union[Sequence[int], Sequence[float], int, float])
IntTensorLike = NewType("IntTensorLike", Union[int, Sequence[int]])
