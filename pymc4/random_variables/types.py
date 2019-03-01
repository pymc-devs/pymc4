from typing import NewType, Union, Sequence


TensorLike = NewType("TensorLike", Union[Sequence[int], Sequence[float], int, float])
IntTensorLike = NewType("IntTensorLike", Union[int, Sequence[int]])
