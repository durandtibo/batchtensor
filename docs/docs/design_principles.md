# Design principles

- **Batch is dim 0, sequence is dim 1.** Every `*_along_batch` and
  `*_along_seq` function is fixed to these dimensions. Data laid out as
  `(seq, batch, ...)` must be transposed first.
- **Nested functions apply the tensor function with `recursive_apply`.**
  Dicts, lists and tuples are traversed and the tensor function is applied to
  each `torch.Tensor` leaf. The container type is preserved.
- **Non-tensor leaves** are returned unchanged by pointwise and reduction
  functions that go through `recursive_apply`.
- **Joining and slicing** (`cat_*`, `stack_*`, `chunk_*`, `split_*`) accept a
  flat `dict` of tensors.
- **Named-tuple results** (`max`, `min`, `median`, `sort`) keep their
  `torch.return_types.*` type in the nested output.
- **Views versus copies.** `slice`, `select`, `chunk` and `split` return views
  that share memory with the input. Most other functions return new tensors.
- **Argument errors** raise `ValueError`.
- **numpy** is only needed for `from_numpy` and `to_numpy`.
