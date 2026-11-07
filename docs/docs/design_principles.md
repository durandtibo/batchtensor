# Design principles

- **Batch is dim 0, sequence is dim 1.** Every `*_along_batch` and
  `*_along_seq` function is fixed to these dimensions. Data laid out as
  `(seq, batch, ...)` must be transposed first.
- **Nested functions apply the tensor function with `recursive_apply`.**
  Dicts, lists and tuples are traversed and the tensor function is applied to
  each `torch.Tensor` leaf. The container type is preserved.
- **Leaves must be `torch.Tensor`.** Non-tensor leaves are not skipped or
  returned unchanged: the call fails with the error of the underlying
  operation (the exception type depends on the function).
- **Joining and slicing** (`cat_*`, `stack_*`, `chunk_*`, `split_*`) recurse
  into dicts, lists and tuples like the other nested functions. They return
  plain `dict`, `list` and `tuple` containers.
- **Named-tuple results** (`max`, `min`, `median`, `sort`) keep their
  `torch.return_types.*` type in the nested output.
- **Views versus copies.** `slice`, `select`, `chunk` and `split` return views
  that share memory with the input. Most other functions return new tensors.
- **Argument errors** raise `ValueError`.
- **numpy** is only needed for `from_numpy` and `to_numpy`.
