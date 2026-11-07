# Design principles

- **Batch is dim 0, sequence is dim 1.** Every `*_along_batch` and
  `*_along_seq` function is fixed to these dimensions. Data laid out as
  `(seq, batch, ...)` must be transposed first.
- **Nested functions apply the tensor function with `recursive_apply`.**
  Dicts, lists and tuples are traversed and the tensor function is applied to
  each `torch.Tensor` leaf. The container type is preserved.
- **Leaves must be `torch.Tensor`.** A non-tensor leaf is not skipped or
  returned unchanged: the call raises a `TypeError` that names the leaf type.
- **One traversal.** The nested functions go through `apply_to_tensors`,
  `split_tensors` and `combine_tensors` of `batchtensor.utils.structure`.
  Joining and slicing (`cat_*`, `stack_*`, `chunk_*`, `split_*`) keep the
  container types (`OrderedDict`, named tuples). They do not support
  dataclasses.
- **Empty or inconsistent input** raises `ValueError`: an empty sequence for
  `cat_*`/`stack_*`, data without tensor for `chunk_*`/`split_*`/`shuffle_*`,
  and tensors with different sizes along the split dimension.
- **Named-tuple results** (`max`, `min`, `median`, `sort`) keep their
  `torch.return_types.*` type in the nested output.
- **Views versus copies.** `slice`, `select`, `chunk` and `split` return views
  that share memory with the input. Most other functions return new tensors.
- **Argument errors** raise `ValueError`.
- **numpy** is only needed for `from_numpy` and `to_numpy`.
