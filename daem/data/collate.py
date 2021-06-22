import torch as th
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def entity_pad_collate(elems):
    batch = {'left': dict(), 'right': dict()}
    label = th.LongTensor([e[1] for e in elems])
    for where in ['left', 'right']:
        for col in elems[0][0]['left'].keys():
            seq = pad_sequence([th.LongTensor(e[where][col]) for e, _ in elems])
            size = th.LongTensor([e[where][col].shape[0] for e, _ in elems])
            batch[where][col] = dict(seq=seq, size=size)
    return batch, label


def pad_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, th.Tensor):
        out = None
        if th.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        elem_sizes = [elem.shape[0] for elem in batch]
        if np.min(elem_sizes) == np.max(elem_sizes):
            return th.stack(batch, 0, out=out)
        else:
            seq = pad_sequence(batch).long()
            size = th.LongTensor(elem_sizes)
            return dict(seq=seq, size=size)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return pad_collate([th.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return th.as_tensor(batch)
    elif isinstance(elem, float):
        return th.tensor(batch, dtype=th.float64)
    elif isinstance(elem, int_classes):
        return th.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
