# Copyright 2016 Skymind,Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from .op import op

# Array manipulation routines
# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.array-manipulation.html


@op
def reshape(arr, *args):
    if len(args) == 1 and type(args) in (list, tuple):
        args = tuple(args[0])
    return arr.reshape(*args)


@op
def transpose(arr):
    return arr.transpose()


@op
def ravel(arr):
    return arr.ravel()


@op
def flatten(arr):
    return arr.ravel().dup()


@op
def moveaxis(arr, source, destination):
    assert type(source) == type(
        destination), 'source and destination should be of same type.'
    shape = arr.shape()
    ndim = len(shape)
    x = list(range(ndim))
    if type(source) is int:
        if source < 0:
            source += ndim
        if destination < 0:
            destination += ndim
        z = x.pop(source)
        x.insert(destination, z)
        return arr.permute(*x)
    if type(source) in (list, tuple):
        source = list(source)
        destination = list(destination)
        assert len(source) == len(destination)
        for src, dst in zip(source, destination):
            if src < 0:
                src += ndim
            if dst < 0:
                dst += ndim
            z = x.pop(src)
            x.insert(dst, z)
        return arr.permute(*x)


@op
def permute(arr, *axis):
    if len(axis) == 1:
        axis = axis[0]
    assert set(axis) in [set(list(range(len(axis)))),
                         set(list(range(len(arr.shape()))))]
    return arr.permute(*axis)


@op
def expand_dims(arr, axis):
    return arr.expandDims(axis)


@op
def squeeze(arr, axis):
    shape = arr.shape()
    if type(axis) in (list, tuple):
        shape = [shape[i] for i in range(len(shape)) if i not in axis]
    else:
        shape.pop(axis)
    return arr.reshape(*shape)


@op
def concatenate(arrs, axis):
    return Nd4j.concat(axis, *arrs)


@op
def hstack(arrs):
    return Nd4j.hstack(arrs)


@op
def vstack(arrs):
    return Nd4j.vstack(arrs)


@op
def stack(arrs, axis):
    for i, arr in enumerate(arrs):
        shape = arr.shape()
        shape.insert(axis, 1)
        arrs[i] = arr.reshape(*shape)
    return Nd4j.concat(axis, *arrs)


@op
def tile(arr, reps):
    if type(reps) is int:
        return Nd4j.tile(arr, reps)
    else:
        return Nd4j.tile(arr, *reps)


@op
def repeat(arr, repeats, axis=None):
    if type(repeats) is int:
        repeats = (repeats,)
    if axis is None:
        return arr.repeat(-1, *repeats).reshape(-1)
    else:
        return arr.repeat(axis, *repeats)
