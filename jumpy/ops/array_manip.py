from .op import op

# Array manipulation routines
# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.array-manipulation.html

@op
def reshape(arr, *args):
    if len(args) == 1 and type(args) in (list, tuple):
        args = tuple(args[0])
    return arr.reshape(*args)

@op
def ravel(arr):
    return arr.ravel()

@op
def flatten(arr):
    return arr.ravel().dup()

@op
def moveaxis(arr, source, destination):
    assert type(source) == type(destination), 'source and destination should be of same type.'
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
            set(list(range(len(arr.array.shape()))))]
    return arr.permute(*axis)
