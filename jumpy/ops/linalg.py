from .op import op

# Linear algebra
# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html


@op
def dot(arr, other):
	return arr.mmul(other)
