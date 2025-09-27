import numpy as np

print("Basic Array Creation Test:")
arr = np.zeros((2, 2), dtype=np.float32)
print("Shape:", arr.shape)
print("Strides:", arr.strides)
print("Data type:", arr.dtype)

print("\nArray With Offset Test:")
# NumPy doesn't have a direct equivalent for offset, but we can simulate it
arr_with_offset = np.zeros(6, dtype=np.float32)[2:6].reshape(2, 2)
print("Shape:", arr_with_offset.shape)
print("Size:", arr_with_offset.size)

print("\nArray With Different Strides Test:")
arr_diff_strides = np.zeros((3, 3), dtype=np.float32, order='F')
arr_diff_strides[0, 0] = 1
arr_diff_strides[1, 0] = 2
arr_diff_strides[2, 0] = 3
print("Values:", arr_diff_strides[:, 0])

print("\nArray Reshape Test:")
arr_reshape = np.arange(6, dtype=np.float32).reshape(2, 3)
reshaped = arr_reshape.reshape(3, 2)
print("Reshaped shape:", reshaped.shape)
print("Reshaped values:", reshaped)

print("\nArray Slicing Test:")
arr_slice = np.arange(16, dtype=np.float32).reshape(4, 4)
slice_result = arr_slice[1:3, 1:3]
print("Slice shape:", slice_result.shape)
print("Slice values:", slice_result)

print("\nView Offsets 1D Test:")
arr_1d = np.arange(10)
view1_1d = arr_1d[2:7]
view2_1d = arr_1d[5:9:2]
print("View1 1D:", view1_1d)
print("View2 1D:", view2_1d)

print("\nView Offsets 2D Test:")
arr_2d = np.arange(24).reshape(4, 6)
view1_2d = arr_2d[1:3, 2:5]
view2_2d = arr_2d[:, 1:5:2]
print("View1 2D shape:", view1_2d.shape)
print("View1 2D:", view1_2d)
print("View2 2D shape:", view2_2d.shape)
print("View2 2D:", view2_2d)

print("\nView Offsets 3D Test:")
arr_3d = np.arange(60).reshape(3, 4, 5)
view1_3d = arr_3d[1, 1:3, :]
view2_3d = arr_3d[:, 2, 1:5:2]
print("View1 3D shape:", view1_3d.shape)
print("View1 3D:", view1_3d)
print("View2 3D shape:", view2_3d.shape)
print("View2 3D:", view2_3d)

print("\nView Offsets 4D Test:")
arr_4d = np.arange(120).reshape(2, 3, 4, 5)
view1_4d = arr_4d[1, 1:3, :, 1:4]
view2_4d = arr_4d[:, 2, 1:3, ::2]
print("View1 4D shape:", view1_4d.shape)
print("View1 4D first and last:", view1_4d[0, 0, 0], view1_4d[1, 3, 2])
print("View2 4D shape:", view2_4d.shape)
print("View2 4D first and last:", view2_4d[0, 0, 0], view2_4d[1, 1, 1])

print("\nView Offsets 5D Test:")
arr_5d = np.arange(240).reshape(2, 3, 4, 5, 2)
view1_5d = arr_5d[:, 1:3, 2, :, 1]
view2_5d = arr_5d[1, :, ::2, 1:4, :]
print("View1 5D shape:", view1_5d.shape)
print("View1 5D first and last:", view1_5d[0, 0, 0], view1_5d[1, 1, 4])
print("View2 5D shape:", view2_5d.shape)
print("View2 5D first and last:", view2_5d[0, 0, 0, 0], view2_5d[2, 1, 2, 1])

print("\nView Offsets 6D Test:")
arr_6d = np.arange(720).reshape(2, 3, 4, 5, 2, 3)
view1_6d = arr_6d[1, :, 1:3, 2, :, ::2]
view2_6d = arr_6d[:, 2, :, 1:4, 1, :]
print("View1 6D shape:", view1_6d.shape)
print("View1 6D first and last:", view1_6d[0, 0, 0, 0], view1_6d[2, 1, 1, 1])
print("View2 6D shape:", view2_6d.shape)
print("View2 6D first and last:", view2_6d[0, 0, 0, 0], view2_6d[1, 3, 2, 2])

print("\nMixed Datatype Views Test:")
arr_float = np.arange(24, dtype=np.float32).reshape(4, 6)
arr_double = np.arange(24, dtype=np.float64).reshape(4, 6)
arr_long = np.arange(24, dtype=np.int64).reshape(4, 6)
view_float = arr_float[1:3, 2:5]
view_double = arr_double[1:3, 2:5]
view_long = arr_long[1:3, 2:5]
print("Float view:", view_float[0, 0], view_float[1, 2])
print("Double view:", view_double[0, 0], view_double[1, 2])
print("Long view:", view_long[0, 0], view_long[1, 2])

print("\nNested Views Test:")
arr_nested = np.arange(120).reshape(4, 5, 6)
view1_nested = arr_nested[1:3, :, 2:5]
view2_nested = view1_nested[1, 1:4]
print("Nested view shape:", view2_nested.shape)
print("Nested view first and last:", view2_nested[0, 0], view2_nested[2, 2])

print("\nMixed Datatype Operations Test:")
arr_float_op = np.arange(24, dtype=np.float32).reshape(4, 6)
arr_int_op = np.arange(24, dtype=np.int32).reshape(4, 6)
view_float_op = arr_float_op[1:3, 2:5]
view_int_op = arr_int_op[1:3, 2:5]
result_op = view_float_op + view_int_op
print("Mixed datatype result dtype:", result_op.dtype)
print("Mixed datatype result first and last:", result_op[0, 0], result_op[1, 2])

print("\nNon-contiguous Views Test:")
arr_non_contiguous = np.arange(60).reshape(5, 4, 3)
view_non_contiguous = arr_non_contiguous[::2, :, 1:3]
print("Non-contiguous view shape:", view_non_contiguous.shape)
print("Non-contiguous view strides:", view_non_contiguous.strides)
print("Non-contiguous view first and last:", view_non_contiguous[0, 0, 0], view_non_contiguous[2, 3, 1])

print("\nZero-dimension View Test:")
arr_zero_dim = np.arange(24).reshape(4, 6)
view_zero_dim = arr_zero_dim[2:2, 3:5]
print("Zero-dimension view shape:", view_zero_dim.shape)
print("Zero-dimension view size:", view_zero_dim.size)

print("\nView Mutation Test:")
arr_mutation = np.arange(24).reshape(4, 6)
view_mutation = arr_mutation[1:3, 2:5]
view_mutation += 100
print("Mutated original array elements:", arr_mutation[1, 2], arr_mutation[2, 4])

print("\nOverlapping Views Test:")
arr_overlap = np.arange(36).reshape(6, 6)
view1_overlap = arr_overlap[1:4, 1:4]
view2_overlap = arr_overlap[2:5, 2:5]
result_overlap = view1_overlap + view2_overlap
print("Overlapping views result first and middle:", result_overlap[0, 0], result_overlap[1, 1])


print("Complex indexing:")
import numpy as np

# Create the initial array
arr = np.arange(120).reshape(2, 3, 4, 5)

# Create the view using complex indexing
view = arr[1, :, 1:3, 2:5:2]

# Print the shape of the view
print("View shape:", view.shape)

# Print specific elements of the view
print("View[0, 0, 0]:", view[0, 0, 0])
print("View[2, 1, 1]:", view[2, 1, 1])

# Print the entire view for reference
print("Entire view:")
print(view)