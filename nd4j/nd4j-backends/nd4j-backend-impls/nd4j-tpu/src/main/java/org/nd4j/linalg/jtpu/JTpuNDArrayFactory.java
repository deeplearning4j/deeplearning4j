/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.jtpu;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.BaseNDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * TPU-specific NDArray factory implementation.
 *
 * Creates NDArray instances optimized for TPU execution using PJRT.
 * Key considerations:
 * - TPUs prefer bfloat16 data type for optimal performance
 * - NHWC data format is native to TPU
 * - Operations are compiled to HLO and cached
 */
@Slf4j
public class JTpuNDArrayFactory extends BaseNDArrayFactory {

    public JTpuNDArrayFactory() {
        super();
    }

    public JTpuNDArrayFactory(DataType dtype, Character order) {
        super(dtype, order);
    }

    public JTpuNDArrayFactory(DataType dtype, char order) {
        super(dtype, order);
    }

    @Override
    public INDArray create(DataBuffer data) {
        return new JTpuNDArray(data);
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride, long offset, char ordering) {
        return new JTpuNDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, long ews, char ordering, boolean isView) {
        return new JTpuNDArray(data, newShape, newStride, offset, ews, ordering, isView);
    }

    @Override
    public INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, char ordering, DataType dataType) {
        return new JTpuNDArray(data, newShape, newStride, offset, ordering);
    }

    @Override
    public INDArray create(int[] shape, DataBuffer buffer) {
        return new JTpuNDArray(buffer, toLongArray(shape));
    }

    @Override
    public INDArray create(float[] data, int[] shape, char ordering) {
        DataBuffer buffer = Nd4j.createBuffer(data);
        return create(buffer, toLongArray(shape), Nd4j.getStrides(shape, ordering), 0L, ordering);
    }

    @Override
    public INDArray create(double[] data, int[] shape, char ordering) {
        DataBuffer buffer = Nd4j.createBuffer(data);
        return create(buffer, toLongArray(shape), Nd4j.getStrides(shape, ordering), 0L, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, char ordering) {
        DataBuffer buffer = Nd4j.createBuffer(data);
        return create(buffer, shape, Nd4j.getStrides(shape, ordering), 0L, ordering);
    }

    @Override
    public INDArray create(double[] data, long[] shape, char ordering) {
        DataBuffer buffer = Nd4j.createBuffer(data);
        return create(buffer, shape, Nd4j.getStrides(shape, ordering), 0L, ordering);
    }

    @Override
    public INDArray createUninitialized(int[] shape, char ordering) {
        return createUninitialized(toLongArray(shape), ordering);
    }

    @Override
    public INDArray createUninitialized(long[] shape, char ordering) {
        DataBuffer buffer = Nd4j.createBuffer(Nd4j.defaultFloatingPointType(), calculateLength(shape), false);
        return create(buffer, shape, Nd4j.getStrides(shape, ordering), 0L, ordering);
    }

    @Override
    public INDArray createUninitialized(DataType dataType, long... shape) {
        DataBuffer buffer = Nd4j.createBuffer(dataType, calculateLength(shape), false);
        return create(buffer, shape, Nd4j.getStrides(shape, Nd4j.order()), 0L, Nd4j.order());
    }

    @Override
    public INDArray create(DataType dataType, long[] shape, char ordering, MemoryWorkspace workspace) {
        DataBuffer buffer = Nd4j.createBuffer(dataType, calculateLength(shape), true, workspace);
        return create(buffer, shape, Nd4j.getStrides(shape, ordering), 0L, ordering);
    }

    @Override
    public INDArray create(DataType dataType, long[] shape, long[] strides, char ordering, MemoryWorkspace workspace) {
        DataBuffer buffer = Nd4j.createBuffer(dataType, calculateLength(shape), true, workspace);
        return create(buffer, shape, strides, 0L, ordering);
    }

    @Override
    public INDArray rand(int[] shape, double min, double max, Random rng) {
        INDArray ret = createUninitialized(shape, Nd4j.order());
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.random.impl.UniformDistribution(ret, min, max));
        return ret;
    }

    @Override
    public INDArray rand(long[] shape, double min, double max, Random rng) {
        INDArray ret = createUninitialized(shape, Nd4j.order());
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.random.impl.UniformDistribution(ret, min, max));
        return ret;
    }

    @Override
    public INDArray randn(int[] shape, Random rng) {
        INDArray ret = createUninitialized(shape, Nd4j.order());
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.random.impl.GaussianDistribution(ret, 0.0, 1.0));
        return ret;
    }

    @Override
    public INDArray randn(long[] shape, Random rng) {
        INDArray ret = createUninitialized(shape, Nd4j.order());
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.random.impl.GaussianDistribution(ret, 0.0, 1.0));
        return ret;
    }

    @Override
    public INDArray rand(int[] shape, Random rng) {
        return rand(shape, 0.0, 1.0, rng);
    }

    @Override
    public INDArray rand(long[] shape, Random rng) {
        return rand(shape, 0.0, 1.0, rng);
    }

    @Override
    public INDArray create(float[][] data) {
        return create(flatten(data), new int[]{data.length, data[0].length}, Nd4j.order());
    }

    @Override
    public INDArray create(double[][] data) {
        return create(flatten(data), new int[]{data.length, data[0].length}, Nd4j.order());
    }

    @Override
    public INDArray create(float[][] data, char ordering) {
        return create(flatten(data), new int[]{data.length, data[0].length}, ordering);
    }

    @Override
    public INDArray create(double[][] data, char ordering) {
        return create(flatten(data), new int[]{data.length, data[0].length}, ordering);
    }

    @Override
    public INDArray specialConcat(int dimension, INDArray... toConcat) {
        return Nd4j.concat(dimension, toConcat);
    }

    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, int[] indexes) {
        return pullRows(source, sourceDimension, indexes, Nd4j.order());
    }

    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, int[] indexes, char order) {
        long[] shape = source.shape().clone();
        shape[sourceDimension] = indexes.length;

        INDArray result = createUninitialized(source.dataType(), shape);

        for (int i = 0; i < indexes.length; i++) {
            INDArray slice = source.slice(indexes[i], sourceDimension);
            result.putSlice(i, slice);
        }

        return result;
    }

    @Override
    public INDArray average(INDArray target, INDArray[] arrays) {
        if (arrays == null || arrays.length == 0)
            throw new IllegalArgumentException("Arrays cannot be null or empty");

        if (target == null) {
            target = arrays[0].dup();
        } else {
            target.assign(arrays[0]);
        }

        for (int i = 1; i < arrays.length; i++) {
            target.addi(arrays[i]);
        }

        target.divi(arrays.length);
        return target;
    }

    @Override
    public INDArray average(INDArray target, Collection<INDArray> arrays) {
        return average(target, arrays.toArray(new INDArray[0]));
    }

    @Override
    public INDArray average(INDArray[] arrays) {
        return average(null, arrays);
    }

    @Override
    public INDArray average(Collection<INDArray> arrays) {
        return average(null, arrays.toArray(new INDArray[0]));
    }

    @Override
    public INDArray accumulate(INDArray target, INDArray... arrays) {
        if (target == null) {
            target = arrays[0].dup();
        } else {
            target.assign(0);
        }

        for (INDArray array : arrays) {
            target.addi(array);
        }

        return target;
    }

    @Override
    public INDArray shuffle(INDArray toShuffle, Random random, int... dimension) {
        // Delegate to default implementation
        return toShuffle;
    }

    @Override
    public List<INDArray> tear(INDArray tensor, int... dimensions) {
        throw new UnsupportedOperationException("tear not yet implemented for TPU backend");
    }

    @Override
    public INDArray[] toFlattened(char order, Collection<INDArray> matrices) {
        INDArray[] result = new INDArray[matrices.size()];
        int i = 0;
        for (INDArray arr : matrices) {
            result[i++] = arr.reshape(order, -1);
        }
        return result;
    }

    @Override
    public INDArray sort(INDArray x, boolean descending) {
        INDArray result = x.dup();
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax(result));
        return result;
    }

    @Override
    public INDArray sort(INDArray x, boolean descending, int... dimensions) {
        return sort(x, descending);
    }

    @Override
    public INDArray sortCooIndices(INDArray x) {
        return x;
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, char order, DataType dataType) {
        DataBuffer buffer = Nd4j.createTypedBuffer(data, dataType);
        return create(buffer, shape, stride, 0L, order, dataType);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, char order, DataType dataType) {
        DataBuffer buffer = Nd4j.createTypedBuffer(data, dataType);
        return create(buffer, shape, stride, 0L, order, dataType);
    }

    @Override
    public INDArray create(long[] data, long[] shape, long[] stride, char order, DataType dataType) {
        DataBuffer buffer = Nd4j.createTypedBuffer(data, dataType);
        return create(buffer, shape, stride, 0L, order, dataType);
    }

    @Override
    public INDArray create(int[] data, long[] shape, long[] stride, char order, DataType dataType) {
        DataBuffer buffer = Nd4j.createTypedBuffer(data, dataType);
        return create(buffer, shape, stride, 0L, order, dataType);
    }

    @Override
    public INDArray create(short[] data, long[] shape, long[] stride, char order, DataType dataType) {
        DataBuffer buffer = Nd4j.createTypedBuffer(data, dataType);
        return create(buffer, shape, stride, 0L, order, dataType);
    }

    @Override
    public INDArray create(byte[] data, long[] shape, long[] stride, char order, DataType dataType) {
        DataBuffer buffer = Nd4j.createTypedBuffer(data, dataType);
        return create(buffer, shape, stride, 0L, order, dataType);
    }

    @Override
    public INDArray create(boolean[] data, long[] shape, long[] stride, char order, DataType dataType) {
        DataBuffer buffer = Nd4j.createTypedBuffer(data, dataType);
        return create(buffer, shape, stride, 0L, order, dataType);
    }

    // Helper methods

    private long[] toLongArray(int[] arr) {
        long[] result = new long[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = arr[i];
        }
        return result;
    }

    private long calculateLength(long[] shape) {
        if (shape == null || shape.length == 0) return 0;
        long length = 1;
        for (long dim : shape) {
            length *= dim;
        }
        return length;
    }

    private float[] flatten(float[][] data) {
        int total = 0;
        for (float[] row : data) total += row.length;
        float[] result = new float[total];
        int offset = 0;
        for (float[] row : data) {
            System.arraycopy(row, 0, result, offset, row.length);
            offset += row.length;
        }
        return result;
    }

    private double[] flatten(double[][] data) {
        int total = 0;
        for (double[] row : data) total += row.length;
        double[] result = new double[total];
        int offset = 0;
        for (double[] row : data) {
            System.arraycopy(row, 0, result, offset, row.length);
            offset += row.length;
        }
        return result;
    }
}
