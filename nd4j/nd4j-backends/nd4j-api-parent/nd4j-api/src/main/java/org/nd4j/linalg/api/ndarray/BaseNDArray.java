/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ndarray;


import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import com.google.flatbuffers.FlatBufferBuilder;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import net.ericaro.neoitertools.Generator;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.graph.ByteOrder;
import org.nd4j.graph.FlatArray;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.instrumentation.Instrumentation;
import org.nd4j.linalg.api.iter.FirstAxisIterator;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.accum.Min;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.broadcast.*;
import org.nd4j.linalg.api.ops.impl.controlflow.Where;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.*;
import org.nd4j.linalg.api.ops.impl.shape.Tile;
import org.nd4j.linalg.api.ops.impl.transforms.Assign;
import org.nd4j.linalg.api.ops.impl.transforms.MatchConditionTransform;
import org.nd4j.linalg.api.ops.impl.transforms.Negative;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.*;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.*;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.exception.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.*;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.memory.MemcpyDirection;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.nd4j.linalg.util.LongUtils;
import org.nd4j.linalg.util.NDArrayMath;
import org.nd4j.linalg.workspace.WorkspaceUtils;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.*;

import static org.nd4j.linalg.factory.Nd4j.*;


/**
 * NDArray: (think numpy)
 * <p/>
 * A few things of note.
 * <p/>
 * An NDArray can have any number of dimensions.
 * <p/>
 * An NDArray is accessed via strides.
 * <p/>
 * Strides are how to index over
 * a contiguous block of data.
 * <p/>
 * This block of data has 2 orders(as of right now):
 * fortran and c
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseNDArray implements INDArray, Iterable {

    private static final long serialVersionUID = 3285982317165542614L;

    protected transient volatile DataBuffer shapeInformation;
    protected transient volatile DataBuffer data;
    //protected transient DataBuffer shape;
    //protected transient DataBuffer stride;
    protected transient boolean compressed = false;

    // this field holds jvm copy of shapeInfo
    protected transient JvmShapeInfo jvmShapeInfo;



    //Precalculate these arrays (like [3,2,1,0], [2,1,0], [1,0], [0] etc) for use in TAD, to avoid creating same int[]s over and over
    private static final int[][] tadFinalPermuteDimensions;
    static {
        tadFinalPermuteDimensions = new int[32][0];
        tadFinalPermuteDimensions[1] = new int[] {1, 0}; //Edge case for 1d tensors: selectively apply to column vectors
        for (int i = 2; i < 32; i++) {
            tadFinalPermuteDimensions[i] = new int[i];
            for (int k = i - 1, j = 0; k >= 0; k--, j++)
                tadFinalPermuteDimensions[i][j] = k;
        }
    }

    public BaseNDArray() {

    }

    /**
     * Returns true if this array is compressed, and false otherwise
     * @return
     */
    @Override
    public boolean isCompressed() {
        return compressed;
    }

    /**
     * This method marks INDArray instance as compressed
     * PLEASE NOTE: Do not use this method unless you 100% have to
     *
     * @param reallyCompressed
     */
    @Override
    public void markAsCompressed(boolean reallyCompressed) {
        this.compressed = reallyCompressed;
    }

    /**
     *
     * @param buffer
     */
    public BaseNDArray(DataBuffer buffer) {
        this.data = buffer;
        if (buffer.length() >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Length of buffer can not be >= Integer.MAX_VALUE");
        long[] shape = {1, (int) buffer.length()};
        long[] stride = Nd4j.getStrides(shape);
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride,  1, Nd4j.order(), Nd4j.dataType()));
        init(shape, stride);
    }

    /**
     *
     * @param buffer
     * @param shape
     * @param stride
     * @param offset
     * @param ordering
     */
    public BaseNDArray(DataBuffer buffer, int[] shape, int[] stride, long offset, char ordering) {
        this.data = offset > 0 ? Nd4j.createBuffer(buffer, offset, Shape.lengthOfBuffer(shape, stride)) : buffer;
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(ArrayUtil.toLongArray(shape), ArrayUtil.toLongArray(stride),
                Shape.elementWiseStride(shape, stride, ordering == 'f'), ordering, Nd4j.dataType()));
        init(shape, stride);
        // Shape.setElementWiseStride(this.shapeInfo(),Shape.elementWiseStride(shape, stride, ordering == 'f'));

    }

    public BaseNDArray(DataBuffer buffer, long[] shape, long[] stride, long offset, char ordering) {
        this.data = offset > 0 ? Nd4j.createBuffer(buffer, offset, Shape.lengthOfBuffer(shape, stride)) : buffer;
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride,
                Shape.elementWiseStride(shape, stride, ordering == 'f'), ordering, Nd4j.dataType()));
        init(shape, stride);
        // Shape.setElementWiseStride(this.shapeInfo(),Shape.elementWiseStride(shape, stride, ordering == 'f'));
    }

    public BaseNDArray(DataBuffer buffer,  DataBuffer.Type dataType, long[] shape, long[] stride, long offset, char ordering) {
        this.data = offset > 0 ? Nd4j.createBuffer(buffer, offset, Shape.lengthOfBuffer(shape, stride)) : buffer;
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride,
                Shape.elementWiseStride(shape, stride, ordering == 'f'), ordering, dataType));
        init(shape, stride);
        // Shape.setElementWiseStride(this.shapeInfo(),Shape.elementWiseStride(shape, stride, ordering == 'f'));
    }

    /**
     * Initialize the ndarray as a matrix
     * with the given data (indices preserved)
     * @param data
     */
    public BaseNDArray(double[][] data) {
        this(data, Nd4j.order());
    }

    /**
     *
     * @param data
     * @param ordering
     */
    public BaseNDArray(double[][] data, char ordering) {
        this(internalCreateBuffer(ordering == 'c' ? ArrayUtil.flatten(data) : ArrayUtil.flattenF(data)),
                new int[] {data.length, data[0].length},
                Nd4j.getStrides(new int[] {data.length, data[0].length}, ordering), 0, ordering);

        for (int r = 0; r < rows(); r++) {
            assert (data[r].length == columns());
        }
    }


    /**
     * Create with the specified shape and buffer
     *
     * @param shape  the shape
     * @param buffer the buffer
     */
    public BaseNDArray(int[] shape, DataBuffer buffer) {
        this.data = buffer;
        init(shape, Nd4j.getStrides(shape));
    }

    /**
     * Create this ndarray with the given data and shape and 0 offset
     *
     * @param data  the data to use
     * @param shape the shape of the ndarray
     */
    public BaseNDArray(float[] data, int[] shape, char ordering) {
        this(data, shape, 0, ordering);
    }

    /**
     * @param data     the data to use
     * @param shape    the shape of the ndarray
     * @param offset   the desired offset
     * @param ordering the ordering of the ndarray
     */
    public BaseNDArray(float[] data, int[] shape, long offset, char ordering) {
        this(data, shape, Nd4j.getStrides(shape, ordering), offset);
    }

    public BaseNDArray(double[] data, long[] shape, long offset, char ordering) {
        this(data, shape, Nd4j.getStrides(shape, ordering), offset);
    }

    public BaseNDArray(float[] data, long[] shape, long offset, char ordering) {
        this(data, shape, Nd4j.getStrides(shape, ordering), offset);
    }


    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param offset   the desired offset
     * @param ordering the ordering of the ndarray
     */
    public BaseNDArray(int[] shape, int[] stride, long offset, char ordering) {
        this(Nd4j.createBuffer(ArrayUtil.prodLong(shape)), shape, stride, offset, ordering);
    }

    public BaseNDArray(long[] shape, long[] stride, long offset, char ordering) {
        this(Nd4j.createBuffer(ArrayUtil.prodLong(shape)), shape, stride, offset, ordering);
    }

    /**
     * Construct an ndarray of the specified shape.
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param offset   the desired offset
     * @param ordering the ordering of the ndarray
     * @param initialize Whether to initialize the INDArray. If true: initialize. If false: don't.
     */
    public BaseNDArray(int[] shape, int[] stride, long offset, char ordering, boolean initialize) {
        this(Nd4j.createBuffer(ArrayUtil.prodLong(shape), initialize), shape, stride, offset, ordering);
    }

    public BaseNDArray(long[] shape, long[] stride, long offset, char ordering, boolean initialize) {
        this(Nd4j.createBuffer(ArrayUtil.prodLong(shape), initialize), shape, stride, offset, ordering);
    }

    public BaseNDArray(DataBuffer.Type type, long[] shape, long[] stride, long offset, char ordering, boolean initialize) {
        this(Nd4j.createBuffer(type, ArrayUtil.prodLong(shape), initialize), type, shape, stride, offset, ordering);
    }


    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param ordering the ordering of the ndarray
     */
    public BaseNDArray(int[] shape, int[] stride, char ordering) {
        this(shape, stride, 0, ordering);
    }


    /**
     *
     * @param shape
     * @param offset
     * @param ordering
     */
    public BaseNDArray(int[] shape, long offset, char ordering) {
        this(shape, Nd4j.getStrides(shape, ordering), offset, ordering);
    }

    public BaseNDArray(long[] shape, long offset, char ordering) {
        this(shape, Nd4j.getStrides(shape, ordering), offset, ordering);
    }


    /**
     * Create an ndarray
     * with the given shape
     * @param shape
     */
    public BaseNDArray(int[] shape) {
        this(shape, 0, Nd4j.order());
    }

    public BaseNDArray(long[] shape) {
        this(shape, 0, Nd4j.order());
    }


    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public BaseNDArray(int newRows, int newColumns, char ordering) {
        this.data = Nd4j.createBuffer((long) newRows * newColumns);
        val shape = new long[] {newRows, newColumns};
        val stride = Nd4j.getStrides(shape, ordering);
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride,
                Shape.elementWiseStride(shape, stride, ordering == 'f'), ordering, Nd4j.dataType()));
        init(shape, stride);
    }

    public BaseNDArray(long newRows, long newColumns, char ordering) {
        this.data = Nd4j.createBuffer((long) newRows * newColumns);
        long[] shape = new long[] {newRows, newColumns};
        long[] stride = Nd4j.getStrides(shape, ordering);
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride,
                Shape.elementWiseStride(shape, stride, ordering == 'f'), ordering, Nd4j.dataType()));
        init(shape, stride);
    }


    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape, char ordering) {
        this(slices, shape, Nd4j.getStrides(shape, ordering), ordering);
    }

    public BaseNDArray(List<INDArray> slices, long[] shape, char ordering) {
        this(slices, shape, Nd4j.getStrides(shape, ordering), ordering);
    }


    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape, int[] stride, char ordering) {
        DataBuffer ret = slices.get(0).data().dataType() == (DataBuffer.Type.FLOAT)
                ? Nd4j.createBuffer(new float[ArrayUtil.prod(shape)])
                : Nd4j.createBuffer(new double[ArrayUtil.prod(shape)]);
        this.data = ret;
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(ArrayUtil.toLongArray(shape), ArrayUtil.toLongArray(stride),
                Shape.elementWiseStride(shape, stride, ordering == 'f'), ordering, slices.get(0).dataType()));
        init(shape, stride);
        //    Shape.setElementWiseStride(this.shapeInfo(),Shape.elementWiseStride(shape, stride, ordering == 'f'));

        if (slices.get(0).isScalar()) {
            for (int i = 0; i < length(); i++) {
                putScalar(i, slices.get(i).getDouble(0));
            }
        } else {
            for (int i = 0; i < slices(); i++) {
                putSlice(i, slices.get(i));
            }
        }
    }


    public BaseNDArray(List<INDArray> slices, long[] shape, long[] stride, char ordering) {
        DataBuffer ret = slices.get(0).data().dataType() == (DataBuffer.Type.FLOAT)
                ? Nd4j.createBuffer(new float[ArrayUtil.prod(shape)])
                : Nd4j.createBuffer(new double[ArrayUtil.prod(shape)]);
        this.data = ret;
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride,
                Shape.elementWiseStride(shape, stride, ordering == 'f'), ordering, slices.get(0).dataType()));
        init(shape, stride);
        //    Shape.setElementWiseStride(this.shapeInfo(),Shape.elementWiseStride(shape, stride, ordering == 'f'));

        if (slices.get(0).isScalar()) {
            for (int i = 0; i < length(); i++) {
                putScalar(i, slices.get(i).getDouble(0));
            }
        } else {
            for (int i = 0; i < slices(); i++) {
                putSlice(i, slices.get(i));
            }
        }
    }

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param ordering
     */
    public BaseNDArray(float[] data, int[] shape, int[] stride, char ordering) {
        this(data, shape, stride, 0, ordering);
    }

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     * @param ordering
     */
    public BaseNDArray(float[] data, int[] shape, int[] stride, long offset, char ordering) {
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(ArrayUtil.toLongArray(shape), ArrayUtil.toLongArray(stride),
                Shape.elementWiseStride(shape, stride, ordering == 'f'), ordering, Nd4j.dataType()));
        if (data != null && data.length > 0) {

            val perfD = PerformanceTracker.getInstance().helperStartTransaction();

            this.data = internalCreateBuffer(data, offset);

            PerformanceTracker.getInstance().helperRegisterTransaction(0, perfD, data.length * Nd4j.sizeOfDataType(), MemcpyDirection.HOST_TO_HOST);

            if (offset >= data.length)
                throw new IllegalArgumentException("invalid offset: must be < data.length");
        }

        init(shape, stride);
    }

    public BaseNDArray(float[] data, long[] shape, long[] stride, long offset, char ordering) {
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride,
                Shape.elementWiseStride(shape, stride, ordering == 'f'), ordering, Nd4j.dataType()));
        if (data != null && data.length > 0) {
            this.data = Nd4j.createBuffer(data, offset);
            if (offset >= data.length)
                throw new IllegalArgumentException("invalid offset: must be < data.length");
        }

        init(shape, stride);
    }

    public BaseNDArray(double[] data, long[] shape, long[] stride, long offset, char ordering) {
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride,
                Shape.elementWiseStride(shape, stride, ordering == 'f'), ordering, Nd4j.dataType()));
        if (data != null && data.length > 0) {
            this.data = Nd4j.createBuffer(data, offset);
            if (offset >= data.length)
                throw new IllegalArgumentException("invalid offset: must be < data.length");
        }

        init(shape, stride);
    }

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     */
    public BaseNDArray(DataBuffer data, int[] shape, int[] stride, long offset) {
        this.data = Nd4j.createBuffer(data, offset, ArrayUtil.prodLong(shape));
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(ArrayUtil.toLongArray(shape), ArrayUtil.toLongArray(stride), Shape.elementWiseStride(shape, stride, Nd4j.order() == 'f'), Nd4j.order(), Nd4j.dataType()));
        init(shape, stride);
        //  Shape.setElementWiseStride(this.shapeInfo(),Shape.elementWiseStride(shape, stride, Nd4j.order() == 'f'));


    }

    /**
     *
     * @param data
     * @param shape
     * @param strides
     */
    public BaseNDArray(int[] data, int[] shape, int[] strides) {
        this(internalCreateBuffer(data), shape, strides);
    }

    /**
     *
     * @param data
     * @param shape
     */
    public BaseNDArray(DataBuffer data, int[] shape) {
        this(data, shape, Nd4j.getStrides(shape, Nd4j.order()), 0, Nd4j.order());
    }

    public BaseNDArray(DataBuffer data, long[] shape) {
        this(data, shape, Nd4j.getStrides(shape, Nd4j.order()), 0, Nd4j.order());
    }


    /**
     *
     * @param buffer
     * @param shape
     * @param offset
     */
    public BaseNDArray(DataBuffer buffer, int[] shape, long offset) {
        this(Nd4j.createBuffer(buffer, offset, ArrayUtil.prodLong(shape)), shape, Nd4j.getStrides(shape), offset,
                Nd4j.order());
    }

    /**
     *
     * @param buffer
     * @param shape
     * @param ordering
     */
    public BaseNDArray(DataBuffer buffer, int[] shape, char ordering) {
        this(buffer, shape, Nd4j.getStrides(shape, ordering), 0, ordering);
    }

    public BaseNDArray(DataBuffer buffer, long[] shape, char ordering) {
        this(buffer, shape, Nd4j.getStrides(shape, ordering), 0, ordering);
    }

    /**
     *
     * @param data
     * @param shape
     * @param ordering
     */
    public BaseNDArray(double[] data, int[] shape, char ordering) {
        this(internalCreateBuffer(data), shape, ordering);
    }

    public BaseNDArray(double[] data, long[] shape, char ordering) {
        this(Nd4j.createBuffer(data), shape, ordering);
    }

    public BaseNDArray(float[] data, long[] shape, char ordering) {
        this(Nd4j.createBuffer(data), shape, ordering);
    }

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     * @param ordering
     */
    public BaseNDArray(double[] data, int[] shape, int[] stride, long offset, char ordering) {
        this(internalCreateBuffer(data, offset), shape, stride, offset, ordering);
    }

    /**
     *
     * @param data
     * @param order
     */
    public BaseNDArray(float[] data, char order) {
        this(internalCreateBuffer(data), order);
    }

    protected static DataBuffer internalCreateBuffer(float[] data) {
        val perfX = PerformanceTracker.getInstance().helperStartTransaction();

        val buffer = Nd4j.createBuffer(data);
        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, data.length * Nd4j.sizeOfDataType(), MemcpyDirection.HOST_TO_HOST);

        return buffer;
    }

    protected static DataBuffer internalCreateBuffer(double[] data) {
        val perfX = PerformanceTracker.getInstance().helperStartTransaction();

        val buffer = Nd4j.createBuffer(data);
        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, data.length * Nd4j.sizeOfDataType(), MemcpyDirection.HOST_TO_HOST);

        return buffer;
    }

    protected static DataBuffer internalCreateBuffer(int[] data) {
        val perfX = PerformanceTracker.getInstance().helperStartTransaction();

        val buffer = Nd4j.createBuffer(data);
        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, data.length * Nd4j.sizeOfDataType(), MemcpyDirection.HOST_TO_HOST);

        return buffer;
    }

    protected static DataBuffer internalCreateBuffer(float[] data, long offset) {
        val perfX = PerformanceTracker.getInstance().helperStartTransaction();

        val buffer = Nd4j.createBuffer(data, offset);
        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, data.length * Nd4j.sizeOfDataType(), MemcpyDirection.HOST_TO_HOST);

        return buffer;
    }

    protected static DataBuffer internalCreateBuffer(double[] data, long offset) {
        val perfX = PerformanceTracker.getInstance().helperStartTransaction();

        val buffer = Nd4j.createBuffer(data, offset);
        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, data.length * Nd4j.sizeOfDataType(), MemcpyDirection.HOST_TO_HOST);

        return buffer;
    }

    protected static DataBuffer internalCreateBuffer(int[] data, long offset) {
        val perfX = PerformanceTracker.getInstance().helperStartTransaction();

        val buffer = Nd4j.createBuffer(data, offset);
        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, data.length * Nd4j.sizeOfDataType(), MemcpyDirection.HOST_TO_HOST);

        return buffer;
    }

    /**
     *
     * @param floatBuffer
     * @param order
     */
    public BaseNDArray(DataBuffer floatBuffer, char order) {
        this(floatBuffer, new int[] {(int) floatBuffer.length()},
                Nd4j.getStrides(new int[] {(int) floatBuffer.length()}, order), 0, order);
        if (floatBuffer.length() >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Length of buffer can not be >= Integer.MAX_VALUE");
    }

    /**
     *
     * @param buffer
     * @param shape
     * @param strides
     */
    public BaseNDArray(DataBuffer buffer, int[] shape, int[] strides) {
        this(buffer, shape, strides, 0, Nd4j.order());
    }


    /**
     * Create this ndarray with the given data and shape and 0 offset
     *
     * @param data  the data to use
     * @param shape the shape of the ndarray
     */
    public BaseNDArray(float[] data, int[] shape) {
        this(data, shape, 0);
    }


    /**
     *
     * @param data
     * @param shape
     * @param offset
     */
    public BaseNDArray(float[] data, int[] shape, long offset) {
        this(data, shape, offset, Nd4j.order());

    }

    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public BaseNDArray(int[] shape, int[] stride, long offset) {
        this(new float[ArrayUtil.prod(shape)], shape, stride, offset, Nd4j.order());
    }

    public BaseNDArray(long[] shape, long[] stride, long offset) {
        this(new float[ArrayUtil.prod(shape)], shape, stride, offset, Nd4j.order());
    }

    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public BaseNDArray(int[] shape, int[] stride) {
        this(shape, stride, 0);
    }

    /**
     *
     * @param shape
     * @param offset
     */
    public BaseNDArray(int[] shape, long offset) {
        this(shape, Nd4j.getStrides(shape), offset);
    }

    /**
     *
     * @param shape
     * @param ordering
     */
    public BaseNDArray(int[] shape, char ordering) {
        this(shape, 0, ordering);
    }


    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public BaseNDArray(int newRows, int newColumns) {
        this(newRows, newColumns, Nd4j.order());
    }

    public BaseNDArray(long newRows, long newColumns) {
        this(newRows, newColumns, Nd4j.order());
    }


    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape) {
        this(slices, shape, Nd4j.order());
    }

    public BaseNDArray(List<INDArray> slices, long[] shape) {
        this(slices, shape, Nd4j.order());
    }

    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the ndarray
     */
    public BaseNDArray(List<INDArray> slices, int[] shape, int[] stride) {
        this(slices, shape, stride, Nd4j.order());
    }

    public BaseNDArray(List<INDArray> slices, long[] shape, long[] stride) {
        this(slices, shape, stride, Nd4j.order());
    }

    /**
     *
     * @param data
     * @param shape
     * @param stride
     */
    public BaseNDArray(float[] data, int[] shape, int[] stride) {
        this(data, shape, stride, Nd4j.order());
    }


    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     */
    public BaseNDArray(float[] data, int[] shape, int[] stride, long offset) {
        this(data, shape, stride, offset, Nd4j.order());
    }

    public BaseNDArray(double[] data, long[] shape, long[] stride, long offset) {
        this(data, shape, stride, offset, Nd4j.order());
    }

    public BaseNDArray(float[] data, long[] shape, long[] stride, long offset) {
        this(data, shape, stride, offset, Nd4j.order());
    }

    /**
     *
     * @param data
     */
    public BaseNDArray(float[] data) {
        this(Nd4j.createBuffer(data));
    }


    /**
     * Initialize the ndarray
     * with the given data
     * @param data
     */
    public BaseNDArray(float[][] data) {
        this(data, Nd4j.order());
    }

    /**
     *
     * @param data
     * @param ordering
     */
    public BaseNDArray(float[][] data, char ordering) {
        this(internalCreateBuffer(ordering == 'c' ? ArrayUtil.flatten(data) : ArrayUtil.flattenF(data)),
                new int[] {data.length, data[0].length},
                Nd4j.getStrides(new int[] {data.length, data[0].length}, ordering), 0, ordering);

        for (int r = 0; r < rows(); r++) {
            assert (data[r].length == columns());
        }
    }



    /**
     * Constructor for stride and offset
     *
     * @param buffer
     * @param shape
     * @param offset
     * @param ordering
     */
    public BaseNDArray(DataBuffer buffer, int[] shape, long offset, char ordering) {
        this(buffer, shape, Nd4j.getStrides(shape, ordering), offset, ordering);
    }

    public BaseNDArray(double[] data, int[] shape, int[] stride, long offset) {
        this(internalCreateBuffer(data), shape, stride, offset);
    }


    @Override
    @Deprecated
    public void setWrapAround(boolean wrapAround) {
        throw new UnsupportedOperationException();
    }

    @Override
    @Deprecated
    public boolean isWrapAround() {
        throw new UnsupportedOperationException();
    }

    /**
     * Returns whether the ndarray is valid or not
     * @return true if the ndarray is valid
     * false otherwise
     */
    @Deprecated
    public boolean isValid() {
        try {
            linearIndex(length() - 1);
        } catch (Exception e) {
            return false;
        }
        return true;
    }

    @Override
    @Deprecated
    public INDArray linearViewColumnOrder() {
        return this;
    }

    protected INDArray create(DataBuffer data, int[] shape, long offset) {
        return Nd4j.create(data, shape, offset);
    }



    /**
     * Returns a linear view reference of shape
     * 1,length(ndarray)
     *
     * @return the linear view of this ndarray
     * @deprecated Linear views are not always possible. Use reshape(array.length()) or reshape(1,array.length())
     */
    @Deprecated
    @Override
    public INDArray linearView() {
        return reshape(this.ordering(), 1, this.length());
    }

    @Deprecated
    @Override
    public void resetLinearView() {

    }



    @Override
    public int elementWiseStride() {
        /*
        if(Shape.elementWiseStride(shapeInfo()) < 0 && !attemptedToFindElementWiseStride) {
            INDArray reshapeAttempt = Shape.newShapeNoCopy(this,new int[]{1,length()}, ordering() == 'f');
            if(reshapeAttempt != null && reshapeAttempt.elementWiseStride() > 0) {
               Shape.setElementWiseStride(shapeInfo(), reshapeAttempt.stride(-1));
               this.shapeInformation = Nd4j.getShapeInfoProvider().createShapeInformation(shape(), stride(), offset(),reshapeAttempt.stride(-1), ordering());
            }
            attemptedToFindElementWiseStride = true;
        
        }
        */
        return Shape.elementWiseStride(shapeInfoDataBuffer());
    }

    @Override
    public int elementStride() {
        return 1;
    }

    @Override
    @Deprecated
    public int majorStride() {
        return stride(-1);
    }

    @Override
    @Deprecated
    public int secondaryStride() {
        return majorStride();
    }

    @Override
    public long tensorssAlongDimension(int... dimension) {
        if (dimension == null || dimension.length == 0)
            throw new IllegalArgumentException("Invalid input: dimensions not specified (null or length 0)");
        if (dimension.length >= rank() || dimension.length == 1 && dimension[0] == Integer.MAX_VALUE)
            return 1;
        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] < 0)
                dimension[i] += rank();
        long[] tensorShape = ArrayUtil.keep(shape(), dimension);
        long len = ArrayUtil.prodLong(tensorShape);
        if (len == 0)
            throw new IllegalStateException("Illegal length found after removing index");
        long length = length();
        if (length / len >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Tensors along dimension can not be >= Integer.MAX_VALUE");
        return length / len;
    }

    @Override
    public INDArray tensorAlongDimension(int index, int... dimension) {
        if (dimension == null || dimension.length == 0)
            throw new IllegalArgumentException("Invalid input: dimensions not specified (null or length 0)");

        if (dimension.length >= rank()  || dimension.length == 1 && dimension[0] == Integer.MAX_VALUE)
            return this;
        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] < 0)
                dimension[i] += rank();

        //dedup
        if (dimension.length > 1)
            dimension = Ints.toArray(new ArrayList<>(new TreeSet<>(Ints.asList(dimension))));

        if (dimension.length > 1) {
            Arrays.sort(dimension);
        }

        long tads = tensorssAlongDimension(dimension);
        if (index >= tads)
            throw new IllegalArgumentException("Illegal index " + index + " out of tads " + tads);


        if (dimension.length == 1) {
            if (dimension[0] == 0 && isColumnVector()) {
                return this.transpose();
            } else if (dimension[0] == 1 && isRowVector()) {
                return this;
            }
        }

        Pair<DataBuffer, DataBuffer> tadInfo =
                Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(this, dimension);
        DataBuffer shapeInfo = tadInfo.getFirst();
        val shape = Shape.shape(shapeInfo);
        val stride = Shape.stride(shapeInfo).asLong();
        long offset = offset() + tadInfo.getSecond().getLong(index);
        INDArray toTad = Nd4j.create(data(), shape, stride, offset);
        BaseNDArray baseNDArray = (BaseNDArray) toTad;

        //preserve immutability
        char newOrder = Shape.getOrder(shape, stride, 1);

        int ews = baseNDArray.shapeInfoDataBuffer().getInt(baseNDArray.shapeInfoDataBuffer().length() - 2);

        //TAD always calls permute. Permute EWS is always -1. This is not true
        // for row vector shapes though.
        if (!Shape.isRowVectorShape(baseNDArray.shapeInfoDataBuffer()))
            ews = -1;

        // we create new shapeInfo with possibly new ews & order
        /**
         * NOTE HERE THAT ZERO IS PRESET FOR THE OFFSET AND SHOULD STAY LIKE THAT.
         * Zero is preset for caching purposes.
         * We don't actually use the offset defined in the
         * shape info data buffer.
         * We calculate and cache the offsets separately.
         *
         */
        baseNDArray.setShapeInformation(
                Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride,  ews, newOrder, this.dataType()));

        return toTad;
    }

    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to getScalar
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    @Override
    public INDArray javaTensorAlongDimension(int index, int... dimension) {
        return doTad(index, dimension);
    }

    private void setShapeInformation(Pair<DataBuffer, long[]> shapeInfo) {
        this.shapeInformation = shapeInfo.getFirst();
        this.jvmShapeInfo = new JvmShapeInfo(shapeInfo.getSecond());
    }


    private INDArray doTad(int index, int... dimension) {
        if (dimension == null || dimension.length == 0)
            throw new IllegalArgumentException("Invalid input: dimensions not specified (null or length 0)");

        if (dimension.length >= rank())
            return this;
        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] < 0)
                dimension[i] += rank();

        if (dimension.length > 1)
            Arrays.sort(dimension);

        long tads = tensorssAlongDimension(dimension);
        if (index >= tads)
            throw new IllegalArgumentException("Illegal index " + index + " out of tads " + tads);


        if (dimension.length == 1) {
            if (dimension[0] == 0 && isColumnVector()) {
                return this.transpose();
            } else if (dimension[0] == 1 && isRowVector()) {
                return this;
            }
        }


        long[] tensorShape = ArrayUtil.keep(shape(), dimension);
        int[] reverseDimensions = ArrayUtil.reverseCopy(dimension);
        int[] remove = ArrayUtil.removeIndex(ArrayUtil.range(0, rank()), dimension);
        int[] newPermuteDims = Ints.concat(remove, reverseDimensions);
        int[] finalPermuteDims = tadFinalPermuteDimensions[dimension.length];

        INDArray permuted = permute(newPermuteDims);
        long sliceIdx = NDArrayMath.sliceOffsetForTensor(index, permuted, tensorShape);

        INDArray ret2 = permuted.slice(sliceIdx);
        if (dimension.length == tensorShape.length && ArrayUtil.prodLong(tensorShape) == ret2.length()) {
            if (dimension.length == 1 && ret2.isRowVector())
                return ret2;
            if (finalPermuteDims.length != ret2.rank()) {
                finalPermuteDims = new int[ret2.rank()];
                int count = 0;
                for (int i = finalPermuteDims.length - 1; i >= 0; i--)
                    finalPermuteDims[count++] = i;
            }
            return ret2.permutei(finalPermuteDims);
        }


        int length = ArrayUtil.prod(tensorShape);
        int tensorLength = ArrayUtil.prod(tensorShape);
        long offset = index * tensorLength / NDArrayMath.lengthPerSlice(ret2);

        if (sliceIdx == 0 && length == NDArrayMath.lengthPerSlice(ret2)) {
            // FIXME: LONG
            ret2 = ret2.slice((int) offset);
            if (dimension.length == 1 && ret2.isRowVectorOrScalar())
                return ret2;
            return ret2.permutei(finalPermuteDims);
        }

        else if (length == NDArrayMath.lengthPerSlice(ret2)) {
            offset -= ret2.slices() * (offset / ret2.slices());

            // FIXME: LONG
            ret2 = ret2.slice((int) offset);
            if (dimension.length == 1 && ret2.isRowVectorOrScalar())
                return ret2;
            return ret2.permutei(finalPermuteDims);
        }

        while (ret2.length() > length) {
            sliceIdx = NDArrayMath.sliceOffsetForTensor(index, ret2, tensorShape);
            sliceIdx -= ret2.slices() * (sliceIdx / ret2.slices());
            ret2 = ret2.slice(sliceIdx);
        }

        if (dimension.length == 1 && ret2.isRowVectorOrScalar())
            return ret2;

        return ret2.permutei(finalPermuteDims);
    }



    /**
     * Returns the number of possible vectors for a given dimension
     *
     * @param dimension the dimension to calculate the number of vectors for
     * @return the number of possible vectors along a dimension
     */
    @Override
    public long vectorsAlongDimension(int dimension) {
        if (dimension == 0 && isVector() || isRowVectorOrScalar())
            return 1;
        if (size(dimension) == 1 && !isVector()) {
            for (int i = dimension; i < rank(); i++) {
                if (size(i) != 1)
                    return vectorsAlongDimension(i);
            }

            return length();

        } else if (size(0) == 1 && !isVectorOrScalar()) {
            int realDimension = rank() - getLeadingOnes();
            long length = length();
            if (length / size(realDimension) >= Integer.MAX_VALUE)
                throw new IllegalArgumentException("Vectors along dimension can not be >= Integer.MAX_VALUE");
            return length / size(realDimension);
        }

        long length = length();

        if (dimension >= jvmShapeInfo.rank) {
            if (length / size(jvmShapeInfo.rank - 1) >= Integer.MAX_VALUE)
                throw new IllegalArgumentException("Vectors along dimension can not be >= Integer.MAX_VALUE");
            return (int) (length / size(jvmShapeInfo.rank - 1));
        }
        if (length / size(dimension) >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Vectors along dimension can not be >= Integer.MAX_VALUE");
        return length / size(dimension);
    }

    /**
     * Get the vector along a particular dimension
     *
     * @param index     the index of the vector to get
     * @param dimension the dimension to get the vector from
     * @return the vector along a particular dimension
     */
    @Override
    public INDArray vectorAlongDimension(int index, int dimension) {
        if (dimension < 0)
            dimension = jvmShapeInfo.getRank() + dimension;

        //return the whole thing
        if (dimension == jvmShapeInfo.getRank() - 1 && size(dimension) == 1 && rank() > 2
                || rank() > 2 && dimension == 0 && size(dimension) == 1) {
            return this;
        }

        INDArray ret = tensorAlongDimension(index, dimension);
        if (isMatrix() && ret.isVector() && dimension == 1 && !ret.isRowVector())
            return ret.reshape(ArrayUtil.reverseCopy(ret.shape()));
        else if (isMatrix() && ret.isVector() && dimension == 0 && !ret.isColumnVector())
            return ret.reshape(ArrayUtil.reverseCopy(ret.shape()));
        return ret;
    }

    @Override
    public void setOrder(char order) {
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape(), stride(), elementWiseStride(), order, this.dataType()));
    }

    @Override
    public void setShape(long[] shape) {
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride(), elementWiseStride(), ordering(), this.dataType()));
    }

    @Override
    public void setStride(long[] stride) {
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape(), stride, elementWiseStride(), ordering(), this.dataType()));
    }

    @Override
    public void setShapeAndStride(int[] shape, int[] stride) {
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(ArrayUtil.toLongArray(shape), ArrayUtil.toLongArray(stride),  -1, ordering(), this.dataType()));
    }


    /**
     * Cumulative sum along a dimension
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    @Override
    public INDArray cumsumi(int dimension) {

        if (isVector()) {
            double s = 0.0;
            for (int i = 0; i < length(); i++) {
                s += getDouble(i);
                putScalar(i, s);
            }
        } else if (dimension == Integer.MAX_VALUE) {
            INDArray flattened = ravel();
            double prevVal = flattened.getDouble(0);
            for (int i = 1; i < flattened.length(); i++) {
                double d = prevVal + flattened.getDouble(i);
                flattened.putScalar(i, d);
                prevVal = d;
            }

            return flattened;
        } else {
            for (int i = 0; i < vectorsAlongDimension(dimension); i++) {
                INDArray vec = vectorAlongDimension(i, dimension);
                vec.cumsumi(0);

            }
        }


        return this;
    }

    @Override
    public Number normmaxNumber() {
        return normmax(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number norm2Number() {
        return norm2(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number norm1Number() {
        return norm1(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number stdNumber() {
        return std(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number prodNumber() {
        return prod(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number meanNumber() {
        return mean(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number ameanNumber() {
        return amean(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number varNumber() {
        return var(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number maxNumber() {
        return max(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number amaxNumber() {
        return amax(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number minNumber() {
        return min(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number aminNumber() {
        return amin(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number scan(Condition condition) {
        MatchCondition op = new MatchCondition(this, condition);
        return Nd4j.getExecutioner().exec(op, Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Number sumNumber() {
        return sum(Integer.MAX_VALUE).getDouble(0);
    }

    /**
     * Returns entropy value for this INDArray
     * @return
     */
    @Override
    public Number entropyNumber() {
        return entropy(Integer.MAX_VALUE).getDouble(0);
    }

    /**
     * Returns non-normalized Shannon entropy value for this INDArray
     * @return
     */
    @Override
    public Number shannonEntropyNumber() {
        return shannonEntropy(Integer.MAX_VALUE).getDouble(0);
    }


    /**
     * Returns log entropy value for this INDArray
     * @return
     */
    @Override
    public Number logEntropyNumber() {
        return logEntropy(Integer.MAX_VALUE).getDouble(0);
    }

    /**
     * Cumulative sum along a dimension (in place)
     *
     * @param dimension the dimension to perform cumulative sum along
     * @return the cumulative sum along the specified dimension
     */
    @Override
    public INDArray cumsum(int dimension) {
        return dup().cumsumi(dimension);
    }

    /**
     * Assign all of the elements in the given
     * ndarray to this ndarray
     *
     * @param arr the elements to assign
     * @return this
     */
    @Override
    public INDArray assign(final INDArray arr) {
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.impl.transforms.Set(this, arr, this, length()));
        return this;

    }

    @Override
    public INDArray putScalar(long i, double value) {
        if (i < 0)
            i += rank();

        // TODO: i'm not sure that rank == 1 has fair shortcut here
        if (isScalar() || rank() == 1) {
            autoProcessScalarCall();
            data.put(i, value);
            return this;
        }

        // we cant raise rank here, if original rank is 1
        if (isRowVector() && rank() == 2) {
            return putScalar(0, i, value);
        } else if (isColumnVector() && rank() == 2) {
            return putScalar(i, 0, value);
        }
        long[] indexes = ordering() == 'c' ? Shape.ind2subC(this, i) : Shape.ind2sub(this, i);
        return putScalar(indexes, value);
    }

    @Override
    public INDArray putScalar(long i, float value) {
        return putScalar(i, (double) value);

    }

    @Override
    public INDArray putScalar(long i, int value) {
        return putScalar(i, (double) value);
    }

    @Override
    public INDArray putScalar(int[] indexes, double value) {
        Nd4j.getCompressor().autoDecompress(this);


        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] < 0)
                indexes[i] += this.size(i);
        }

        if (indexes.length == 1) {
            return putScalar(indexes[0], value);
        } else if (indexes.length == 2) {
            return putScalar(indexes[0], indexes[1], value);
        } else if (indexes.length == 3) {
            return putScalar(indexes[0], indexes[1], indexes[2], value);
        } else if (indexes.length == 4) {
            return putScalar(indexes[0], indexes[1], indexes[2], indexes[3], value);
        } else {
            autoProcessScalarCall();
            long offset = Shape.getOffset(jvmShapeInfo.javaShapeInformation, indexes);
            data.put(offset, value);
        }
        return this;
    }

    @Override
    public INDArray putScalar(long[] indexes, double value) {
        Nd4j.getCompressor().autoDecompress(this);


        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] < 0)
                indexes[i] += size(i);
        }

        if (indexes.length == 1) {
            return putScalar(indexes[0], value);
        } else if (indexes.length == 2) {
            return putScalar(indexes[0], indexes[1], value);
        } else if (indexes.length == 3) {
            return putScalar(indexes[0], indexes[1], indexes[2], value);
        } else if (indexes.length == 4) {
            return putScalar(indexes[0], indexes[1], indexes[2], indexes[3], value);
        } else {
            autoProcessScalarCall();
            long offset = Shape.getOffset(jvmShapeInfo.javaShapeInformation, indexes);
            data.put(offset, value);
        }
        return this;
    }

    @Override
    public INDArray putScalar(long[] indexes, float value) {
        return putScalar(indexes, (double) value);
    }

    @Override
    public INDArray putScalar(long row, long col, double value) {
        Nd4j.getCompressor().autoDecompress(this);
        autoProcessScalarCall();

        if (rank() > 2)
            throw new IllegalStateException("Cannot use putScalar(int,int,double) on a rank " + rank() + " INDArray");
        long offset = Shape.getOffsetUnsafe(jvmShapeInfo.javaShapeInformation, row, col);
        data.put(offset, value);
        return this;
    }

    @Override
    public INDArray putScalar(long dim0, long dim1, long dim2, double value) {
        Nd4j.getCompressor().autoDecompress(this);
        autoProcessScalarCall();

        if (rank() != 3)
            throw new IllegalStateException(
                    "Cannot use putScalar(int,int,int,double) on a rank " + rank() + " INDArray");
        long offset = 0; // Shape.getOffsetUnsafe(javaShapeInformation, dim0, dim1, dim2);
        long size_0 = jvmShapeInfo.javaShapeInformation[1];
        long size_1 = jvmShapeInfo.javaShapeInformation[1 + 1];
        long size_2 = jvmShapeInfo.javaShapeInformation[1 + 2];

        if (size_0 != 1)
            offset += dim0 * jvmShapeInfo.javaShapeInformation[1 + 0 + 3];
        if (size_1 != 1)
            offset += dim1 * jvmShapeInfo.javaShapeInformation[1 + 1 + 3];
        if (size_2 != 1)
            offset += dim2 * jvmShapeInfo.javaShapeInformation[1 + 2 + 3];

        data.put(offset, value);
        return this;
    }

    @Override
    public INDArray putScalar(long dim0, long dim1, long dim2, long dim3, double value) {
        Nd4j.getCompressor().autoDecompress(this);
        autoProcessScalarCall();

        if (rank() != 4)
            throw new IllegalStateException(
                    "Cannot use putScalar(int,int,int,int,double) on a rank " + rank() + " INDArray");
        long offset = Shape.getOffsetUnsafe(jvmShapeInfo.javaShapeInformation, dim0, dim1, dim2, dim3);
        data.put(offset, value);
        return this;
    }


    @Override
    public INDArray putScalar(int[] indexes, float value) {
        return putScalar(indexes, (double) value);
    }

    @Override
    public INDArray putScalar(int[] indexes, int value) {
        return putScalar(indexes, (double) value);
    }

    @Override
    public INDArray putScalar(long[] indexes, int value) {
        return putScalar(indexes, (double) value);
    }

    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a copied ndarray with the given
     * binary conditions
     */
    @Override
    public INDArray eps(Number other) {
        return dup().epsi(other);
    }

    /**
     * Returns an ndarray with 1 if the element is epsilon equals
     *
     * @param other the number to compare
     * @return a ndarray with the given
     * binary conditions
     */
    @Override
    public INDArray epsi(Number other) {
        INDArray otherArr = Nd4j.valueArrayOf(shape(), other.doubleValue());
        return epsi(otherArr);
    }

    /**
     * epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    @Override
    public INDArray eps(INDArray other) {
        return dup().epsi(other);
    }

    /**
     * In place epsilon equals than comparison:
     * If the given number is less than the
     * comparison number the item is 0 otherwise 1
     *
     * @param other the number to compare
     * @return
     */
    @Override
    public INDArray epsi(INDArray other) {
        Nd4j.getExecutioner().exec(new Eps(this, other, this, length()));
        return this;
    }

    @Override
    public INDArray lt(Number other) {
        return dup().lti(other);
    }

    @Override
    public INDArray lte(Number other) {
        return dup().ltei(other);
    }

    @Override
    public INDArray lti(Number other) {

        Nd4j.getExecutioner().exec(new ScalarLessThan(this, other));
        return this;
    }

    @Override
    public INDArray ltei(Number other) {

        Nd4j.getExecutioner().exec(new ScalarLessThanOrEqual(this, other));
        return this;
    }

    @Override
    public INDArray eq(Number other) {
        return dup().eqi(other);
    }

    @Override
    public INDArray eqi(Number other) {

        Nd4j.getExecutioner().exec(new ScalarEquals(this, other));
        return this;
    }

    @Override
    public INDArray gt(Number other) {
        return dup().gti(other);
    }

    @Override
    public INDArray gte(Number other) {
        return dup().gtei(other);
    }

    @Override
    public INDArray gtei(Number other) {
        Nd4j.getExecutioner().exec(new ScalarGreaterThanOrEqual(this, other));
        return this;
    }

    @Override
    public INDArray gti(Number other) {
        Nd4j.getExecutioner().exec(new ScalarGreaterThan(this, other));
        return this;
    }

    @Override
    public INDArray lt(INDArray other) {
        return dup().lti(other);
    }

    @Override
    public INDArray lti(INDArray other) {
        Nd4j.getExecutioner().exec(new OldLessThan(this, other, this, length()));
        return this;
    }

    @Override
    public INDArray neq(Number other) {
        return dup().neqi(other);
    }

    @Override
    public INDArray neqi(Number other) {
        Nd4j.getExecutioner().exec(new ScalarNotEquals(this, other));
        return this;
    }

    @Override
    public INDArray neq(INDArray other) {
        return dup().neqi(other);
    }

    @Override
    public INDArray neqi(INDArray other) {
        Nd4j.getExecutioner().exec(new OldNotEqualTo(this, other, this, length()));
        return this;
    }

    @Override
    public INDArray eq(INDArray other) {
        return dup().eqi(other);
    }

    @Override
    public INDArray eqi(INDArray other) {
        Nd4j.getExecutioner().exec(new OldEqualTo(this, other, this, length()));
        return this;
    }

    @Override
    public INDArray gt(INDArray other) {
        return dup().gti(other);
    }

    @Override
    public INDArray gti(INDArray other) {
        Nd4j.getExecutioner().exec(new OldGreaterThan(this, other, this, length()));
        return this;
    }

    /**
     * Negate each element.
     */
    @Override
    public INDArray neg() {
        return Nd4j.getExecutioner().exec(new Negative(this, Nd4j.createUninitialized(this.shape(), this.ordering())))
                .z();
    }

    /**
     * Negate each element (in-place).
     */
    @Override
    public INDArray negi() {
        Nd4j.getExecutioner().exec(new Negative(this));
        return this;
    }

    @Override
    public INDArray rdiv(Number n, INDArray result) {
        return rdivi(n, result);
    }

    @Override
    public INDArray rdivi(Number n, INDArray result) {

        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;
        Nd4j.getExecutioner().exec(new ScalarReverseDivision(this, null, result, result.length(), n));
        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);
        return result;
    }

    @Override
    public INDArray rsub(Number n, INDArray result) {
        return rsubi(n, result);
    }

    @Override
    public INDArray rsubi(Number n, INDArray result) {

        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;

        Nd4j.getExecutioner().exec(new ScalarReverseSubtraction(this, null, result, result.lengthLong(), n));

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);
        return result;
    }

    @Override
    public INDArray div(Number n, INDArray result) {
        return divi(n, result);
    }

    @Override
    public INDArray divi(Number n, INDArray result) {

        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;
        Nd4j.getExecutioner().exec(new ScalarDivision(this, null, result, result.lengthLong(), n));


        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    @Override
    public INDArray mul(Number n, INDArray result) {
        return muli(n, result);
    }

    @Override
    public INDArray muli(Number n, INDArray result) {
        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;
        Nd4j.getExecutioner().exec(new ScalarMultiplication(this, null, result, result.lengthLong(), n));

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    @Override
    public INDArray sub(Number n, INDArray result) {
        return subi(n, result);
    }

    @Override
    public INDArray subi(Number n, INDArray result) {

        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;

        Nd4j.getExecutioner().exec(new ScalarSubtraction(this, null, result, result.lengthLong(), n));

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    @Override
    public INDArray add(Number n, INDArray result) {
        return addi(n, result);
    }

    @Override
    public INDArray addi(Number n, INDArray result) {
        if (Double.isNaN(n.doubleValue()))
            n = Nd4j.EPS_THRESHOLD;

        Nd4j.getExecutioner().exec(new ScalarAdd(this, null, result, result.lengthLong(), n));

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }



    /**
     * Returns the element at the specified row/column
     * This will throw an exception if the
     *
     * @param row    the row of the element to return
     * @param column the row of the element to return
     * @return a scalar indarray of the element at this index
     */
    @Override
    public INDArray getScalar(long row, long column) {
        return getScalar(new long[] {row, column});
    }

    @Override
    public INDArray dup() {
        WorkspaceUtils.assertValidArray(this, "Cannot duplicate INDArray");
        if (this.isCompressed() && this.ordering() == Nd4j.order().charValue()) {
            INDArray ret = Nd4j.createArrayFromShapeBuffer(data().dup(), this.shapeInfoDataBuffer());
            ret.markAsCompressed(true);
            return ret;
        }
        Nd4j.getCompressor().autoDecompress(this);
        INDArray ret = Shape.toOffsetZeroCopy(this);
        return ret;
    }

    @Override
    public INDArray dup(char order) {
        WorkspaceUtils.assertValidArray(this, "Cannot duplicate INDArray");
        if (this.isCompressed() && this.ordering() == order) {
            INDArray ret = Nd4j.createArrayFromShapeBuffer(data().dup(), this.shapeInfoDataBuffer());
            ret.markAsCompressed(true);
            return ret;
        }
        Nd4j.getCompressor().autoDecompress(this);
        return Shape.toOffsetZeroCopy(this, order);
    }

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    @Override
    public int getInt(int... indices) {
        return (int) getDouble(indices);
    }

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to get
     * @return the array with the specified elements
     */
    @Override
    public double getDouble(int... indices) {
        autoProcessScalarCall();
        Nd4j.getCompressor().autoDecompress(this);

        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0)
                indices[i] += rank();
        }
        if (indices.length == 1) {
            if (rank() == 1)
                return Shape.getDouble(this, indices[0]);
            else if (isRowVector())
                return Shape.getDouble(this, 0, indices[0]);
            else if (isColumnVector())
                return Shape.getDouble(this, indices[0], 0);
            else if (isScalar() && indices[0] == 0)
                return data().getDouble(0);
            else
                throw new IllegalStateException("Indexes length must be > 1 for non vectors and scalars");
        }
        return Shape.getDouble(this, indices);
    }

    @Override
    public double getDouble(long... indices) {
        autoProcessScalarCall();
        Nd4j.getCompressor().autoDecompress(this);

        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0)
                indices[i] += rank();
        }
        if (indices.length == 1) {
            if (rank() == 1)
                return Shape.getDouble(this, indices[0]);
            else if (isRowVector())
                return Shape.getDouble(this, 0, indices[0]);
            else if (isColumnVector())
                return Shape.getDouble(this, indices[0], 0);
            else if (isScalar() && indices[0] == 0)
                return data().getDouble(0);
            else
                throw new IllegalStateException("Indexes length must be > 1 for non vectors and scalars");
        }
        return Shape.getDouble(this, indices);
    }

    /**
     * Returns the elements at the the specified indices
     *
     * @param indices the indices to get
     * @return the array with the specified elements
     */
    @Override
    public float getFloat(int... indices) {
        return (float) getDouble(indices);
    }

    @Override
    public float getFloat(long... indices) {
        return (float) getDouble(indices);
    }

    /**
     * Test whether a matrix is scalar.
     */
    @Override
    public boolean isScalar() {
        if (isEmpty())
            return false;

        if (jvmShapeInfo.rank == 0) {
            return true;
        } else if (jvmShapeInfo.rank > 2) {
            return false;
        } else if (jvmShapeInfo.rank == 1) {
            return shape()[0] == 1;
        } else if (jvmShapeInfo.rank == 2) {
            return shape()[0] == 1 && shape()[1] == 1 || length() == 1;
        }

        else
            return false;

    }

    /**
     * Inserts the element at the specified index
     *
     * @param indices the indices to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int[] indices, INDArray element) {
        Nd4j.getCompressor().autoDecompress(this);
        if (!element.isScalar())
            throw new IllegalArgumentException("Unable to insert anything but a scalar");
        if (isRowVector() && indices[0] == 0 && indices.length == 2) {
            int ix = 0; //Shape.offset(javaShapeInformation);
            for (int i = 1; i < indices.length; i++)
                ix += indices[i] * stride(i);
            if (ix >= data.length())
                throw new IllegalArgumentException("Illegal indices " + Arrays.toString(indices));
            data.put(ix, element.getDouble(0));
        } else {
            int ix = 0; //Shape.offset(javaShapeInformation);
            for (int i = 0; i < indices.length; i++)
                if (size(i) != 1)
                    ix += indices[i] * stride(i);
            if (ix >= data.length())
                throw new IllegalArgumentException("Illegal indices " + Arrays.toString(indices));
            data.put(ix, element.getDouble(0));
        }


        return this;

    }

    @Override
    public INDArray match(INDArray comp, Condition condition) {
        return Nd4j.getExecutioner().exec(new MatchConditionTransform(this,comp,condition)).z();
    }

    @Override
    public INDArray match(Number comp, Condition condition) {
        return Nd4j.getExecutioner().exec(new MatchConditionTransform(this,comp.doubleValue(),condition)).z();
    }

    @Override
    public INDArray getWhere(INDArray comp, Condition condition) {
        return BooleanIndexing.chooseFrom(new INDArray[]{this,comp},condition);
    }

    @Override
    public INDArray getWhere(Number comp, Condition condition) {
        return BooleanIndexing.chooseFrom(new INDArray[]{this},Arrays.asList(comp.doubleValue()),Collections.<Integer>emptyList(),condition);
    }

    @Override
    public INDArray putWhere(INDArray comp, INDArray put, Condition condition) {
        Nd4j.getCompressor().autoDecompress(this);
        MatchConditionTransform matchCondition = new MatchConditionTransform(this,comp,condition);
        Nd4j.getExecutioner().exec(matchCondition);
        return putWhereWithMask(matchCondition.z(),put);
    }

    @Override
    public INDArray putWhere(Number comp, INDArray put, Condition condition) {
        return putWhere(Nd4j.scalar(comp),put,condition);
    }

    @Override
    public INDArray putWhere(Number comp, Number put, Condition condition) {
        return putWhere(Nd4j.scalar(comp),Nd4j.scalar(put),condition);
    }


    @Override
    public INDArray putWhereWithMask(INDArray mask, INDArray put) {
        INDArray output = dup();
        Nd4j.getExecutioner().exec(new Where(new INDArray[]{mask,this,put},new INDArray[]{output}));
        return output;
    }

    @Override
    public INDArray putWhereWithMask(INDArray mask, Number put) {
        return putWhereWithMask(mask,Nd4j.scalar(put));
    }

    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int i, int j, INDArray element) {
        return put(new int[] {i, j}, element);
    }

    /**
     * Inserts the element at the specified index
     *
     * @param i       the row insert into
     * @param j       the column to insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int i, int j, Number element) {
        return putScalar(new int[] {i, j}, element.doubleValue());
    }

    /**
     * Assigns the given matrix (put) to the specified slice
     *
     * @param slice the slice to assign
     * @param put   the slice to put
     * @return this for chainability
     */
    @Override
    public INDArray putSlice(int slice, INDArray put) {
        Nd4j.getCompressor().autoDecompress(this);


        if (isScalar()) {
            assert put.isScalar() : "Invalid dimension. Can only insert a scalar in to another scalar";
            put(0, put.getScalar(0));
            return this;
        } else if (isVector()) {
            assert put.isScalar() || put.isVector() && put
                    .length() == length() : "Invalid dimension on insertion. Can only insert scalars input vectors";
            if (put.isScalar())
                putScalar(slice, put.getDouble(0));
            else
                for (int i = 0; i < length(); i++)
                    putScalar(i, put.getDouble(i));

            return this;
        }

        assertSlice(put, slice);


        INDArray view = slice(slice);

        if (put.length() == 1)
            putScalar(slice, put.getDouble(0));
        else if (put.isVector())
            for (int i = 0; i < put.length(); i++)
                view.putScalar(i, put.getDouble(i));
        else {
            assert Shape.shapeEquals(view.shape(), put.shape());
            INDArray linear = view;
            INDArray putLinearView = put;
            for (int i = 0; i < linear.length(); i++) {
                linear.putScalar(i, putLinearView.getDouble(i));
            }


        }

        return this;

    }

    protected void assertSlice(INDArray put, long slice) {

        assert slice <= slices() : "Invalid slice specified " + slice;
        long[] sliceShape = put.shape();
        if (Shape.isRowVectorShape(sliceShape)) {
            return;
        } else {
            long[] requiredShape = ArrayUtil.removeIndex(shape(), 0);

            //no need to compare for scalar; primarily due to shapes either being [1] or length 0
            if (put.isScalar())
                return;

            if (isVector() && put.isVector() && put.length() < length())
                return;
            //edge case for column vectors
            if (Shape.isColumnVectorShape(sliceShape))
                return;
            if (!Shape.shapeEquals(sliceShape, requiredShape) && !Shape.isRowVectorShape(requiredShape)
                    && !Shape.isRowVectorShape(sliceShape))
                throw new IllegalStateException(String.format("Invalid shape size of %s . Should have been %s ",
                        Arrays.toString(sliceShape), Arrays.toString(requiredShape)));
        }
    }

    /**
     * Returns true if this ndarray is 2d
     * or 3d with a singleton element
     *
     * @return true if the element is a matrix, false otherwise
     */
    public boolean isMatrix() {
        int rank = rank();
        return (rank == 2 && (size(0) != 1 && size(1) != 1));
    }


    @Override
    @Deprecated
    public long index(long row, long column) {
        if (!isMatrix()) {
            if (isColumnVector()) {
                long idx = linearIndex(row);
                return idx;
            } else if (isRowVector()) {
                long idx = linearIndex(column);
                return idx;
            } else
                throw new IllegalStateException("Unable to get row/column from a non matrix");
        }


        return (row * stride(0) + column * stride(1));
    }

    protected INDArray newShape(long[] newShape, char ordering) {

        return Nd4j.create(data(), newShape, stride(), 0, ordering);
    }

    protected INDArray create(DataBuffer data, int[] newShape, int[] newStrides, long offset, char ordering) {
        return Nd4j.create(data, newShape, newStrides, offset, ordering);
    }

    protected INDArray create(DataBuffer data, int[] newShape, int[] newStrides, long offset) {
        return Nd4j.create(data, newShape, newStrides, offset);
    }

    protected INDArray create(int[] shape) {
        return Nd4j.create(shape, getStrides(shape, Nd4j.order()), 0);
    }

    protected INDArray create(int[] shape, int[] strides, long offset) {
        return Nd4j.create(shape, strides, offset);
    }

    protected int[] getStrides(int[] shape, char ordering) {
        return Nd4j.getStrides(shape, ordering);
    }


    /**
     * Returns the square of the Euclidean distance.
     */
    @Override
    public double squaredDistance(INDArray other) {
        double d2 = distance2(other);
        return d2 * d2;
    }

    /**
     * Returns the (euclidean) distance.
     */
    @Override
    public double distance2(INDArray other) {
        Nd4j.getCompressor().autoDecompress(this);
        return Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(this, other)).getFinalResult().doubleValue();
    }

    /**
     * Returns the (1-norm) distance.
     */
    @Override
    public double distance1(INDArray other) {
        Nd4j.getCompressor().autoDecompress(this);
        return Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(this, other)).getFinalResult().doubleValue();
    }



    @Override
    public INDArray get(INDArray indices) {
        if(indices.rank() > 2) {
            throw new ND4JIllegalArgumentException("Indices must be a vector or matrix.");
        }

        if(indices.rows() == rank()) {
            INDArray ret = Nd4j.create(indices.columns());

            for(int i = 0; i < indices.columns(); i++) {
                int[] specifiedIndex = indices.getColumn(i).dup().data().asInt();
                val v = getDouble(specifiedIndex);
                ret.putScalar(i, v);
            }

            return ret;
        }
        else {
            List<INDArray> arrList = new ArrayList<>();

            if(indices.isMatrix() || indices.isColumnVector()
                    || (indices.isScalar() && indices.rank() == 2)) { // we need this for compatibility with legacy code
                for(int i = 0; i < indices.rows(); i++) {
                    if(i == 0)  {
                        INDArray row = indices.getRow(i);
                        for(int j = 0; j < row.length(); j++) {
                            arrList.add(slice(row.getInt(j)));
                        }
                    }
                    else {
                        INDArray row = indices.slice(i);
                        for(int j = 0; j < row.length(); j++) {
                            INDArray put = arrList.get(j).slice(row.getInt(j));
                            put = put.reshape(Longs.concat(new long[]{1},put.shape()));
                            arrList.set(j,put);
                        }
                    }

                }
            }
            else if(indices.isRowVector()) {
                for(int i = 0; i < indices.length(); i++) {
                    INDArray add = slice(indices.getInt(i));
                    add = add.reshape(Longs.concat(new long[] {1,},add.shape()));
                    arrList.add(add);
                }
            }

            return Nd4j.concat(0,arrList.toArray(new INDArray[arrList.size()]));

        }


    }

    @Override
    public INDArray get(List<List<Integer>> indices) {
        INDArrayIndex[] indArrayIndices = new INDArrayIndex[indices.size()];
        for(int i = 0; i < indArrayIndices.length; i++) {
            indArrayIndices[i] = new SpecifiedIndex(Ints.toArray(indices.get(i)));
        }

        boolean hasNext = true;
        Generator<List<List<Long>>> iterate = SpecifiedIndex.iterate(indArrayIndices);
        List<INDArray> resultList = new ArrayList<>();
        while(hasNext) {
            try {
                List<List<Long>> next = iterate.next();
                int[][] nextArr = new int[next.size()][];
                for(int i = 0; i < next.size(); i++) {
                    nextArr[i] = Ints.toArray(next.get(i));
                }

                int[] curr = Ints.concat(nextArr);
                INDArray currSlice = this;
                for(int j = 0; j < curr.length; j++) {
                    currSlice = currSlice.slice(curr[j]);
                }

                //slice drops the first dimension, this adds a 1 to match normal numpy behavior
                currSlice = currSlice.reshape(Longs.concat(new long[]{1},currSlice.shape()));

                resultList.add(currSlice);


            }
            catch(NoSuchElementException e) {
                hasNext = false;
            }
        }




        return Nd4j.concat(0,resultList.toArray(new INDArray[resultList.size()]));
    }

    @Override
    public INDArray put(List<List<Integer>> indices, INDArray element) {
        INDArrayIndex[] indArrayIndices = new INDArrayIndex[indices.size()];
        for(int i = 0; i < indArrayIndices.length; i++) {
            indArrayIndices[i] = new SpecifiedIndex(Ints.toArray(indices.get(i)));
        }

        boolean hasNext = true;
        Generator<List<List<Long>>> iterate = SpecifiedIndex.iterate(indArrayIndices);

        if(indices.size() == rank()) {
            NdIndexIterator ndIndexIterator = new NdIndexIterator(element.shape());

            while(hasNext) {
                try {
                    List<List<Long>> next = iterate.next();
                    int[][] nextArr = new int[next.size()][];
                    for(int i = 0; i < next.size(); i++) {
                        nextArr[i] = Ints.toArray(next.get(i));
                    }

                    int[] curr = Ints.concat(nextArr);
                    putScalar(curr,element.getDouble(ndIndexIterator.next()));

                }
                catch(NoSuchElementException e) {
                    hasNext = false;
                }
            }

        }
        else {
            if(indices.size() >= 2) {
                while(hasNext) {
                    try {
                        List<List<Long>> next = iterate.next();
                        int[][] nextArr = new int[next.size()][];
                        for(int i = 0; i < next.size(); i++) {
                            nextArr[i] = Ints.toArray(next.get(i));
                        }

                        int[] curr = Ints.concat(nextArr);
                        INDArray currSlice = this;
                        for(int j = 0; j < curr.length; j++) {
                            currSlice = currSlice.slice(curr[j]);
                        }

                        Nd4j.getExecutioner().exec(new Assign(new INDArray[]{currSlice,element},new INDArray[]{currSlice}));

                    }
                    catch(NoSuchElementException e) {
                        hasNext = false;
                    }
                }


            }

        }


        return this;
    }

    @Override
    public INDArray put(INDArray indices, INDArray element) {
        if(indices.rank() > 2) {
            throw new ND4JIllegalArgumentException("Indices must be a vector or matrix.");
        }

        if(indices.rows() == rank()) {
            NdIndexIterator ndIndexIterator = new NdIndexIterator(element.shape());
            for(int i = 0; i < indices.columns(); i++) {
                int[] specifiedIndex = indices.getColumn(i).dup().data().asInt();
                putScalar(specifiedIndex,element.getDouble(ndIndexIterator.next()));
            }

        }
        else {
            List<INDArray> arrList = new ArrayList<>();

            if(indices.isMatrix() || indices.isColumnVector()) {
                for(int i = 0; i < indices.rows(); i++) {
                    INDArray row = indices.getRow(i);
                    for(int j = 0; j < row.length(); j++) {
                        INDArray slice = slice(row.getInt(j));
                        Nd4j.getExecutioner().exec(new Assign(new INDArray[]{slice,element},new INDArray[]{slice}));
                        arrList.add(slice(row.getInt(j)));
                    }


                }
            }
            else if(indices.isRowVector()) {
                for(int i = 0; i < indices.length(); i++) {
                    arrList.add(slice(indices.getInt(i)));
                }
            }

        }


        return this;

    }


    @Override
    public INDArray put(INDArrayIndex[] indices, INDArray element) {
        Nd4j.getCompressor().autoDecompress(this);
        if (indices[0] instanceof SpecifiedIndex && element.isVector()) {
            indices[0].reset();
            int cnt = 0;
            while (indices[0].hasNext()) {
                long idx = indices[0].next();
                // FIXME: LONG
                putScalar((int) idx, element.getDouble(cnt));
                cnt++;
            }
            return this;
        } else {
            return get(indices).assign(element);
        }
    }

    @Override
    public INDArray put(INDArrayIndex[] indices, Number element) {
        Nd4j.getCompressor().autoDecompress(this);
        INDArray get = get(indices);
        for (int i = 0; i < get.length(); i++)
            get.putScalar(i, element.doubleValue());
        return this;
    }


    /**
     * Mainly here for people coming from numpy.
     * This is equivalent to a call to permute
     *
     * @param dimension the dimension to swap
     * @param with      the one to swap it with
     * @return the swapped axes view
     */
    @Override
    public INDArray swapAxes(int dimension, int with) {
        int[] shape = ArrayUtil.range(0, shape().length);
        shape[dimension] = with;
        shape[with] = dimension;
        return permute(shape);
    }


    @Override
    public boolean isView() {
        /*
            We don't really use Shape offset value anywhere
            And it's possible to be not a view, and have non-empty originalBuffer
         */
        // length/data.length can be different in case of Threshold conversion
        return Shape.offset(jvmShapeInfo.javaShapeInformation) > 0
                || (length() < data().length() && data.dataType() != DataBuffer.Type.INT)
                || data().originalDataBuffer() != null;
    }

    @Override
    public boolean isSparse() {
        return false;
    }

    @Override
    public DataBuffer data() {
        return data;
    }

    @Override
    public void setData(DataBuffer data) {
        this.data = data;
    }

    /**
     * Number of slices: aka shape[0]
     *
     * @return the number of slices
     * for this nd array
     */
    @Override
    public long slices() {
        if (isRowVector())
            return length();

        return size(0);
    }

    @Override
    public INDArray subArray(ShapeOffsetResolution resolution) {
        Nd4j.getCompressor().autoDecompress(this);
        long[] offsets = resolution.getOffsets();
        int[] shape = LongUtils.toInts(resolution.getShapes());
        int[] stride = LongUtils.toInts(resolution.getStrides());

        //        if (offset() + resolution.getOffset() >= Integer.MAX_VALUE)
        //            throw new IllegalArgumentException("Offset of array can not be >= Integer.MAX_VALUE");

        long offset = (offset() + resolution.getOffset());


        int n = shape.length;

        // FIXME: shapeInfo should be used here
        if (shape.length < 1)
            return create(Nd4j.createBufferDetached(shape));
        if (offsets.length != n)
            throw new IllegalArgumentException("Invalid offset " + Arrays.toString(offsets));
        if (stride.length != n)
            throw new IllegalArgumentException("Invalid stride " + Arrays.toString(stride));

        if (shape.length == rank() && Shape.contentEquals(shape, shapeOf())) {
            if (ArrayUtil.isZero(offsets)) {
                return this;
            } else {
                throw new IllegalArgumentException("Invalid subArray offsets");
            }
        }

        char newOrder = Shape.getOrder(shape, stride, 1);

        return create(data, Arrays.copyOf(shape, shape.length), stride, offset, newOrder);
    }

    @Override
    public INDArray subArray(long[] offsets, int[] shape, int[] stride) {
        Nd4j.getCompressor().autoDecompress(this);
        int n = shape.length;

        // FIXME: shapeInfo should be used here
        if (shape.length < 1)
            return create(Nd4j.createBufferDetached(shape));
        if (offsets.length != n)
            throw new IllegalArgumentException("Invalid offset " + Arrays.toString(offsets));
        if (stride.length != n)
            throw new IllegalArgumentException("Invalid stride " + Arrays.toString(stride));

        if (Shape.contentEquals(shape, shapeOf())) {
            if (ArrayUtil.isZero(offsets)) {
                return this;
            } else {
                throw new IllegalArgumentException("Invalid subArray offsets");
            }
        }

        long[] dotProductOffsets = offsets;
        int[] dotProductStride = stride;

        long offset = Shape.offset(jvmShapeInfo.javaShapeInformation) + NDArrayIndex.offset(dotProductStride, dotProductOffsets);
        if (offset >= data().length())
            offset = ArrayUtil.sumLong(offsets);

        return create(data, Arrays.copyOf(shape, shape.length), stride, offset, ordering());
    }

    protected INDArray create(DataBuffer buffer) {
        return Nd4j.create(buffer);
    }

    @Override
    public INDArray cond(Condition condition) {
        return dup().condi(condition);
    }

    @Override
    public INDArray condi(Condition condition) {
        Nd4j.getCompressor().autoDecompress(this);
        INDArray linear = this;
        for (int i = 0; i < length(); i++) {
            boolean met = condition.apply(linear.getDouble(i));
            linear.putScalar(i, met ? 1 : 0);
        }
        return this;
    }


    protected void init(int[] shape, int[] stride) {

        //default row vector
        if (shape.length == 1) {
            init(new int[] {1, shape[0]}, new int[] {1, stride[0]});
        }

        //null character
        if (ordering() == '\u0000') {
            //Shape.setOrder(shapeInfo(), Nd4j.order());
            val si = Nd4j.getShapeInfoProvider().createShapeInformation(ArrayUtil.toLongArray(shape), ArrayUtil.toLongArray(stride), 1, Nd4j.order(), this.dataType());
            setShapeInformation(si);
        }

    }

    protected void init(long[] shape, long[] stride) {

        //default row vector
        if (shape.length == 1) {
            init(new long[] {1, shape[0]}, new long[] {1, stride[0]});
        }

        //null character
        if (ordering() == '\u0000') {
            val si = Nd4j.getShapeInfoProvider().createShapeInformation(shape,stride, 1, Nd4j.order(), this.dataType());
            setShapeInformation(si);
        }

    }


    @Override
    public INDArray getScalar(long i) {
        if (i > this.length())
            throw new ND4JIllegalStateException("Index can't be greater then array length");

        if (i < 0)
            i += this.length();

        long idx = this.isScalar() ? 0 : Shape.getOffset(jvmShapeInfo.javaShapeInformation, Shape.ind2subC(this.shape(), i));
        val buffer = Nd4j.createBuffer( this.data(), this.data().originalOffset() + idx, 1);
        val shape = Nd4j.getShapeInfoProvider().createShapeInformation(new long[0], new long[0],1,'c', dataType());
        return Nd4j.createArrayFromShapeBuffer(buffer, shape);
    }

    /**
     * Do a row wise op (a,s,m,d)
     * a : add
     * s : subtract
     * m : multiply
     * d : divide
     * h : reverse subtraction
     * t : reverse division
     *
     * @param columnVector the column  vector
     * @param operation    the operation
     * @return
     */
    protected INDArray doColumnWise(INDArray columnVector, char operation) {
        Nd4j.getCompressor().autoDecompress(this);
       if(columnVector.isScalar()) {
           switch (operation) {
               case 'a':
                   addi(columnVector.getDouble(0));
                   break;
               case 'p':
                   assign(columnVector.getDouble(0));
                   break;
               case 's':
                   subi(columnVector.getDouble(0));
                   break;
               case 'm':
                   muli(columnVector.getDouble(0));
                   break;
               case 'd':
                   divi(columnVector.getDouble(0));
                   break;
               case 'h':
                   rsubi(columnVector.getDouble(0));
                   break;
               case 't':
                   rdivi(columnVector.getDouble(0));
                   break;

           }

           return this;
       }

       else if(isScalar()) {
           switch (operation) {
               case 'a':
                   return columnVector.addi(getDouble(0));
               case 'p':
                   return columnVector.assign(getDouble(0));
               case 's':
                   return columnVector.subi(getDouble(0));
               case 'm':
                   return columnVector.muli(getDouble(0));
               case 'd':
                   return columnVector.divi(getDouble(0));
               case 'h':
                   return columnVector.rsubi(getDouble(0));
               case 't':
                   return columnVector.rdivi(getDouble(0));

           }
       }

        //Input validation: require (a) columnVector to actually be a column vector, and (b) this.size(0) to match columnVector.size(0)
        //Or, simply require it to be a rank 1 vector
        if ((!columnVector.isColumnVector() && columnVector.rank() > 1) || this.size(0) != columnVector.size(0) || columnVector.length() <= 1) {
            throw new IllegalStateException("Mismatched shapes (shape = " + Arrays.toString(shape())
                    + ", column vector shape =" + Arrays.toString(columnVector.shape()) + ")");
        }

        if (columnVector.data().sameUnderlyingData(data()))
            return doColumnWise(columnVector.dup(), operation);
        if (isVector()) {
            switch (operation) {
                case 'a':
                    addi(columnVector);
                    break;
                case 'p':
                    assign(columnVector);
                    break;
                case 's':
                    subi(columnVector);
                    break;
                case 'm':
                    muli(columnVector);
                    break;
                case 'd':
                    divi(columnVector);
                    break;
                case 'h':
                    rsubi(columnVector);
                    break;
                case 't':
                    rdivi(columnVector);
                    break;
            }

            return this;
        }
        if (rows() == 1 && columnVector.isScalar()) {
            applyScalarOp(columnVector, operation);
        } else {
            // special optimization case, broadcast turns into ScalarOp Along Dimension
            if (rank() == 2 && elementWiseStride() == 1 && ordering() == 'c' && columnVector.elementWiseStride() == 1) {
                switch (operation) {
                    case 'a': {
                        ScalarAdd op = new ScalarAdd(this, columnVector, this, this.length(), 0.0);
                        op.setDimension(1);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 'p': {
                        ScalarSet op = new ScalarSet(this, columnVector, this, this.length(), 0.0);
                        op.setDimension(1);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 's': {
                        ScalarSubtraction op = new ScalarSubtraction(this, columnVector, this, this.length(), 0.0);
                        op.setDimension(1);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 'm': {
                        ScalarMultiplication op =
                                new ScalarMultiplication(this, columnVector, this, this.length(), 0.0);
                        op.setDimension(1);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 'd': {
                        ScalarDivision op = new ScalarDivision(this, columnVector, this, this.length(), 0.0);
                        op.setDimension(1);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 'h': {
                        ScalarReverseSubtraction op =
                                new ScalarReverseSubtraction(this, columnVector, this, this.length(), 0.0);
                        op.setDimension(1);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 't': {
                        ScalarReverseDivision op =
                                new ScalarReverseDivision(this, columnVector, this, this.length(), 0.0);
                        op.setDimension(1);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }

                }
            } else {
                applyBroadcastOp(columnVector, operation);
            }

        }

        return this;

    }

    @Override
    @Deprecated
    public boolean isCleanedUp() {
        return false;
    }

    @Override
    @Deprecated
    public void cleanup() {
        if (Nd4j.shouldInstrument)
            Nd4j.getInstrumentation().log(this, Instrumentation.DESTROYED);
    }



    /**
     * Do a row wise op (a,s,m,d)
     * a : add
     * s : subtract
     * m : multiply
     * d : divide
     * h : reverse subtraction
     * t : reverse division
     *
     * @param rowVector the row vector
     * @param operation the operation
     * @return
     */
    protected INDArray doRowWise(INDArray rowVector, final char operation) {
        Nd4j.getCompressor().autoDecompress(this);


        if(rowVector.isScalar()) {
            switch (operation) {
                case 'a':
                    addi(rowVector.getDouble(0));
                    break;
                case 'p':
                    assign(rowVector.getDouble(0));
                    break;
                case 's':
                    subi(rowVector.getDouble(0));
                    break;
                case 'm':
                    muli(rowVector.getDouble(0));
                    break;
                case 'd':
                    divi(rowVector.getDouble(0));
                    break;
                case 'h':
                    rsubi(rowVector.getDouble(0));
                    break;
                case 't':
                    rdivi(rowVector.getDouble(0));
                    break;

            }

            return this;
        }
        else if(isScalar()) {
            switch (operation) {
                case 'a':
                    return rowVector.addi(getDouble(0));
                case 'p':
                    return rowVector.assign(getDouble(0));
                case 's':
                    return rowVector.subi(getDouble(0));
                case 'm':
                    return rowVector.muli(getDouble(0));
                case 'd':
                    return rowVector.divi(getDouble(0));
                case 'h':
                    return rowVector.rsubi(getDouble(0));
                case 't':
                    return rowVector.rdivi(getDouble(0));

            }
        }

        //Input validation: require (a) rowVector to actually be a row vector, and (b) this.size(1) to match rowVector.size(1)
        if (!rowVector.isRowVector() || this.rank() > 1 && rowVector.rank() > 1 &&  this.size(1) != rowVector.size(1) || rowVector.length() <= 1) {
            throw new IllegalStateException("Mismatched shapes (shape = " + Arrays.toString(shape())
                    + ", row vector shape =" + Arrays.toString(rowVector.shape()) + ")");
        }

        if (rowVector.data().sameUnderlyingData(data()))
            return doRowWise(rowVector.dup(), operation);

        if (isVector()) {
            switch (operation) {
                case 'a':
                    addi(rowVector);
                    break;
                case 'p':
                    assign(rowVector);
                    break;
                case 's':
                    subi(rowVector);
                    break;
                case 'm':
                    muli(rowVector);
                    break;
                case 'd':
                    divi(rowVector);
                    break;
                case 'h':
                    rsubi(rowVector);
                    break;
                case 't':
                    rdivi(rowVector);
                    break;
            }

            return this;
        }

        if (rank() == 2 && columns() == 1 && rowVector.isScalar()) {
            applyScalarOp(rowVector, operation);
        } else {
            // special optimization case, broadcast turns into ScalarOp Along Dimension
            if (rank() == 2 && elementWiseStride() == 1 && ordering() == 'f' && rowVector.elementWiseStride() == 1) {
                switch (operation) {
                    case 'a': {
                        ScalarAdd op = new ScalarAdd(this, rowVector, this, this.length(), 0.0);
                        op.setDimension(0);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 'p': {
                        ScalarSet op = new ScalarSet(this, rowVector, this, this.length(), 0.0);
                        op.setDimension(0);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 's': {
                        ScalarSubtraction op = new ScalarSubtraction(this, rowVector, this, this.length(), 0.0);
                        op.setDimension(0);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 'm': {
                        ScalarMultiplication op = new ScalarMultiplication(this, rowVector, this, this.length(), 0.0);
                        op.setDimension(0);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 'd': {
                        ScalarDivision op = new ScalarDivision(this, rowVector, this, this.length(), 0.0);
                        op.setDimension(0);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 'h': {
                        ScalarReverseSubtraction op =
                                new ScalarReverseSubtraction(this, rowVector, this, this.length(), 0.0);
                        op.setDimension(0);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }
                    case 't': {
                        ScalarReverseDivision op = new ScalarReverseDivision(this, rowVector, this, this.length(), 0.0);
                        op.setDimension(0);
                        Nd4j.getExecutioner().exec(op);
                        break;
                    }

                }
            } else {
                applyBroadcastOp(rowVector, operation);
            }
        }

        return this;
    }


    private void applyBroadcastOp(INDArray vector, final char operation) {
        Nd4j.getCompressor().autoDecompress(this);
        int alongDimension = Shape.isRowVectorShape(vector.shape()) ? 1 : 0;

        // FIXME: probably this is wrong, because strict equality is always false in current DataBuffer mechanics
        if (this.data() == vector.data())
            vector = vector.dup();
        switch (operation) {
            case 'a':
                Nd4j.getExecutioner().exec(new BroadcastAddOp(this, vector, this, alongDimension), alongDimension);
                return;
            case 's':
                Nd4j.getExecutioner().exec(new BroadcastSubOp(this, vector, this, alongDimension), alongDimension);
                return;
            case 'm':
                Nd4j.getExecutioner().exec(new BroadcastMulOp(this, vector, this, alongDimension), alongDimension);
                return;
            case 'd':
                Nd4j.getExecutioner().exec(new BroadcastDivOp(this, vector, this, alongDimension), alongDimension);
                return;
            case 'h':
                Nd4j.getExecutioner().exec(new BroadcastRSubOp(this, vector, this, alongDimension), alongDimension);
                return;
            case 't':
                Nd4j.getExecutioner().exec(new BroadcastRDivOp(this, vector, this, alongDimension), alongDimension);
                return;
            case 'p':
                Nd4j.getExecutioner().exec(new BroadcastCopyOp(this, vector, this, alongDimension), alongDimension);
                return;
            default:
                throw new UnsupportedOperationException("Unknown operation: " + operation);
        }
    }

    private void applyScalarOp(INDArray vector, char operation) {
        Nd4j.getCompressor().autoDecompress(this);
        switch (operation) {
            case 'a':
                addi(vector.getDouble(0));
                break;
            case 's':
                subi(vector.getDouble(0));
                break;
            case 'm':
                muli(vector.getDouble(0));
                break;
            case 'd':
                divi(vector.getDouble(0));
                break;
            case 'h':
                rsubi(vector.getDouble(0));
                break;
            case 't':
                rdivi(vector.getDouble(0));
                break;
        }
    }

    protected DataBuffer shapeOf() {
        //        if (shape == null)
        //            shape = Shape.shapeOf(shapeInfoDataBuffer());
        //        return shape;

        return Shape.shapeOf(shapeInfoDataBuffer());
    }

    protected DataBuffer strideOf() {
        //        if (stride == null)
        //            stride = Shape.stride(shapeInfoDataBuffer());
        //        return stride;
        return Shape.stride(shapeInfoDataBuffer());
    }

    @Override
    public int stride(int dimension) {
        int rank = jvmShapeInfo.rank;
        if (dimension < 0)
            return (int) stride()[dimension + rank];
        return (int) stride()[dimension];
    }

    @Override
    public INDArray rdiviColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 't');
    }

    @Override
    public INDArray rdivColumnVector(INDArray columnVector) {
        return dup().rdiviColumnVector(columnVector);
    }

    @Override
    public INDArray rdiviRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 't');
    }

    @Override
    public INDArray rdivRowVector(INDArray rowVector) {
        return dup().rdiviRowVector(rowVector);
    }

    @Override
    public INDArray rsubiColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 'h');
    }

    @Override
    public INDArray rsubColumnVector(INDArray columnVector) {
        return dup().rsubiColumnVector(columnVector);
    }

    @Override
    public INDArray rsubiRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 'h');
    }

    @Override
    public INDArray rsubRowVector(INDArray rowVector) {
        return dup().rsubiRowVector(rowVector);
    }

    /**
     * Inserts the element at the specified index
     *
     * @param i       the index insert into
     * @param element a scalar ndarray
     * @return a scalar ndarray of the element at this index
     */
    @Override
    public INDArray put(int i, INDArray element) {
        if (!element.isScalar())
            throw new IllegalArgumentException("Element must be a scalar");
        return putScalar(i, element.getDouble(0));
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray diviColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 'd');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray divColumnVector(INDArray columnVector) {
        return dup().diviColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray diviRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 'd');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray divRowVector(INDArray rowVector) {
        return dup().diviRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray muliColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 'm');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray mulColumnVector(INDArray columnVector) {
        return dup().muliColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray muliRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 'm');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray mulRowVector(INDArray rowVector) {
        return dup().muliRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subiColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 's');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subColumnVector(INDArray columnVector) {
        return dup().subiColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subiRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 's');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray subRowVector(INDArray rowVector) {
        return dup().subiRowVector(rowVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addiColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 'a');
    }

    @Override
    public INDArray putiColumnVector(INDArray columnVector) {
        return doColumnWise(columnVector, 'p');
    }

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addColumnVector(INDArray columnVector) {
        return dup().addiColumnVector(columnVector);
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addiRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 'a');
    }

    @Override
    public INDArray putiRowVector(INDArray rowVector) {
        return doRowWise(rowVector, 'p');
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    @Override
    public INDArray addRowVector(INDArray rowVector) {
        return dup().addiRowVector(rowVector);
    }


    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmul(INDArray other, INDArray result,MMulTranspose mMulTranspose) {
        MMulTranspose mMulTranspose1 = MMulTranspose.builder()
                .a(this)
                .b(other)
                .transposeA(mMulTranspose.isTransposeA())
                .transposeB(mMulTranspose.isTransposeB())
                .transposeResult(mMulTranspose.isTransposeResult())
                .build();
        return mMulTranspose1.getA().mmul(mMulTranspose1.getB(),result);
    }

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmul(INDArray other, MMulTranspose mMulTranspose) {
        MMulTranspose mMulTranspose1 = MMulTranspose.builder()
                .a(this)
                .b(other)
                .transposeA(mMulTranspose.isTransposeA())
                .transposeB(mMulTranspose.isTransposeB())
                .transposeResult(mMulTranspose.isTransposeResult())
                .build();
        System.out.println(mMulTranspose1.getA());
        System.out.println(mMulTranspose1.getB());
        return mMulTranspose1.getA().mmul(mMulTranspose1.getB());
    }

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmul(INDArray other) {
        // FIXME: for 1D case, we probably want vector output here?
        long[] shape = {rows(), other.rank() == 1 ? 1 : other.columns()};
        INDArray result = createUninitialized(shape, 'f');
        if (result.isScalar())
            return Nd4j.scalar(Nd4j.getBlasWrapper().dot(this, other));
        return mmuli(other, result);
    }

    protected INDArray create(int[] shape, char ordering) {
        return Nd4j.create(shape, ordering);
    }

    @Override
    public double[][] toDoubleMatrix() {
        if(!isMatrix()) {
            throw new ND4JIllegalStateException("Unable to create a 2d array from a non matrix!");
        }

        if (this.rows() > Integer.MAX_VALUE || this.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        double[][] ret = new double[(int) rows()][(int) columns()];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = getRow(i).dup().data().asDouble();
        }

        return ret;
    }

    @Override
    public double[] toDoubleVector() {
        if(!isVectorOrScalar()) {
            throw new ND4JIllegalStateException("Unable to create a 1d array from a non vector!");
        }


        return dup().data().asDouble();
    }

    @Override
    public float[] toFloatVector() {
        if(!isVectorOrScalar()) {
            throw new ND4JIllegalStateException("Unable to create a 1d array from a non vector!");
        }

        return dup().data().asFloat();
    }

    @Override
    public float[][] toFloatMatrix() {
        if(!isMatrix()) {
            throw new ND4JIllegalStateException("Unable to create a 2d array from a non matrix!");
        }

        if (this.rows() > Integer.MAX_VALUE || this.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        float[][] ret = new float[(int) rows()][ (int) columns()];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = getRow(i).dup().data().asFloat();
        }

        return ret;
    }

    @Override
    public int[] toIntVector() {
        if(!isVectorOrScalar()) {
            throw new ND4JIllegalStateException("Unable to create a 1d array from a non vector!");
        }
        return dup().data().asInt();
    }

    @Override
    public long[] toLongVector() {
        if(!isVectorOrScalar()) {
            throw new ND4JIllegalStateException("Unable to create a 1d array from a non vector!");
        }
        return dup().data().asLong();
    }

    @Override
    public long[][] toLongMatrix() {
        if(!isMatrix()) {
            throw new ND4JIllegalStateException("Unable to create a 2d array from a non matrix!");
        }

        if (this.rows() > Integer.MAX_VALUE || this.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        long[][] ret = new long[(int) rows()][(int) columns()];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = getRow(i).dup().data().asLong();
        }

        return ret;
    }

    @Override
    public int[][] toIntMatrix() {
        if(!isMatrix()) {
            throw new ND4JIllegalStateException("Unable to create a 2d array from a non matrix!");
        }

        if (this.rows() > Integer.MAX_VALUE || this.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        int[][] ret = new int[(int) rows()][(int) columns()];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = getRow(i).dup().data().asInt();
        }

        return ret;
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmul(INDArray other, INDArray result) {
        return mmuli(other, result);
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public INDArray div(INDArray other) {
        return divi(other, Nd4j.createUninitialized(this.shape(), this.ordering()));
    }

    /**
     * copy (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    @Override
    public INDArray div(INDArray other, INDArray result) {
        return divi(other, result);
    }

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the addition
     */
    @Override
    public INDArray mul(INDArray other) {
        return muli(other, Nd4j.createUninitialized(this.shape(), this.ordering()));
    }

    /**
     * copy (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    @Override
    public INDArray mul(INDArray other, INDArray result) {
        return muli(other, result);
    }

    /**
     * copy subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public INDArray sub(INDArray other) {
        return subi(other, Nd4j.createUninitialized(this.shape(), this.ordering()));
    }

    /**
     * copy subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    @Override
    public INDArray sub(INDArray other, INDArray result) {
        return subi(other, result);
    }

    /**
     * copy addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public INDArray add(INDArray other) {
        return addi(other, Nd4j.createUninitialized(this.shape(), this.ordering()));
    }

    /**
     * copy addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    @Override
    public INDArray add(INDArray other, INDArray result) {
        return dup().addi(other, result);
    }


    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @param transpose the transpose status of each ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmuli(INDArray other, MMulTranspose transpose) {
        return dup().mmuli(other, this,transpose);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmuli(INDArray other) {
        return dup().mmuli(other, this);
    }


    /**
     * Perform an in place matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmuli(INDArray other, INDArray result, MMulTranspose transpose) {
        MMulTranspose mMulTranspose = MMulTranspose.builder()
                .a(this)
                .b(other)
                .transposeA(transpose.isTransposeA())
                .transposeB(transpose.isTransposeB())
                .transposeResult(transpose.isTransposeResult())
                .build();
        return mMulTranspose.getA().mmuli(mMulTranspose.getB(),result);
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmuli(INDArray other, INDArray result) {
        LinAlgExceptions.assertMultiplies(this, other);


        if (other.isScalar()) {
            return muli(other.getDouble(0), result);
        }
        if (isScalar()) {
            return other.muli(getDouble(0), result);
        }

        /* check sizes and resize if necessary */


        if (result == this || result == other) {
            /* actually, blas cannot do multiplications in-place. Therefore, we will fake by
             * allocating a temporary object on the side and copy the result later.
             */
            INDArray temp = Nd4j.create(result.shape(), Nd4j.getStrides(result.shape(), 'f'));

            if (other.columns() == 1 || other.rank() == 1) {
                Nd4j.getBlasWrapper().level2().gemv(BlasBufferUtil.getCharForTranspose(result),
                        BlasBufferUtil.getCharForTranspose(this), 1.0, this, other, 0.0, temp);
            }

            else {
                Nd4j.getBlasWrapper().level3().gemm(BlasBufferUtil.getCharForTranspose(result),
                        BlasBufferUtil.getCharForTranspose(this), BlasBufferUtil.getCharForTranspose(temp), 1.0,
                        this, other, 0.0, temp);
            }

            result.assign(temp);


        } else {

            //We require that the result array is 'f' (fortran) order
            // However, user might have called mmuli with a c order array for the result
            // In which case, we need to allocate a temporary f order array, and later do an assign to the real result array

            boolean requiresTemp = result.ordering() == 'c';
            INDArray gemmResultArr;
            if (requiresTemp) {
                //Can use createUninitialized due to beta==0.0 parameter in gemm
                gemmResultArr = Nd4j.createUninitialized(result.shape(), 'f');
            } else {
                gemmResultArr = result;
            }

            if (other.columns() == 1 || other.rank() == 1) {
                Nd4j.getBlasWrapper().level2().gemv(
                        ordering(),
                        BlasBufferUtil.getCharForTranspose(other),
                        1.0,
                        this,
                        other,
                        0.0,
                        gemmResultArr);
            } else {
                //gemm doesn't support strides so vectors and views
                //don't work
                Nd4j.getBlasWrapper().level3().gemm(ordering(),
                        BlasBufferUtil.getCharForTranspose(other),
                        BlasBufferUtil.getCharForTranspose(gemmResultArr),
                        1.0,
                        this,
                        other,
                        0.0,
                        gemmResultArr);
            }

            if (requiresTemp) {
                result.assign(gemmResultArr);
            }
        }

        // 1D edge case: reshape back to vector
        if (other.rank() == 1)
            result = result.reshape(result.length());

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);
        return result;
    }

    private INDArray create(int[] shape, int[] stride) {
        return Nd4j.create(shape, stride);
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other the second ndarray to divide
     * @return the result of the divide
     */
    @Override
    public INDArray divi(INDArray other) {
        return divi(other, this);
    }

    /**
     * in place (element wise) division of two matrices
     *
     * @param other  the second ndarray to divide
     * @param result the result ndarray
     * @return the result of the divide
     */
    @Override
    public INDArray divi(INDArray other, INDArray result) {
        if (other.isScalar()) {
            return divi(other.getDouble(0), result);
        }

        if (isScalar()) {
            return other.rdivi(getDouble(0), result);
        }


        if(!Shape.shapeEquals(this.shape(),other.shape())) {
            int[] broadcastDimensions = Shape.getBroadcastDimensions(this.shape(),other.shape());
            Nd4j.getExecutioner().exec(new BroadcastDivOp(this,other,result,broadcastDimensions),broadcastDimensions);
            return result;
        }


        LinAlgExceptions.assertSameShape(other, result);
        Nd4j.getExecutioner().exec(new OldDivOp(this, other, result, length()));

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);
        return result;
    }

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other the second ndarray to multiply
     * @return the result of the multiplication
     */
    @Override
    public INDArray muli(INDArray other) {
        return muli(other, this);
    }

    /**
     * in place (element wise) multiplication of two matrices
     *
     * @param other  the second ndarray to multiply
     * @param result the result ndarray
     * @return the result of the multiplication
     */
    @Override
    public INDArray muli(INDArray other, INDArray result) {
        if (other.isScalar()) {
            return muli(other.getDouble(0), result);
        }
        if (isScalar()) {
            return other.muli(getDouble(0), result);
        }



        if(!Shape.shapeEquals(this.shape(),other.shape())) {
            int[] broadcastDimensions = Shape.getBroadcastDimensions(this.shape(),other.shape());
            Nd4j.getExecutioner().exec(new BroadcastMulOp(this,other,result,broadcastDimensions),broadcastDimensions);
            return result;
        }

        LinAlgExceptions.assertSameShape(other, result);

        Nd4j.getExecutioner().exec(new OldMulOp(this, other, result, length()));

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    /**
     * in place subtraction of two matrices
     *
     * @param other the second ndarray to subtract
     * @return the result of the addition
     */
    @Override
    public INDArray subi(INDArray other) {
        return subi(other, this);
    }

    /**
     * in place subtraction of two matrices
     *
     * @param other  the second ndarray to subtract
     * @param result the result ndarray
     * @return the result of the subtraction
     */
    @Override
    public INDArray subi(INDArray other, INDArray result) {
        if (other.isScalar()) {
            return subi(other.getDouble(0), result);
        }
        if (isScalar()) {
            return other.rsubi(getDouble(0), result);
        }


        if(!Shape.shapeEquals(this.shape(),other.shape())) {
            int[] broadcastDimensions = Shape.getBroadcastDimensions(this.shape(),other.shape());
            Nd4j.getExecutioner().exec(new BroadcastSubOp(this,other,result,broadcastDimensions),broadcastDimensions);
            return result;
        }


        LinAlgExceptions.assertSameShape(other, result);


        Nd4j.getExecutioner().exec(new OldSubOp(this, other,result));

        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    /**
     * in place addition of two matrices
     *
     * @param other the second ndarray to add
     * @return the result of the addition
     */
    @Override
    public INDArray addi(INDArray other) {
        return addi(other, this);
    }

    /**
     * in place addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    @Override
    public INDArray addi(INDArray other, INDArray result) {
        if (other.isScalar()) {
            return this.addi(other.getDouble(0), result);
        }

        if (isScalar()) {
            return other.addi(getDouble(0), result);
        }

        if(!Shape.shapeEquals(this.shape(),other.shape())) {
            int[] broadcastDimensions = Shape.getBroadcastDimensions(this.shape(),other.shape());
            result = Nd4j.createUninitialized(Shape.broadcastOutputShape(this.shape(),other.shape()));
            Nd4j.getExecutioner().exec(new BroadcastAddOp(this,other,result,broadcastDimensions),broadcastDimensions);
            return result;
        }

        LinAlgExceptions.assertSameShape(other, result);

        Nd4j.getExecutioner().exec(new OldAddOp(this, other, result, length()));


        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    /**
     * Returns the normmax along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public INDArray normmax(int... dimension) {
        return Nd4j.getExecutioner().exec(new NormMax(this), dimension);
    }

    /**
     * Reverse division
     *
     * @param other the matrix to divide from
     * @return
     */
    @Override
    public INDArray rdiv(INDArray other) {
        return dup().rdivi(other);
    }

    /**
     * Reverse divsion (in place)
     *
     * @param other
     * @return
     */
    @Override
    public INDArray rdivi(INDArray other) {
        return rdivi(other, this);
    }

    /**
     * Reverse division
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    @Override
    public INDArray rdiv(INDArray other, INDArray result) {
        return dup().rdivi(other, result);
    }

    /**
     * Reverse division (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    @Override
    public INDArray rdivi(INDArray other, INDArray result) {
        return other.divi(this, result);
    }

    /**
     * Reverse subtraction
     *
     * @param other  the matrix to subtract from
     * @param result the result ndarray
     * @return
     */
    @Override
    public INDArray rsub(INDArray other, INDArray result) {
        return dup().rsubi(other, result);
    }

    /**
     * @param other
     * @return
     */
    @Override
    public INDArray rsub(INDArray other) {
        return dup().rsubi(other);
    }

    /**
     * @param other
     * @return
     */
    @Override
    public INDArray rsubi(INDArray other) {
        return rsubi(other, this);
    }

    /**
     * Reverse subtraction (in-place)
     *
     * @param other  the other ndarray to subtract
     * @param result the result ndarray
     * @return the ndarray with the operation applied
     */
    @Override
    public INDArray rsubi(INDArray other, INDArray result) {
        return other.subi(this, result);
    }

    /**
     * Set the value of the ndarray to the specified value
     *
     * @param value the value to assign
     * @return the ndarray with the values
     */
    @Override
    public INDArray assign(Number value) {
        Nd4j.getExecutioner().exec(new ScalarSet(this, value));
        return this;
    }


    /**
     * Assign all elements from given ndarray that are matching given condition,
     * ndarray to this ndarray
     *
     * @param arr       the elements to assign
     * @param condition
     * @return this
     */
    @Override
    public INDArray assignIf(INDArray arr, Condition condition) {
        BooleanIndexing.assignIf(this, arr, condition);
        return this;
    }

    /**
     * Replaces all elements in this ndarray that are matching give condition, with corresponding elements from given array
     *
     * @param arr
     * @param condition
     * @return
     */
    @Override
    public INDArray replaceWhere(INDArray arr, Condition condition) {
        Nd4j.getCompressor().autoDecompress(this);
        BooleanIndexing.replaceWhere(this, arr, condition);
        return this;
    }

    @Override
    @Deprecated
    public long linearIndex(long i) {
        long idx = i;
        for (int j = 0; j < jvmShapeInfo.rank - 1; j++) {
            if (size((int) i) == 1)
                continue;
            idx += i * stride(j);
        }
        return Shape.offset(jvmShapeInfo.javaShapeInformation) + (idx);
    }



    /**
     * Returns the specified slice of this matrix.
     * In matlab, this would be equivalent to (given a 2 x 2 x 2):
     * A(:,:,x) where x is the slice you want to return.
     * <p/>
     * The slice is always relative to the final dimension of the matrix.
     *
     * @param slice the slice to return
     * @return the specified slice of this matrix
     */
    @Override
    public INDArray slice(long slice) {
        Nd4j.getCompressor().autoDecompress(this);


        long slices = slices();
        if (slice >= slices)
            throw new IllegalArgumentException("Illegal slice " + slice);

        if (jvmShapeInfo.rank == 0 || isVector()) {
            if (slice == 0 || isVector()) {
                return createScalarForIndex(slice, true);
            }
            else {
                throw new IllegalArgumentException("Can't slice a 0-d NDArray");
            }

        }


        if (slice < 0)
            slice += rank();
        INDArrayIndex[] indexes = new INDArrayIndex[rank()];
        indexes[0] = NDArrayIndex.point(slice);
        for (int i = 1; i < rank(); i++) {
            indexes[i] = NDArrayIndex.all();
        }
        return get(indexes);
    }



    protected INDArray createScalarForIndex(long i, boolean applyOffset) {
        if(isVector())
            return getScalar(i);
        return Nd4j.create(data(), new long[] {1, 1}, new long[] {1, 1}, i);
    }

    protected INDArray createScalar(double d) {
        return Nd4j.scalar(d);
    }



    @Override
    public int getTrailingOnes() {
        int numLeadingOnes = 0;
        for (int i = rank() - 1; i > 0; i--) {
            if (size(i) == 1)
                numLeadingOnes++;
        }

        return numLeadingOnes;
    }



    @Override
    public int getLeadingOnes() {
        int numLeadingOnes = 0;
        for (int i = 0; i < rank(); i++) {
            if (size(i) == 1)
                numLeadingOnes++;
        }

        return numLeadingOnes;
    }



    /**
     * Returns the slice of this from the specified dimension
     *
     * @param slice     the dimension to return from
     * @param dimension the dimension of the slice to return
     * @return the slice of this matrix from the specified dimension
     * and dimension
     */
    @Override
    public INDArray slice(long slice, int dimension) {
        Nd4j.getCompressor().autoDecompress(this);

        long slices = size(dimension);
        if (slice >= slices)
            throw new IllegalArgumentException("Illegal slice " + slice);

        if (jvmShapeInfo.rank == 0) {
            if (slice == 0)
                return createScalarForIndex(slice, true);
            else
                throw new IllegalArgumentException("Can't slice a 0-d NDArray");

        }


        if (slice < 0)
            slice += rank();
        INDArrayIndex[] indexes = new INDArrayIndex[rank()];
        indexes[dimension] = NDArrayIndex.point(slice);
        for (int i = 0; i < rank(); i++) {
            if (i != dimension)
                indexes[i] = NDArrayIndex.all();
        }
        return get(indexes);

    }

    /**
     * Fetch a particular number on a multi dimensional scale.
     *
     * @param indexes the indexes to get a number from
     * @return the number at the specified indices
     */
    @Override
    public INDArray getScalar(int[] indexes) {
        if (indexes.length > rank())
            throw new ND4JIllegalStateException("Indexes can't be longer then array rank");

        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] < 0)
                indexes[i] += this.size(i);
        }
        long idx = Shape.getOffset(jvmShapeInfo.javaShapeInformation, indexes);
        val buffer = Nd4j.createBuffer(this.data(), idx, 1);
        val shape = Nd4j.getShapeInfoProvider().createShapeInformation(new long[0], new long[0],1, 'c', this.dataType());
        return Nd4j.createArrayFromShapeBuffer(buffer, shape);
    }

    @Override
    public INDArray getScalar(long... indexes) {
        if (indexes.length > rank())
            throw new ND4JIllegalStateException("Indexes can't be longer then array rank");

        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] < 0)
                indexes[i] += this.size(i);
        }

        long idx = Shape.getOffset(jvmShapeInfo.javaShapeInformation, indexes);
        val buffer = Nd4j.createBuffer(this.data(), idx, 1);
        val shape = Nd4j.getShapeInfoProvider().createShapeInformation(new long[0], new long[0],1,'c', this.dataType());
        return Nd4j.createArrayFromShapeBuffer(buffer, shape);
    }

    @Override
    public INDArray rdiv(Number n) {
        //return dup().rdivi(n);
        return rdivi(n, Nd4j.createUninitialized(this.shape(), this.ordering()));
    }

    @Override
    public INDArray rdivi(Number n) {
        return rdivi(n, this);
    }

    @Override
    public INDArray rsub(Number n) {
        //return dup().rsubi(n);
        return rsubi(n, Nd4j.createUninitialized(this.shape(), this.ordering()));
    }

    @Override
    public INDArray rsubi(Number n) {
        return rsubi(n, this);
    }

    @Override
    public INDArray div(Number n) {
        //return dup().divi(n);
        return divi(n, Nd4j.createUninitialized(this.shape(), this.ordering()));
    }

    @Override
    public INDArray divi(Number n) {
        return divi(n, this);
    }

    @Override
    public INDArray mul(Number n) {
        // return dup().muli(n);
        return muli(n, Nd4j.createUninitialized(this.shape(), this.ordering()));
    }

    @Override
    public INDArray muli(Number n) {
        return muli(n, this);
    }

    @Override
    public INDArray sub(Number n) {
        //return dup().subi(n);
        return subi(n, Nd4j.createUninitialized(this.shape(), this.ordering()));
    }

    @Override
    public INDArray subi(Number n) {
        return subi(n, this);
    }

    @Override
    public INDArray add(Number n) {
        //return dup().addi(n);
        return addi(n, Nd4j.createUninitialized(this.shape(), this.ordering()));
    }

    @Override
    public INDArray addi(Number n) {
        return addi(n, this);
    }



    /**
     * Replicate and tile array to fill out to the given shape
     * See:
     * https://github.com/numpy/numpy/blob/master/numpy/matlib.py#L310-L358
     * @param shape the new shape of this ndarray
     * @return the shape to fill out to
     */
    @Override
    public INDArray repmat(int[] shape) {
        Nd4j.getCompressor().autoDecompress(this);


        long rows = rows() * shape[0];
        long cols = columns() * shape[1];
        INDArray ret = reshape(1, length()).repeat(0, shape[0]).reshape(rows, columns()).repeat(0, shape[1]);
        return ret.reshape(rows, cols);
    }

    @Override
    public INDArray repeat(int dimension, int... repeats) {
        return repeat(dimension, ArrayUtil.toLongArray(repeats));
    }

    @Override
    public INDArray repeat(int dimension, long... repeats) {
        Nd4j.getCompressor().autoDecompress(this);


        if (dimension < 0)
            dimension += rank();

        if (repeats.length < rank()) {
            if (dimension > 0)
                repeats = Longs.concat(ArrayUtil.nTimes((long) rank() - repeats.length, 1), repeats);
                //append rather than prepend for dimension == 0
            else
                repeats = Longs.concat(repeats, ArrayUtil.nTimes((long) rank() - repeats.length, 1));

        }

        long[] newShape = new long[rank()];

        for (int i = 0; i < newShape.length; i++)
            newShape[i] = size(i) * repeats[i];

        INDArray ret = Nd4j.create(newShape);

        //number of times to repeat each value
        long repeatDelta = ArrayUtil.prod(newShape) / length();
        for (int i = 0; i < tensorssAlongDimension(dimension); i++) {
            INDArray thisTensor = tensorAlongDimension(i, dimension);
            INDArray retTensor = ret.tensorAlongDimension(i, dimension);
            int retIdx = 0;
            for (int k = 0; k < thisTensor.length(); k++) {
                for (int j = 0; j < repeatDelta; j++) {
                    retTensor.putScalar(retIdx++, thisTensor.getDouble(k));
                }
            }
        }

        return ret;
    }


    /**
     * Insert a row in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param row   the row insert into
     * @param toPut the row to insert
     * @return this
     */
    @Override
    public INDArray putRow(long row, INDArray toPut) {
        if (isRowVector() && toPut.isVector()) {
            return assign(toPut);
        }
        return put(new INDArrayIndex[] {NDArrayIndex.point(row), NDArrayIndex.all()}, toPut);
    }

    /**
     * Insert a column in to this array
     * Will throw an exception if this
     * ndarray is not a matrix
     *
     * @param column the column to insert
     * @param toPut  the array to put
     * @return this
     */
    @Override
    public INDArray putColumn(int column, INDArray toPut) {
        Nd4j.getCompressor().autoDecompress(this);

        if (isColumnVector() && toPut.isVector()) {
            return assign(toPut);
        }
        return put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.point(column)}, toPut);

    }


    @Override
    public double getDouble(long i) {
        Nd4j.getCompressor().autoDecompress(this);

        if (i >= length()) {
            throw new IllegalArgumentException("Unable to get linear index " + i + ": values is greater than length (" + length() + ")");
        }

        autoProcessScalarCall();

        if (i == 0)
            return data().getDouble(i);

        long[] dimensions = ordering() == 'c' ? Shape.ind2subC(this, i) : Shape.ind2sub(this, i);
        Shape.assertShapeLessThan(dimensions, shape());
        return getDouble(dimensions);

    }

    @Override
    public double getDouble(long i, long j) {
        return getDouble(new long[] {i, j});
    }

    @Override
    public float getFloat(long i) {
        return (float) getDouble(i);
    }

    @Override
    public float getFloat(long i, long j) {
        return (float) getDouble(i, j);
    }

    /**
     * Return transposed copy of this matrix.
     */
    @Override
    public INDArray transpose() {
        return transposei();
    }


    /**
     *
     * Return transposed version of this matrix.
     *
     * PLEASE NOTE: This method is NOT in place, it will return transposed copy instead.
     */
    @Override
    public INDArray transposei() {
        return permute(ArrayUtil.reverseCopy(ArrayUtil.range(0, rank())));
    }

    protected INDArray create(DataBuffer data, int[] shape, int[] strides) {
        return Nd4j.create(data, shape, strides, 0, ordering());
    }

    @Override
    public INDArray reshape(char order, int... newShape) {
        // FIXME: int cast
        return reshape(order, ArrayUtil.toLongArray(newShape));
    }

    @Override
    public INDArray reshape(char order, long... newShape) {
        Nd4j.getCompressor().autoDecompress(this);

        // special case for empty reshape
        if (this.length() == 1 && (newShape == null || newShape.length == 0)) {
            return Nd4j.create(this.data(), new int[0], new int[0], 0);
        }

        if (newShape == null || newShape.length < 1)
            throw new ND4JIllegalStateException(
                    "Can't reshape(long...) without shape arguments. Got empty shape instead.");

        // TODO: maybe toFlatten() makes more sense here?
        // reshape(-1) special case
        if (newShape.length == 1 && newShape[0] == -1)
            newShape[0] = this.length();

        int numberNegativesOnes = 0;
        long[] shape = ArrayUtil.copy(newShape);


        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 0) {
                if (numberNegativesOnes >= 1)
                    throw new IllegalArgumentException("Only one dimension can be negative ones. Got shape "
                            + Arrays.toString(newShape));

                numberNegativesOnes++;

                int shapeLength = 1;
                for (int j = 0; j < shape.length; j++)
                    if (shape[j] >= 1)
                        shapeLength *= shape[j];
                long realShape = Math.abs(length() / shapeLength);
                long[] thisNewShape = new long[shape.length];
                for (int j = 0; j < shape.length; j++) {
                    if (i != j) {
                        thisNewShape[j] = shape[j];
                    } else
                        thisNewShape[j] = realShape;
                }

                shape = thisNewShape;
                break;

            }
        }

        long prod = ArrayUtil.prodLong(shape);

        if (prod != this.lengthLong()){
            throw new ND4JIllegalStateException("New shape length doesn't match original length: [" + prod + "] vs [" + this.lengthLong() + "]. Original shape: "+Arrays.toString(this.shape())+" New Shape: "+Arrays.toString(newShape));
        }





        INDArray reshapeAttempt = Shape.newShapeNoCopy(this, shape, order == 'f');
        if (reshapeAttempt != null) {
            // kinda strange get/set usage
            //  reshapeAttempt.setOrder(Shape.getOrder(reshapeAttempt));
            return reshapeAttempt;
        }


        INDArray ret = Nd4j.createUninitialized(shape, order);
        if (order != ordering()) {
            ret.setData(dup(order).data());
        } else
            ret.assign(this);
        return ret;
    }

    @Override
    public double getDoubleUnsafe(long offset) {
        return data().getDouble(offset);
    }

    @Override
    public INDArray putScalarUnsafe(long offset, double value) {
        autoProcessScalarCall();
        data().put(offset, value);
        return this;
    }

    @Override
    public int innerMostStride() {
        if (ordering() == 'c')
            return stride(-1);
        return stride(0);
    }

    @Override
    public INDArray reshape(char order, int rows, int columns) {
        return reshape(order, new long[] {rows, columns});
    }

    /**
     * Reshape the ndarray in to the specified dimensions,
     * possible errors being thrown for invalid shapes
     *
     * Note here that one dimension can be -1.
     * The dimension that is -1 will be inferred from the shape and
     * the length of the ndarray
     *
     * @param shape the shape of the ndarray.
     * @return the new reshaped nd array
     */

    @Override
    public INDArray reshape(int[] shape) {
        return reshape(Nd4j.order(), shape);
    }

    @Override
    public INDArray reshape(long... shape) {
        return reshape(Nd4j.order(), shape);
    }

    @Override
    public void checkDimensions(INDArray other) {
        assert Shape.contentEquals(other.shape(),
                Shape.shapeOf(shapeInformation)) : " Other array should have been shape: "
                + Shape.toString(Shape.shapeOf(shapeInformation)) + " but was "
                + Arrays.toString(other.shape());
        assert Shape.contentEquals(other.stride(),
                Shape.stride(shapeInformation)) : " Other array should have been stride: "
                + Shape.toString(Shape.stride(shapeInformation)) + " but was "
                + Arrays.toString(other.stride());
        assert Shape.offset(jvmShapeInfo.javaShapeInformation) == other.offset() : "Offset of this array is "
                + Shape.offset(jvmShapeInfo.javaShapeInformation) + " but other was " + other.offset();

    }


    /**
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
    @Override
    public INDArray prod(int... dimension) {
        return Nd4j.getExecutioner().exec(new Prod(this), dimension);
    }

    /**
     * Returns the overall mean of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray mean(int... dimension) {
        return Nd4j.getExecutioner().exec(new Mean(this), dimension);
    }

    @Override
    public INDArray amean(int... dimension) {
        return Nd4j.getExecutioner().exec(new AMean(this), dimension);
    }

    @Override
    public INDArray mean(@NonNull INDArray result, int... dimension) {
        return Nd4j.getExecutioner().exec(new Mean(this, null, result), dimension);
    }

    /**
     * Returns the overall variance of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray var(int... dimension) {
        return Nd4j.getExecutioner().exec(new Variance(this), dimension);
    }

    /**
     * Returns the overall variance of this ndarray
     *
     * @param biasCorrected boolean on whether to apply corrected bias
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray var(boolean biasCorrected, int... dimension) {
        return Nd4j.getExecutioner().exec(new Variance(this, biasCorrected), dimension);
    }

    /**
     * Returns the overall max of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray max(int... dimension) {
        return Nd4j.getExecutioner().exec(new Max(this), dimension);
    }

    @Override
    public INDArray amax(int... dimension) {
        return Nd4j.getExecutioner().exec(new AMax(this), dimension);
    }

    /**
     * Returns the overall min of this ndarray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this ndarray
     */
    @Override
    public INDArray min(int... dimension) {
        return Nd4j.getExecutioner().exec(new Min(this), dimension);
    }

    @Override
    public INDArray amin(int... dimension) {
        return Nd4j.getExecutioner().exec(new AMin(this), dimension);
    }

    /**
     * Returns the sum along the last dimension of this ndarray
     *
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this ndarray
     */
    @Override
    public INDArray sum(int... dimension) {
        return Nd4j.getExecutioner().exec(new Sum(this), dimension);
    }


    /**
     * Returns entropy along dimension
     * @param dimension
     * @return
     */
    @Override
    public INDArray entropy(int... dimension) {
        return Nd4j.getExecutioner().exec(new Entropy(this), dimension);
    }

    /**
     * Returns non-normalized Shannon entropy along dimension
     * @param dimension
     * @return
     */
    @Override
    public INDArray shannonEntropy(int... dimension) {
        return Nd4j.getExecutioner().exec(new ShannonEntropy(this), dimension);
    }

    /**
     * Returns log entropy along dimension
     * @param dimension
     * @return
     */
    @Override
    public INDArray logEntropy(int... dimension) {
        return Nd4j.getExecutioner().exec(new LogEntropy(this), dimension);
    }

    @Override
    public INDArray sum(@NonNull INDArray result, int... dimension) {
        return Nd4j.getExecutioner().exec(new Sum(this, null, result), dimension);
    }


    /**
     * Returns the norm1 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    @Override
    public INDArray norm1(int... dimension) {
        return Nd4j.getExecutioner().exec(new Norm1(this), dimension);
    }


    /**
     * Standard deviation of an ndarray along a dimension
     *
     * @param dimension the dimension to getScalar the std along
     * @return the standard deviation along a particular dimension
     */
    @Override
    public INDArray std(int... dimension) {
        return Nd4j.getExecutioner().exec(new StandardDeviation(this), dimension);
    }

    @Override
    public INDArray std(boolean biasCorrected, int... dimension) {
        return Nd4j.getExecutioner().exec(new StandardDeviation(this, biasCorrected),
                dimension);
    }

    @Override
    public Number stdNumber(boolean biasCorrected) {
        return Nd4j.getExecutioner().exec(new StandardDeviation(this, biasCorrected),
                new int[] {Integer.MAX_VALUE})
                .getDouble(0);
    }

    /**
     * Returns the norm2 along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    @Override
    public INDArray norm2(int... dimension) {
        return Nd4j.getExecutioner().exec(new Norm2(this), dimension);
    }



    /**
     * Number of columns (shape[1]), throws an exception when
     * called when not 2d
     *
     * @return the number of columns in the array (only 2d)
     */
    @Override
    public int columns() {
        // FIXME: int cast
        if (isMatrix())
            return (int) size(1);
        else if (Shape.isColumnVectorShape(shape())) {
            return 1;
        } else if (Shape.isRowVectorShape(shape())) {
            return (int) length();
        }
        throw new IllegalStateException("Rank is [" + rank() + "]; columns() call is not valid");


    }

    /**
     * Returns the number of rows
     * in the array (only 2d) throws an exception when
     * called when not 2d
     *
     * @return the number of rows in the matrix
     */
    @Override
    public int rows() {
        // FIXME:
        if (isMatrix())
            return (int) size(0);
        else if (Shape.isRowVectorShape(shape())) {
            return 1;
        } else if (Shape.isColumnVectorShape(shape())) {
            return (int) length();
        }

        throw new IllegalStateException("Rank is " + rank() + " rows() call is not valid");
    }


    /**
     * Flattens the array for linear indexing
     *
     * @return the flattened version of this array
     */
    @Override
    public INDArray ravel(char ordering) {
        Nd4j.getCompressor().autoDecompress(this);


        if (length() >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Length can not be >= Integer.MAX_VALUE");
        INDArray ret = create(new int[] {1, (int) length()}, ordering);
        NDArrayIndex index = new NDArrayIndex(this.shape());

        for (int i = 0; i < length(); i++) {
            // FIXME: LONG
            double val = getDouble((int) index.next());
            ret.putScalar(new int[] {0, i}, val);
        }

        return ret;

    }

    /**
     * Flattens the array for linear indexing
     *
     * @return the flattened version of this array
     */
    @Override
    public INDArray ravel() {
        return reshape(1, length());
    }

    /**
     * Flattens the array for linear indexing
     *
     * @return the flattened version of this array
     */
    @Override
    public void sliceVectors(List<INDArray> list) {
        if (isVector())
            list.add(this);
        else {
            for (int i = 0; i < slices(); i++) {
                slice(i).sliceVectors(list);
            }
        }
    }

    /**
     * Reshape the matrix. Number of elements must not change.
     *
     * @param newRows
     * @param newColumns
     */
    @Override
    public INDArray reshape(long newRows, long newColumns) {
        return reshape(new long[] {newRows, newColumns});
    }

    /**
     * Get the specified column
     *
     * @param c
     */
    @Override
    public INDArray getColumn(long c) {
        Nd4j.getCompressor().autoDecompress(this);

        if (isColumnVector() && c == 0)
            return this;
        else if (isColumnVector() && c > 0)
            throw new IllegalArgumentException("Illegal index for row");
        else if(isRowVector()) {
            return Nd4j.scalar(getDouble(c));
        }
        return get(NDArrayIndex.all(), NDArrayIndex.point(c));
    }


    /**
     * Get whole rows from the passed indices.
     *
     * @param rindices
     */
    @Override
    public INDArray getRows(int[] rindices) {
        Nd4j.getCompressor().autoDecompress(this);

        if (!isMatrix() && !isVector())
            throw new IllegalArgumentException("Unable to get columns from a non matrix or vector");
        if (isVector())
            return Nd4j.pullRows(this, 1, rindices);
        else {
            INDArray ret = Nd4j.create(rindices.length, columns());
            for (int i = 0; i < rindices.length; i++)
                ret.putRow(i, getRow(rindices[i]));
            return ret;
        }
    }

    /**
     * Returns a subset of this array based on the specified
     * indexes
     *
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    @Override
    public INDArray get(INDArrayIndex... indexes) {
        Nd4j.getCompressor().autoDecompress(this);
        if(indexes.length > rank()) {
            int numNonNewAxis = 0;
            for(int i = 0; i < indexes.length; i++) {
                if(!(indexes[i] instanceof NewAxis))
                    numNonNewAxis++;
            }

            if(numNonNewAxis > rank()) {
                throw new IllegalArgumentException("Too many indices for array. Number of indexes must be <= rank()");
            }
        }


        //check for row/column vector and point index being 0
        if (indexes.length == 1 && indexes[0] instanceof NDArrayIndexAll || (indexes.length == 2 && (isRowVector()
                && indexes[0] instanceof PointIndex && indexes[0].offset() == 0
                && indexes[1] instanceof NDArrayIndexAll
                || isColumnVector() && indexes[1] instanceof PointIndex && indexes[0].offset() == 0
                && indexes[0] instanceof NDArrayIndexAll)))
            return this;

        indexes = NDArrayIndex.resolve(shapeInfoDataBuffer(), indexes);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(this);
        resolution.exec(indexes);

        if (indexes.length < 1)
            throw new IllegalStateException("Invalid index found of zero length");

        // FIXME: LONG
        int[] shape = LongUtils.toInts(resolution.getShapes());
        int numSpecifiedIndex = 0;
        for (int i = 0; i < indexes.length; i++)
            if (indexes[i] instanceof SpecifiedIndex)
                numSpecifiedIndex++;

        if (shape != null && numSpecifiedIndex > 0) {
            Generator<List<List<Long>>> gen = SpecifiedIndex.iterate(indexes);
            INDArray ret = Nd4j.create(shape, 'c');
            int count = 0;
            while (true) {
                try {
                    List<List<Long>> next = gen.next();
                    List<Long> coordsCombo = new ArrayList<>();
                    for (int i = 0; i < next.size(); i++) {
                        if (next.get(i).size() > 1)
                            throw new IllegalStateException("Illegal entry returned");
                        coordsCombo.add(next.get(i).get(0));
                    }
                    ret.putScalar(count++, getDouble(Ints.toArray(coordsCombo)));


                } catch (NoSuchElementException e) {
                    break;
                }

                if (count >= ret.length())
                    break;
            }

            return ret;

        }

        INDArray ret = subArray(resolution);
        return ret;
    }


    /**
     * Get whole columns
     * from the passed indices.
     *
     * @param cindices
     */
    @Override
    public INDArray getColumns(int... cindices) {
        if (!isMatrix() && !isVector())
            throw new IllegalArgumentException("Unable to get columns from a non matrix or vector");
        if (isVector()) {
            return Nd4j.pullRows(this, 0, cindices, this.ordering());
        } else {
            INDArray ret = Nd4j.create(rows(), cindices.length);
            for (int i = 0; i < cindices.length; i++)
                ret.putColumn(i, getColumn(cindices[i]));
            return ret;
        }

    }

    protected INDArray create(int rows, int length) {
        return create(new int[] {rows, length});
    }

    /**
     * Get a copy of a row.
     *
     * @param r the row to get
     */
    @Override
    public INDArray getRow(long r) {
        if (isRowVector() && r == 0)
            return this;
        else if (isRowVector() && r > 0)
            throw new IllegalArgumentException("Illegal index for row: requested row " + r + " but this.size(0)=" + this.size(0));
        INDArray result = get(NDArrayIndex.point(r), NDArrayIndex.all());

        // FIXME: this is bad
        if (!this.isView() && this.ordering() == 'c' && result.elementWiseStride() == 1 && result.ordering() != 'c') {
            ((BaseNDArray) result).setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(result.shape(), result.stride(), 1, 'c', this.dataType()));
        }

        return result;
    }


    /**
     * This method allows you to compare INDArray against other INDArray, with variable eps
     *
     * @param o
     * @param eps
     * @return
     */
    public boolean equalsWithEps(Object o, double eps) {
        Nd4j.getCompressor().autoDecompress(this);


        if (o == null)
            return false;

        if (!(o instanceof INDArray))
            return false;

        INDArray n = (INDArray) o;

        if (n.isSparse()) {
            return n.equals(this);
        }

        if (this.length() != n.length())
            return false;

        if (this.isEmpty() != n.isEmpty())
            return false;

        if (this.isEmpty() && n.isEmpty())
            return true;

        //epsilon equals
        if (isScalar() && n.isScalar()) {
            if (data.dataType() == DataBuffer.Type.FLOAT) {
                double val = getDouble(0);
                double val2 = n.getDouble(0);

                if (Double.isNaN(val) != Double.isNaN(val2))
                    return false;

                return Math.abs(val - val2) < eps;
            } else {
                double val = getDouble(0);
                double val2 = n.getDouble(0);

                if (Double.isNaN(val) != Double.isNaN(val2))
                    return false;

                return Math.abs(val - val2) < eps;
            }

        } else if (isVector() && n.isVector()) {

            EqualsWithEps op = new EqualsWithEps(this, n, eps);
            Nd4j.getExecutioner().exec(op);
            double diff = op.getFinalResult().doubleValue();

            return diff < 0.5;
        }

        if (!Arrays.equals(this.shape(), n.shape()))
            return false;


        if (!Shape.shapeEquals(shape(), n.shape())) {
            return false;
        }


        if (slices() != n.slices())
            return false;

        if (n.ordering() == ordering()) {
            EqualsWithEps op = new EqualsWithEps(this, n, eps);
            Nd4j.getExecutioner().exec(op);
            double diff = op.getFinalResult().doubleValue();

            return diff < 0.5;
        } else {
            EqualsWithEps op = new EqualsWithEps(this, n, eps);
            Nd4j.getExecutioner().exec(op);
            double diff = op.getFinalResult().doubleValue();

            return diff < 0.5;
        }
    }

    @Override
    public boolean equalShapes(@NonNull INDArray other){
        if(rank() != other.rank())
            return false;
        for( int i=0; i<rank(); i++ ){
            if(size(i) != other.size(i)){
                return false;
            }
        }
        return true;
    }

    /**
     * Compare two matrices. Returns true if and only if other is also a
     * DoubleMatrix which has the same size and the maximal absolute
     * difference in matrix elements is smaller than 1e-5.
     *
     * @param o
     */
    @Override
    public boolean equals(Object o) {
        return equalsWithEps(o, Nd4j.EPS_THRESHOLD);
    }

    @Override
    public DataBuffer shapeInfoDataBuffer() {
        return shapeInformation;
    }

    @Override
    public LongBuffer shapeInfo() {
        return shapeInformation.asNioLong();
    }

    /**
     * Returns the shape(dimensions) of this array
     *
     * @return the shape of this matrix
     */
    public long[] shape() {
        return jvmShapeInfo.shape;
    }

    /**
     * Returns the shape information debugging
     * information
     *
     * @return the shape information debugging information
     */
    @Override
    public String shapeInfoToString() {
        return Shape.shapeToString(this);
    }

    /**
     * Returns the stride(indices along the linear index for which each slice is accessed) of this array
     *
     * @return the stride of this array
     */
    @Override
    public long[] stride() {
        return jvmShapeInfo.stride;
    }


    @Override
    public long offset() {
        if (data().offset() >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Offset of buffer can not be >= Integer.MAX_VALUE");
        //  return Shape.offset(shapeInfo());
        return data().offset();
    }

    @Override
    public char ordering() {
        return jvmShapeInfo.order;
    }

    /**
     * Returns the size of this array
     * along a particular dimension
     *
     * @param dimension the dimension to return from
     * @return the shape of the specified dimension
     */
    @Override
    public long size(int dimension) {
        if (dimension < 0)
            dimension += jvmShapeInfo.rank;

        if (isScalar()) {
            if (dimension == 0 || dimension == 1 || dimension < 0)
                return length();
            else
                throw new IllegalArgumentException("Illegal dimension for scalar " + dimension);
        }

        if (dimension >= rank())
            throw new IllegalArgumentException("Invalid size: cannot get size of dimension " + dimension + " for rank "
                    + rank() + " NDArray (array shape: " + Arrays.toString(this.shape()) + ")");

        return jvmShapeInfo.shape[dimension];
    }

    @Override
    public int rank() {
        return jvmShapeInfo.rank;
    }

    /**
     * Returns the total number of elements in the ndarray
     *
     * @return the number of elements in the ndarray
     */
    @Override
    public long length() {
        return jvmShapeInfo.length;
    }

    /**
     * Returns the total number of elements in the ndarray
     *
     * @return the number of elements in the ndarray
     */
    @Override
    @Deprecated
    public long lengthLong() {
        return jvmShapeInfo.length;
    }

    @Override
    public INDArray broadcast(INDArray result) {
        Nd4j.getCompressor().autoDecompress(this);

        val shape = result.shape();

        if (Shape.shapeEquals(shape, shape()))
            return this;

        // if we're on scalar, we can just create new array
        if (this.isScalar())
            return Nd4j.createUninitialized(shape).assign(this.getDouble(0));




        boolean compatible = true;
        int count = shape.length - 1;
        int thisCount = jvmShapeInfo.rank - 1;
        for (int i = shape.length - 1; i > 0; i--) {
            if (count < 0 || thisCount < 0)
                break;
            if (shape[count] != shape()[thisCount] && shape[count] != 1 && shape()[thisCount] != 1) {
                compatible = false;
                break;
            }

            count--;
            thisCount--;
        }

        if (!compatible)
            throw new IllegalArgumentException("Incompatible broadcast from " + Arrays.toString(shape()) + " to "
                    + Arrays.toString(shape));



        long[] retShape = new long[shape.length];
        List<Integer> broadCastDimensions = new ArrayList<>();
        List<Integer> nonBroadCastDimensions = new ArrayList<>();
        for (int i = 0; i < retShape.length; i++) {
            if (shape().length == 1) {
                if (i == 0) {
                    if (i < shape().length)
                        retShape[i] = Math.max(1, shape[i]);
                    else
                        retShape[i] = shape[i];
                } else {
                    if (i < shape().length)
                        retShape[i] = Math.max(shape[i], size(i));
                    else
                        retShape[i] = shape[i];
                }
            } else {
                if (i < rank() && size(i) == 1)
                    broadCastDimensions.add(i);
                else
                    nonBroadCastDimensions.add(i);
                if (i < shape().length)
                    retShape[i] = Math.max(shape[i], size(i));
                else
                    retShape[i] = shape[i];
            }

        }


        if (isRowVector()) {
            //number of times to repeat each value
            for (int i = 0; i < result.slices(); i++) {
                result.putSlice(i, this);
            }
        } else if (isColumnVector()) {
            for (int i = 0; i < result.columns(); i++) {
                result.putColumn(i, this);
            }
        }

        else {
            // FIXME: int cast
            int[] repeat = new int[shape.length];
            for(int i = 0; i < shape.length; i++) {
                if(i < rank()) {
                    if(size(i) == 1)
                        repeat[i] = (int) shape[i];
                    else {
                        repeat[i] = 1;
                    }
                }

                else {
                    repeat[i] = (int) shape[i];
                }
            }

            if (this.isView()) {
                Nd4j.getExecutioner().exec(new Tile(new INDArray[]{this.dup(this.ordering())},new INDArray[]{result},repeat));
            } else
                Nd4j.getExecutioner().exec(new Tile(new INDArray[]{this},new INDArray[]{result},repeat));

            //result = Nd4j.tile(this,repeat);
        }
        return result;

    }

    /**
     * Broadcasts this ndarray to be the specified shape
     *
     * @param shape the new shape of this ndarray
     * @return the broadcasted ndarray
     */
    @Override
    public INDArray broadcast(long... shape) {
      return broadcast(Nd4j.createUninitialized(shape));
    }

    @Override
    public INDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable) {
        // FIXME: int cast
        return dimShuffle(rearrange, ArrayUtil.toLongArray(newOrder), broadCastable);
    }

    /**
     * Dimshuffle: an extension of permute that adds the ability
     * to broadcast various dimensions.
     * <p/>
     * See theano for more examples.
     * This will only accept integers and xs.
     * <p/>
     * An x indicates a dimension should be broadcasted rather than permuted.
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    @Override
    public INDArray dimShuffle(Object[] rearrange, long[] newOrder, boolean[] broadCastable) {
        Nd4j.getCompressor().autoDecompress(this);

        if (broadCastable.length != jvmShapeInfo.rank)
            throw new IllegalArgumentException(
                    "The broadcastable dimensions must be the same length as the current shape");

        boolean broadcast = false;
        Set<Object> set = new HashSet<>();
        for (int i = 0; i < rearrange.length; i++) {
            set.add(rearrange[i]);
            if (rearrange[i] instanceof Integer) {
                Integer j = (Integer) rearrange[i];
                if (j >= broadCastable.length)
                    throw new IllegalArgumentException(
                            "Illegal dimension, dimension must be < broadcastable.length (aka the real dimensions");
            } else if (rearrange[i] instanceof Character) {
                Character c = (Character) rearrange[i];
                if (c != 'x')
                    throw new IllegalArgumentException("Illegal input: Must be x");
                broadcast = true;

            } else
                throw new IllegalArgumentException("Only characters and integers allowed");
        }

        //just do permute
        if (!broadcast) {
            int[] ret = new int[rearrange.length];
            for (int i = 0; i < ret.length; i++)
                ret[i] = (Integer) rearrange[i];
            return permute(ret);
        } else {
            List<Integer> drop = new ArrayList<>();
            for (int i = 0; i < broadCastable.length; i++) {
                if (!set.contains(i)) {
                    if (broadCastable[i])
                        drop.add(i);
                    else
                        throw new IllegalArgumentException(
                                "We can't drop the given dimension because its not broadcastable");
                }

            }


            //list of dimensions to keep
            int[] shuffle = new int[broadCastable.length];
            int count = 0;
            for (int i = 0; i < rearrange.length; i++) {
                if (rearrange[i] instanceof Integer) {
                    shuffle[count++] = (Integer) rearrange[i];
                }
            }


            List<Integer> augment = new ArrayList<>();
            for (int i = 0; i < rearrange.length; i++) {
                if (rearrange[i] instanceof Character)
                    augment.add(i);
            }

            Integer[] augmentDims = augment.toArray(new Integer[1]);

            count = 0;

            int dropIdx = 0;
            int[] newShape = new int[shuffle.length + drop.size()];
            for (int i = 0; i < newShape.length; i++) {
                if (i < shuffle.length) {
                    newShape[count++] = shuffle[i];
                } else
                    newShape[count++] = drop.get(dropIdx++);
            }

            INDArray ret;   //TODO is this correct? This was old behaviour before adding permute input check
            if(newShape.length == this.rank()){
                ret = permute(newShape);
            } else {
                ret = dup();
            }
            List<Long> newDims = new ArrayList<>();
            long[] shape = Arrays.copyOfRange(ret.shape(), 0, shuffle.length);
            for (int i = 0; i < shape.length; i++) {
                newDims.add(shape[i]);
            }

            for (int i = 0; i < augmentDims.length; i++) {
                newDims.add(augmentDims[i], 1L);
            }

            long[] toReshape = ArrayUtil.toArrayLong(newDims);


            ret = ret.reshape(toReshape);
            return ret;

        }


    }

    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    @Override
    public INDArray permute(int... rearrange) {
        Preconditions.checkArgument(rearrange.length == rank(), "Incorrect number of arguments for permute function:" +
                " got arguments %s for rank %s array. Number of arguments must equal array rank", rearrange, rank());
        Nd4j.getCompressor().autoDecompress(this);
        boolean alreadyInOrder = true;
        //IntBuffer shapeInfo = shapeInfo();
        int rank = jvmShapeInfo.rank;
        for (int i = 0; i < rank; i++) {
            if (rearrange[i] != i) {
                alreadyInOrder = false;
                break;
            }
        }

        if (alreadyInOrder)
            return this;

        checkArrangeArray(rearrange);
        int[] newShape = doPermuteSwap(shapeOf(), rearrange);
        int[] newStride = doPermuteSwap(strideOf(), rearrange);

        char newOrder = Shape.getOrder(newShape, newStride, elementStride());

        INDArray value = create(data(), newShape, newStride, offset(), newOrder);
        return value;
    }

    /**
     * An <b>in-place</b> version of permute. The array  shape information (shape, strides)
     * is modified by this operation (but not the data itself)
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     *
     * @param rearrange the dimensions to swap to
     * @return the current array
     */
    @Override
    public INDArray permutei(int... rearrange) {
        Preconditions.checkArgument(rearrange.length == rank(), "Incorrect number of arguments for permute function:" +
                " got arguments %s for rank %s array. Number of arguments must equal array rank", rearrange, rank());
        boolean alreadyInOrder = true;
        val shapeInfo = shapeInfo();
        int rank = jvmShapeInfo.rank;
        for (int i = 0; i < rank; i++) {
            if (rearrange[i] != i) {
                alreadyInOrder = false;
                break;
            }
        }

        if (alreadyInOrder)
            return this;

        checkArrangeArray(rearrange);
        val newShape = doPermuteSwap(Shape.shapeOf(shapeInfo), rearrange);
        val newStride = doPermuteSwap(Shape.stride(shapeInfo), rearrange);
        char newOrder = Shape.getOrder(newShape, newStride, elementStride());

        //Set the shape information of this array: shape, stride, order.
        //Shape info buffer: [rank, [shape], [stride], offset, elementwiseStride, order]
        /*for( int i=0; i<rank; i++ ){
            shapeInfo.put(1+i,newShape[i]);
            shapeInfo.put(1+i+rank,newStride[i]);
        }
        shapeInfo.put(3+2*rank,newOrder);
        */
        val ews = shapeInfo.get(2 * rank + 2);
        /*
        if (ews < 1 && !attemptedToFindElementWiseStride)
            throw new RuntimeException("EWS is -1");
            */

        val si = Nd4j.getShapeInfoProvider().createShapeInformation(newShape, newStride,  ews, newOrder, dataType());
        setShapeInformation(si);


        if (shapeInfo.get(2 * rank + 2) > 0) {
            //for the backend to work - no ews for permutei
            //^^ not true anymore? Not sure here. Marking this for raver
            setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(newShape, newStride, -1, newOrder, dataType()));
        }

        //this.shape = null;
        //this.stride = null;


        return this;
    }


    protected long[] doPermuteSwap(LongBuffer shape, int[] rearrange) {
        val ret = new long[rearrange.length];
        for (int i = 0; i < rearrange.length; i++) {
            ret[i] = shape.get(rearrange[i]);
        }
        return ret;
    }

    protected int[] doPermuteSwap(IntBuffer shape, int[] rearrange) {
        int[] ret = new int[rearrange.length];
        for (int i = 0; i < rearrange.length; i++) {
            ret[i] = shape.get(rearrange[i]);
        }
        return ret;
    }

    protected int[] doPermuteSwap(DataBuffer shape, int[] rearrange) {
        int[] ret = new int[rearrange.length];
        for (int i = 0; i < rearrange.length; i++) {
            ret[i] = shape.getInt(rearrange[i]);
        }

        return ret;
    }


    protected void checkArrangeArray(int[] arr) {
        assert arr.length == jvmShapeInfo.rank : "Invalid rearrangement: number of arrangement != shape";
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] >= arr.length)
                throw new IllegalArgumentException("The specified dimensions can't be swapped. Given element " + i
                        + " was >= number of dimensions");
            if (arr[i] < 0)
                throw new IllegalArgumentException("Invalid dimension: " + i + " : negative value");


        }

        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr.length; j++) {
                if (i != j && arr[i] == arr[j])
                    throw new IllegalArgumentException("Permute array must have unique elements");
            }
        }

    }

    protected void autoProcessScalarCall() {
       /* if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.DISABLED && Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.SCOPE_PANIC)
            OpProfiler.getInstance().processScalarCall();*/
    }

    /**
     * Checks whether the matrix is a vector.
     */
    @Override
    public boolean isVector() {
        if (jvmShapeInfo.rank == 1)
            return true;

        return isRowVector() || isColumnVector();
    }

    @Override
    public boolean isVectorOrScalar() {
        return isVector() || isScalar();
    }

    @Override
    public boolean isSquare() {
        return isMatrix() && rows() == columns();
    }

    /**
     * Checks whether the matrix is a row vector.
     */
    @Override
    public boolean isRowVector() {
        return (rank() == 2 && rows() == 1) && length() > 1 || rank() == 1 && length() > 1;
    }

    /**
     * Checks whether the matrix is a column vector.
     */
    @Override
    public boolean isColumnVector() {
        return rank() == 2 && columns() == 1 && length() > 1;
    }

    @Override
    public boolean isColumnVectorOrScalar() {
        return isColumnVector() || isScalar();
    }

    @Override
    public boolean isRowVectorOrScalar() {
        return isRowVector() || isScalar();
    }

    /**
     * Generate string representation of the matrix.
     * Printing will switch to scientific notation on a per element basis
     *      - when abs value is greater than or equal to 10000
     *      - when abs value is less than or equal to 0.0001 and not zero
     *
     *  If the number of elements in the array is greater than 1000 (by default) only the first and last three elements
     *  in a dimension are included. This can be changed globally using {@link NDArrayStrings#setMaxPrintElements(long)}
     *
     *
     */
    @Override
    public String toString() {
        if (!isCompressed() && !preventUnpack)
            return new NDArrayStrings().format(this);
        else if (isCompressed() && compressDebug)
            return "COMPRESSED ARRAY. SYSTEM PROPERTY compressdebug is true. This is to prevent auto decompression from being triggered.";
        else if (preventUnpack)
            return "Array string unpacking is disabled.";
        return new NDArrayStrings().format(this);
    }

    /**
     * Returns a scalar (individual element)
     * of a scalar ndarray
     *
     * @return the individual item in this ndarray
     */
    @Override
    public Object element() {

        if (!isScalar())
            throw new IllegalStateException("Unable to retrieve element from non scalar matrix");
        if (data.dataType() == DataBuffer.Type.FLOAT)
            return data.getFloat(0);
        return data.getDouble(0);
    }

    @Override
    public INDArray remainder(INDArray denominator) {
        return remainder(denominator, Nd4j.createUninitialized(this.shape()));
    }

    @Override
    public INDArray remainder(INDArray denominator, INDArray result) {
        RemainderOp op = new RemainderOp(this, denominator, result);
        Nd4j.getExecutioner().exec(op);
        return result;
    }

    @Override
    public INDArray remainder(Number denominator) {
        return remainder(denominator, Nd4j.createUninitialized(this.shape()));
    }

    @Override
    public INDArray remainder(Number denominator, INDArray result) {
        ScalarRemainder op = new ScalarRemainder(this, null, result, this.length(), denominator);
        Nd4j.getExecutioner().exec(op);
        return result;
    }

    @Override
    public INDArray remainderi(INDArray denominator) {
        RemainderOp op = new RemainderOp(this, denominator, this);
        Nd4j.getExecutioner().exec(op);
        return this;
    }

    @Override
    public INDArray remainderi(Number denominator) {
        ScalarRemainder op = new ScalarRemainder(this, null, this, this.length(), denominator);
        Nd4j.getExecutioner().exec(op);
        return this;
    }

    @Override
    public INDArray fmod(INDArray denominator) {
        return fmod(denominator, Nd4j.createUninitialized(this.shape()));
    }

    @Override
    public INDArray fmod(INDArray denominator, INDArray result) {
        OldFModOp op = new OldFModOp(this, denominator, result);
        Nd4j.getExecutioner().exec(op);
        return result;
    }

    @Override
    public INDArray fmod(Number denominator) {
        return fmod(denominator, Nd4j.createUninitialized(this.shape()));
    }

    @Override
    public INDArray fmod(Number denominator, INDArray result) {
        ScalarFMod op = new ScalarFMod(this, null, result, this.length(), denominator);
        Nd4j.getExecutioner().exec(op);
        return result;
    }

    @Override
    public INDArray fmodi(INDArray denominator) {
        OldFModOp op = new OldFModOp(this, denominator, this);
        Nd4j.getExecutioner().exec(op);
        return this;
    }

    @Override
    public INDArray fmodi(Number denominator) {
        ScalarFMod op = new ScalarFMod(this, null, this, this.length(), denominator);
        Nd4j.getExecutioner().exec(op);
        return this;
    }

    @Override
    public Iterator<Object> iterator() {
        return new FirstAxisIterator(this);
    }

    /**
     * Returns the start of where the ndarray is for the original data buffer
     *
     * @return
     */
    @Override
    public long originalOffset() {
        if (data().originalOffset() >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Original offset of buffer can not be >= Integer.MAX_VALUE");

        return data().originalOffset();
    }

    private void readObject(ObjectInputStream s) {
        try {
            s.defaultReadObject();
            read(s);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    private void writeObject(ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        write(out);
    }

    //Custom serialization for Java serialization
    protected void write(ObjectOutputStream out) throws IOException {
        if (this.isView()) {
            //As per Nd4j.write, duplicate before writing to the output stream
            //BaseDataBuffer.write(...) doesn't know about strides etc, so dup (or equiv. strategy) is necessary here
            //Furthermore, because we only want to save the *actual* data for a view (not the full data), the shape info
            // (mainly strides, offset, element-wise stride) may be different in the duped array vs. the view array
            INDArray copy = this.dup();
            copy.shapeInfoDataBuffer().write(out);
            copy.data().write(out);
        } else {
            shapeInformation.write(out);
            data().write(out);
        }
    }

    //Custom deserialization for Java serialization
    protected void read(ObjectInputStream s) {
        shapeInformation = Nd4j.createBuffer(new int[Shape.shapeInfoLength(rank())], 0);
        shapeInformation.read(s);
        setShapeInformation(Pair.create(shapeInformation, shapeInformation.asLong()));
        data = Nd4j.createBuffer(length(), false);
        data().read(s);
    }


    /**
     * This method returns index of highest value along specified dimension(s)
     *
     * @param dimension
     * @return
     */
    @Override
    public INDArray argMax(int... dimension) {
        return Nd4j.argMax(this, dimension);
    }


    /**
     * This method returns True, if this INDArray instance is attached to some Workspace. False otherwise.
     *
     * @return
     */
    @Override
    public boolean isAttached() {
        if (isEmpty())
            return false;

        if (data == null && !isEmpty())
            throw new IllegalStateException();

        return data.isAttached() ||
                (data.underlyingDataBuffer() != null && data.underlyingDataBuffer().isAttached()) ||
                (data.originalDataBuffer() != null && data.originalDataBuffer().isAttached());
    }

    /**
     * This method checks, if given attached INDArray is still in scope of its parent Workspace
     * <p>
     * PLEASE NOTE: if this INDArray isn't attached to any Workspace, this method will return true
     *
     * @return
     */
    @Override
    public boolean isInScope() {
        if (!isAttached())
            return true;

        return data.isInScope();
    }

    /**
     * This metod detaches INDArray from Workspace, returning copy. Basically it's dup() into new memory chunk.
     * <p>
     * PLEASE NOTE: If this INDArray instance is NOT attached - it will be returned unmodified.
     *
     * @return
     */
    @Override
    public INDArray detach() {
        if (!isAttached())
            return this;

        WorkspaceUtils.assertValidArray(this, "Cannot detach INDArray");

        Nd4j.getExecutioner().commit();

        /*
         two options here
         1) we're within some workspace
         2) we're out of any workspace
        */
        if (Nd4j.getMemoryManager().getCurrentWorkspace() == null) {
            if (!isView()) {
                Nd4j.getExecutioner().commit();
                DataBuffer buffer = Nd4j.createBuffer(this.lengthLong(), false);

                Nd4j.getMemoryManager().memcpy(buffer, this.data());

                return Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInfoDataBuffer());
            } else {
                INDArray copy = Nd4j.createUninitialized(this.shape(), this.ordering());
                copy.assign(this);
                Nd4j.getExecutioner().commit();

                return copy;
            }
        } else {
            MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();
            Nd4j.getMemoryManager().setCurrentWorkspace(null);
            INDArray copy = null;

            if (!isView()) {
                Nd4j.getExecutioner().commit();
                DataBuffer buffer = Nd4j.createBuffer(this.lengthLong(), false);

                //Pointer.memcpy(buffer.pointer(), this.data.pointer(), this.lengthLong() * Nd4j.sizeOfDataType(this.data.dataType()));
                Nd4j.getMemoryManager().memcpy(buffer, this.data());

                copy = Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInfoDataBuffer()); //this.dup(this.ordering());


            } else {
                copy = Nd4j.createUninitialized(this.shape(), this.ordering());
                copy.assign(this);
                Nd4j.getExecutioner().commit();
            }

            Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

            return copy;
        }
    }

    /**
     * This method detaches INDArray from current Workspace, and attaches it to Workspace above, if any.
     * <p>
     * PLEASE NOTE: If this INDArray instance is NOT attached - it will be returned unmodified.
     * PLEASE NOTE: If current Workspace is the top-tier one, effect will be equal to detach() call - detached copy will be returned
     *
     * @return
     */
    @Override
    public INDArray leverage() {
        WorkspaceUtils.assertValidArray(this, "Cannot leverage INDArray to new workspace");
        if (!isAttached())
            return this;

        MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        if (workspace == null) {
            return this.detach();
        }

        MemoryWorkspace parentWorkspace = workspace.getParentWorkspace();

        if (this.data.getParentWorkspace() == parentWorkspace)
            return this;

        // if there's no parent ws - just detach
        if (parentWorkspace == null)
            return this.detach();
        else {
            Nd4j.getExecutioner().commit();

            // temporary set parent ws as current ws
            Nd4j.getMemoryManager().setCurrentWorkspace(parentWorkspace);

            INDArray copy = null;
            if (!this.isView()) {
                Nd4j.getExecutioner().commit();
                DataBuffer buffer = Nd4j.createBuffer(this.lengthLong(), false);
                Nd4j.getMemoryManager().memcpy(buffer, this.data());

                copy = Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInfoDataBuffer());
            } else {
                copy = this.dup(this.ordering());
                Nd4j.getExecutioner().commit();
            }

            // restore current ws
            Nd4j.getMemoryManager().setCurrentWorkspace(workspace);
            return copy;
        }
    }

    /**
     * This method detaches INDArray from current Workspace, and attaches it to Workspace with a given Id
     *
     * PLEASE NOTE: If this INDArray instance is NOT attached - it will be returned unmodified.
     * PLEASE NOTE: If Workspace with target Id wasn't created before - this array will be returned unmodified.
     * PLEASE NOTE: If target workspace is the current one - this array will be returned unmodified.
     *
     * @param id
     * @return
     */
    @Override
    public INDArray leverageTo(String id) {
        return leverageTo(id, false);
    }

    /**
     * This method detaches INDArray from current Workspace, and attaches it to Workspace with a given Id.
     * If enforceExistence == true, and no workspace with the specified ID exists, then an {@link Nd4jNoSuchWorkspaceException}
     * is thrown. Otherwise, if enforceExistance == false and no workspace with the specified ID exists, then the current
     * INDArray is returned unmodified (same as {@link #leverage()}
     *
     * @param id ID of the workspace to leverage to
     * @param enforceExistence If true, and the specified workspace does not exist: an {@link Nd4jNoSuchWorkspaceException}
     *                         will be thrown.
     * @return The INDArray, leveraged to the specified workspace
     * @see #leverageTo(String)
     */
    @Override
    public INDArray leverageTo(String id, boolean enforceExistence) throws Nd4jNoSuchWorkspaceException {
        WorkspaceUtils.assertValidArray(this, "Cannot leverage INDArray to new workspace");
        if (!isAttached())
            return this;

        if (!Nd4j.getWorkspaceManager().checkIfWorkspaceExists(id)) {
            if(enforceExistence){
                throw new Nd4jNoSuchWorkspaceException(id);
            } else {
                return this;
            }
        }

        MemoryWorkspace current = Nd4j.getMemoryManager().getCurrentWorkspace();
        MemoryWorkspace target = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(id);

        if (this.data.getParentWorkspace() == target)
            return this;

        Nd4j.getMemoryManager().setCurrentWorkspace(target);
        INDArray copy = null;
        if (!this.isView()) {
            Nd4j.getExecutioner().commit();
            DataBuffer buffer = Nd4j.createBuffer(this.lengthLong(), false);
            Nd4j.getMemoryManager().memcpy(buffer, this.data());

            copy = Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInfoDataBuffer());
        } else {
            copy = this.dup(this.ordering());
            Nd4j.getExecutioner().commit();
        }

        Nd4j.getMemoryManager().setCurrentWorkspace(current);

        return copy;
    }

    /**
     * This method detaches INDArray from current Workspace, and attaches it to Workspace with a given Id, if a workspace
     * with the given ID is open and active.
     *
     * If the workspace does not exist, or is not active, the array is detached from any workspaces.
     *
     * @param id ID of the workspace to leverage to
     * @return The INDArray, leveraged to the specified workspace (if it exists and is active) otherwise the detached array
     * @see #leverageTo(String)
     */
    public INDArray leverageOrDetach(String id){
        if(!isAttached()){
            return this;
        }

        if(!Nd4j.getWorkspaceManager().checkIfWorkspaceExistsAndActive(id)){
            return detach();
        }
        return leverageTo(id);
    }

    /**
     * This method pulls this INDArray into current Workspace.
     *
     * PLEASE NOTE: If there's no current Workspace - INDArray returned as is
     *
     * @return Migrated INDArray or <i>this</i> if no current workspace
     * @see #migrate(boolean)
     */
    @Override
    public INDArray migrate() {
        return migrate(false);
    }

    /**
     * This method pulls this INDArray into current Workspace, or optionally detaches if no workspace is present.<br>
     * That is:<br>
     * If current workspace is present/active, INDArray is migrated to it.<br>
     * If no current workspace is present/active, one of two things occur:
     * 1. If detachOnNoWs arg is true: if there is no current workspace, INDArray is detached
     * 2. If detachOnNoWs arg is false: this INDArray is returned as-is (no-op) - equivalent to {@link #migrate()}
     *
     * @param detachOnNoWs If true: detach on no WS. If false and no workspace: return this.
     * @return Migrated INDArray
     */
    @Override
    public INDArray migrate(boolean detachOnNoWs){
        WorkspaceUtils.assertValidArray(this, "Cannot leverage INDArray to new workspace");

        MemoryWorkspace current = Nd4j.getMemoryManager().getCurrentWorkspace();

        if (current == null) {
            if(detachOnNoWs){
                return detach();
            } else {
                return this;
            }
        }

        INDArray copy = null;

        if (!this.isView()) {
            Nd4j.getExecutioner().commit();
            DataBuffer buffer = Nd4j.createBuffer(this.lengthLong(), false);
            Nd4j.getMemoryManager().memcpy(buffer, this.data());

            copy = Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInfoDataBuffer());
        } else {
            copy = this.dup(this.ordering());
            Nd4j.getExecutioner().commit();
        }

        return copy;
    }

    @Override
    public Number percentileNumber(Number quantile) {
        if (quantile.intValue() < 0 || quantile.intValue() > 100)
            throw new ND4JIllegalStateException("Percentile value should be in 0...100 range");

        if (isScalar())
            return this.getDouble(0);

        INDArray sorted = Nd4j.sort(this.dup(this.ordering()), true);

        return getPercentile(quantile, sorted);
    }

    @Override
    public Number medianNumber() {
        return percentileNumber(50);
    }

    @Override
    public INDArray median(int... dimension) {
        return percentile(50, dimension);
    }

    protected double getPercentile(Number quantile, INDArray sorted) {
        if (quantile.intValue() == 0)
            return sorted.getDouble(0);
        else if (quantile.intValue() == 100)
            return sorted.getDouble(sorted.length() - 1);

        double pos = (quantile.doubleValue() / 100.0) * (double) (sorted.length() + 1);

        double fposition = FastMath.floor(pos);
        int position = (int)fposition;

        double diff = pos - fposition;

        double lower = sorted.getDouble(position-1);
        double upper = sorted.getDouble(position);

        return lower + diff * (upper - lower);
    }

    @Override
    public INDArray percentile(Number quantile, int... dimension) {
        if (quantile.doubleValue() < 0 || quantile.doubleValue() > 100)
            throw new ND4JIllegalStateException("Percentile value should be in 0...100 range");

        if (isScalar())
            return Nd4j.scalar(this.getDouble(0));

        INDArray sorted = Nd4j.getNDArrayFactory().sort(this.dup(this.ordering()), false, dimension);

        // there's no practical sense doing this on GPU, stride will be just size of TAD.
        INDArray ret = Nd4j.createUninitialized(sorted.tensorssAlongDimension(dimension));
        for (int i = 0; i < ret.length(); i++) {
            ret.putScalar(i, getPercentile(quantile, sorted.tensorAlongDimension(i, dimension)));
        }

        return ret;

    }

    @Override
    public int toFlatArray(FlatBufferBuilder builder) {
        int shape = FlatArray.createShapeVector(builder, this.shapeInfoDataBuffer().asLong());
        int buffer = this.isEmpty() ? 0 : FlatArray.createBufferVector(builder, this.data().asBytes());
        val type = this.isEmpty() ? SameDiff.getDataTypeAsByte(Nd4j.dataType()) : SameDiff.getDataTypeAsByte(this.data().dataType());
        int array = FlatArray.createFlatArray(builder, shape, buffer, type, ByteOrder.BE);

        return array;
    }

    /*
     * ------- Sparse methods -------
     */

    @Override
    public DataBuffer getVectorCoordinates() {
        throw new UnsupportedOperationException("Not a sparse ndarray");
    }

    @Override
    public INDArray toDense() {
        return this;
    }

    @Override
    public int nnz() {
        throw new UnsupportedOperationException("Not a sparse ndarray");
    }

    @Override
    public SparseFormat getFormat() {
        return SparseFormat.NONE;
    }

    @Override
    public DataBuffer sparseInfoDataBuffer() {
        throw new UnsupportedOperationException("Not a sparse ndarray");
    }

    @Override
    public int[] flags() {
        throw new UnsupportedOperationException("Not a sparse ndarray");
    }

    @Override
    public int[] hiddenDimensions() {
        throw new UnsupportedOperationException("Not a sparse ndarray");
    }

    @Override
    public int[] sparseOffsets() {
        throw new UnsupportedOperationException("Not a sparse ndarray");
    }

    @Override
    public int underlyingRank() {
        throw new UnsupportedOperationException("Not a sparse ndarray");

    }

    @Override
    public INDArray convertToHalfs() {
        if (data.dataType() == DataBuffer.Type.HALF)
            return this;

        val factory = Nd4j.getNDArrayFactory();
        val buffer = Nd4j.createBuffer(new long[]{this.length()}, DataBuffer.Type.HALF);

        factory.convertDataEx(convertType(data.dataType()), this.data().addressPointer(), DataBuffer.TypeEx.FLOAT16, buffer.addressPointer(), buffer.length());

        return Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInformation);
    }

    @Override
    public INDArray convertToFloats() {
        if (data.dataType() == DataBuffer.Type.FLOAT)
            return this;

        val factory = Nd4j.getNDArrayFactory();
        val buffer = Nd4j.createBuffer(new long[]{this.length()}, DataBuffer.Type.FLOAT);

        factory.convertDataEx(convertType(data.dataType()), this.data().addressPointer(), DataBuffer.TypeEx.FLOAT, buffer.addressPointer(), buffer.length());

        return Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInformation);
    }

    @Override
    public INDArray convertToDoubles() {
        if (data.dataType() == DataBuffer.Type.DOUBLE)
            return this;

        val factory = Nd4j.getNDArrayFactory();
        val buffer = Nd4j.createBuffer(new long[]{this.length()}, DataBuffer.Type.DOUBLE);

        factory.convertDataEx(convertType(data.dataType()), this.data().addressPointer(), DataBuffer.TypeEx.DOUBLE, buffer.addressPointer(), buffer.length());

        return Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInformation);
    }

    protected static DataBuffer.TypeEx convertType(DataBuffer.Type type) {
        if (type == DataBuffer.Type.HALF) {
            return DataBuffer.TypeEx.FLOAT16;
        } else if (type == DataBuffer.Type.FLOAT) {
            return DataBuffer.TypeEx.FLOAT;
        } else if (type == DataBuffer.Type.DOUBLE) {
            return DataBuffer.TypeEx.DOUBLE;

        } else if(type == DataBuffer.Type.INT) {
            return DataBuffer.TypeEx.INT8;
        } else if(type == DataBuffer.Type.LONG) {
            return DataBuffer.TypeEx.INT16;

        } else
            throw new IllegalStateException("Unknown dataType: [" + type + "]");
    }

    /**
     * This method returns true if this INDArray is special case: no-value INDArray
     *
     * @return
     */
    @Override
    public boolean isEmpty() {
        return Shape.isEmpty(jvmShapeInfo.javaShapeInformation);
    }


    @Override
    public long[] shapeInfoJava() {
        return jvmShapeInfo.javaShapeInformation;
    }

    @Override
    public DataBuffer.Type dataType() {
        if (data != null)
            return data.dataType();

        val e = Shape.extras(jvmShapeInfo.javaShapeInformation);

        if (e != 0) {
            val t = ArrayOptionsHelper.dataType(jvmShapeInfo.javaShapeInformation);
            if (t != DataBuffer.Type.UNKNOWN)
                return t;
        }

        return DataBuffer.Type.UNKNOWN;
    }
}
