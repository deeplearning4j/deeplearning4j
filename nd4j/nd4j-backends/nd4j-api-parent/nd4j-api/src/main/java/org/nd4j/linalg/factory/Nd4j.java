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

package org.nd4j.linalg.factory;

import com.google.common.base.Function;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import lombok.NonNull;
import lombok.val;
import lombok.var;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.autodiff.samediff.serde.FlatBuffersMapper;
import org.nd4j.base.Preconditions;
import org.nd4j.config.ND4JEnvironmentVars;
import org.nd4j.config.ND4JSystemProperties;
import org.nd4j.context.Nd4jContext;
import org.nd4j.graph.FlatArray;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.*;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.api.buffer.factory.DefaultDataBufferFactory;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.concurrency.BasicAffinityManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MemoryWorkspaceManager;
import org.nd4j.linalg.api.ndarray.*;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.scalar.ReplaceNans;
import org.nd4j.linalg.api.ops.impl.reduce.TensorMmul;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate;
import org.nd4j.linalg.api.ops.impl.shape.Diag;
import org.nd4j.linalg.api.ops.impl.shape.DiagPart;
import org.nd4j.linalg.api.ops.impl.shape.Stack;
import org.nd4j.linalg.api.ops.impl.transforms.same.OldReverse;
import org.nd4j.linalg.api.ops.random.custom.RandomExponential;
import org.nd4j.linalg.api.ops.random.impl.*;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.factory.DefaultDistributionFactory;
import org.nd4j.linalg.api.rng.distribution.factory.DistributionFactory;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.cache.BasicConstantHandler;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.compression.BasicNDArrayCompressor;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.convolution.ConvolutionInstance;
import org.nd4j.linalg.convolution.DefaultConvolutionInstance;
import org.nd4j.linalg.env.EnvironmentalAction;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.exception.ND4JUnknownDataTypeException;
import org.nd4j.linalg.factory.Nd4jBackend.NoAvailableBackendException;
import org.nd4j.linalg.memory.BasicMemoryManager;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.linalg.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.tools.PropertyParser;
import org.nd4j.versioncheck.VersionCheck;

import java.io.*;
import java.lang.ref.ReferenceQueue;
import java.lang.reflect.Constructor;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

/**
 * Creation of ndarrays via classpath discovery.
 *
 * @author Adam Gibson
 */
public class Nd4j {

    public final static String DATA_BUFFER_OPS = "databufferfactory";
    public final static String CONVOLUTION_OPS = "convops";
    /**@deprecated Use {@link ND4JSystemProperties#DTYPE}*/
    @Deprecated
    public final static String DTYPE = ND4JSystemProperties.DTYPE;
    public final static String BLAS_OPS = "blas.ops";
    public final static String SPARSE_BLAS_OPS = "sparseblas.ops";
    public final static String NATIVE_OPS = "native.ops";
    public final static String ORDER_KEY = "ndarray.order";
    public final static String NDARRAY_FACTORY_CLASS = "ndarrayfactory.class";
    public final static String SPARSE_NDARRAY_FACTORY_CLASS = "sparsendarrayfactory.class";
    public final static String OP_EXECUTIONER = "opexec";
    public final static String OP_FACTORY = "opfactory";
    public final static String DISTRIBUTION = "dist";
    public final static String SHAPEINFO_PROVIDER = "shapeinfoprovider";
    public final static String SPARSEINFO_PROVIDER = "sparseinfoprovider";
    public final static String CONSTANT_PROVIDER = "constantsprovider";
    public final static String AFFINITY_MANAGER = "affinitymanager";
    //disable toString() on compressed arrays for debugging. Should be off by default.
    public final static String COMPRESSION_DEBUG = "compressiondebug";
    public final static String MEMORY_MANAGER = "memorymanager";
    public final static String WORKSPACE_MANAGER = "workspacemanager";
    public final static String RANDOM_PROVIDER = "random";
    /**@deprecated Use {@link ND4JSystemProperties#LOG_INITIALIZATION}*/
    @Deprecated
    public static final String LOG_INIT_ENV_PROPERTY = ND4JSystemProperties.LOG_INITIALIZATION;

    //the datatype used for allocating buffers
    protected static DataType dtype = DataType.FLOAT;
    //the allocation mode for the heap
    public static DataBuffer.AllocationMode alloc = DataBuffer.AllocationMode.HEAP;
    public static char ORDER = 'c';
    public static double EPS_THRESHOLD = 1e-5;
    private static boolean allowsOrder = false;
    public static boolean compressDebug = false;
    public static volatile boolean preventUnpack;
    public static Nd4jBackend backend;
    public static RandomFactory randomFactory;
    private static MemoryWorkspaceManager workspaceManager;
    private static DeallocatorService deallocatorService;
    private static final AtomicInteger numThreads = new AtomicInteger(-1);
    private static AtomicReference<DataType> defaultFloatingPointDataType;

    protected static Class<? extends MemoryWorkspaceManager> workspaceManagerClazz;
    protected static Class<? extends BlasWrapper> blasWrapperClazz;
    protected static Class<? extends BlasWrapper> sparseBlasWrapperClazz;
    protected static Class<? extends NDArrayFactory> ndArrayFactoryClazz;
    protected static Class<? extends NDArrayFactory> sparseNDArrayClazz;
    protected static Class<? extends ConvolutionInstance> convolutionInstanceClazz;
    protected static Class<? extends DataBufferFactory> dataBufferFactoryClazz;
    protected static Class<? extends OpExecutioner> opExecutionerClazz;
    protected static Class<? extends org.nd4j.linalg.api.rng.Random> randomClazz;
    protected static Class<? extends DistributionFactory> distributionFactoryClazz;
    protected static Class<? extends BaseShapeInfoProvider> shapeInfoProviderClazz;
    protected static Class<? extends BaseSparseInfoProvider> sparseInfoProviderClazz;
    protected static Class<? extends BasicConstantHandler> constantProviderClazz;
    protected static Class<? extends BasicAffinityManager> affinityManagerClazz;
    protected static Class<? extends BasicMemoryManager> memoryManagerClazz;

    protected static DataBufferFactory DATA_BUFFER_FACTORY_INSTANCE;
    protected static BlasWrapper BLAS_WRAPPER_INSTANCE;
    protected static BlasWrapper SPARSE_BLAS_WRAPPER_INSTANCE;
    protected static NDArrayFactory INSTANCE;
    protected static NDArrayFactory SPARSE_INSTANCE;
    protected static ConvolutionInstance CONVOLUTION_INSTANCE;
    protected static OpExecutioner OP_EXECUTIONER_INSTANCE;
    protected static DistributionFactory DISTRIBUTION_FACTORY;
    protected static ShapeInfoProvider shapeInfoProvider;
    protected static SparseInfoProvider sparseInfoProvider;
    protected static ConstantHandler constantHandler;
    protected static AffinityManager affinityManager;
    protected static MemoryManager memoryManager;

    protected static AtomicBoolean fallbackMode;

    protected static Properties props = new Properties();
    protected static ReferenceQueue<INDArray> referenceQueue = new ReferenceQueue<>();
    protected static ReferenceQueue<DataBuffer> bufferQueue = new ReferenceQueue<>();

    private final static Logger logger = Logger.getLogger(Nd4j.class.getName());

    protected static final INDArray[] EMPTY_ARRAYS = new INDArray[DataType.values().length];

    static {
        fallbackMode = new AtomicBoolean(false);
        Nd4j nd4j = new Nd4j();
        nd4j.initContext();
    }

    public enum PadMode {
        CONSTANT, EDGE, LINEAR_RAMP, MAXIMUM, MEAN, MEDIAN, MINIMUM, REFLECT, SYMMETRIC, WRAP

    }

    /**
     * @see #pad(INDArray, int[][], PadMode)
     */
    public static INDArray pad(INDArray toPad, int[][] padWidth, PadMode padMode) {
        return pad(toPad, padWidth, ArrayUtil.zerosMatrix(toPad.shape()), padMode);
    }

    /**
     * Pad the given ndarray to the size along each dimension
     * @param toPad the ndarray to pad
     * @param padWidth the width to pad along each dimension
     * @param constantValues the values to append for each dimension
     * @param padMode the mode to pad in
     * @return the padded ndarray
     * based on the specified mode
     */
    public static INDArray pad(INDArray toPad, int[][] padWidth, List<double[]> constantValues, PadMode padMode) {
        switch (padMode) {
            case CONSTANT:
                if (padWidth.length < toPad.rank())
                    throw new IllegalArgumentException("Please specify a pad width for each dimension");

                List<int[]> sizes = new ArrayList<>();
                for (int i = 0; i < toPad.rank(); i++) {
                    sizes.add(padWidth[i]);
                }

                INDArray ret = toPad;
                for (int i = 0; i < toPad.rank(); i++) {
                    int[] pad = sizes.get(i);
                    double[] constant = constantValues.get(i);
                    int padBefore = pad[0];
                    int padAfter = pad[1];
                    if (constant.length < 2) {
                        double val = constant[0];
                        constant = new double[2];
                        constant[0] = val;
                        constant[1] = val;
                    }

                    double beforeVal = constant[0];
                    double afterVal = constant[1];
                    ret = Nd4j.prepend(ret, padBefore, beforeVal, i);
                    ret = Nd4j.append(ret, padAfter, afterVal, i);

                }

                return ret;

            default:
                throw new UnsupportedOperationException();

        }
    }

    /**
     * @see #pad(INDArray, int[][], PadMode)
     */
    public static INDArray pad(INDArray toPad, int[] padWidth, List<double[]> constantValues, PadMode padMode) {
        switch (padMode) {
            case CONSTANT:
                if (padWidth.length < toPad.rank())
                    throw new IllegalArgumentException("Please specify a pad width for each dimension");

                toPad = Nd4j.stripOnes(toPad);

                List<int[]> sizes = new ArrayList<>();
                for (int i = 0; i < toPad.rank(); i++) {
                    sizes.add(padWidth);
                }

                INDArray ret = toPad;
                //TODO: Remove duplicate code.
                for (int i = 0; i < toPad.rank(); i++) {
                    int[] pad = sizes.get(i);
                    double[] constant = constantValues.get(i);
                    int padBefore = pad[0];
                    int padAfter = pad[1];
                    if (constant.length < 2) {
                        double val = constant[0];
                        constant = new double[2];
                        constant[0] = val;
                        constant[1] = val;
                    }

                    double beforeVal = constant[0];
                    double afterVal = constant[1];
                    ret = Nd4j.prepend(ret, padBefore, beforeVal, i);
                    ret = Nd4j.append(ret, padAfter, afterVal, i);

                }

                return ret;

            default:
                throw new UnsupportedOperationException();

        }
    }

    /**
     * @see #pad(INDArray, int[][], PadMode)
     */
    public static INDArray pad(INDArray toPad, int[] padWidth, PadMode padMode) {
        return pad(toPad, padWidth, ArrayUtil.zerosMatrix(padWidth), padMode);
    }

    /**
     * Append the given array with the specified value size along a particular axis.
     * The prepend method has the same signature and prepends the given array.
     * @param arr the array to append to
     * @param padAmount the pad amount of the array to be returned
     * @param val the value to append
     * @param axis the axis to append to
     * @return the newly created array
     */
    public static INDArray append(INDArray arr, int padAmount, double val, int axis) {
        return appendImpl(arr, padAmount, val, axis, true);
    }

    /**
     * @see #append(INDArray, int, double, int)
     */
    public static INDArray prepend(INDArray arr, int padAmount, double val, int axis) {
        return appendImpl(arr, padAmount, val, axis, false);
    }

    /**
     * Append / Prepend shared implementation.
     * @param appendFlag flag to determine Append / Prepend.
     * @see #append(INDArray, int, double, int)
     */
    private static INDArray appendImpl(INDArray arr, int padAmount, double val, int axis, boolean appendFlag){
        if (padAmount == 0)
            return arr;
        long[] paShape = ArrayUtil.copy(arr.shape());
        if (axis < 0)
            axis = axis + arr.shape().length;
        paShape[axis] = padAmount;
        INDArray concatArray = Nd4j.valueArrayOf(paShape, val, arr.dataType());
        return appendFlag ? Nd4j.concat(axis, arr, concatArray) : Nd4j.concat(axis, concatArray, arr);
    }
    
    /**
     * Expand the array dimensions.
     * This is equivalent to
     * adding a new axis dimension
     * @param input the input array
     * @param dimension the dimension to add the
     *                  new axis at
     * @return the array with the new axis dimension
     */
    public static INDArray expandDims(INDArray input, int dimension) {
        if (dimension < 0)
            dimension += input.rank();
        long[] shape = input.shape();
        long[] indexes = new long[input.rank() + 1];
        for (int i = 0; i < indexes.length; i++)
            indexes[i] = i < dimension ? shape[i] : i == dimension ? 1 : shape[i - 1];
        return input.reshape(input.ordering(), indexes);
    }

    /**
     * Squeeze : removes a dimension of size 1
     * @param input the input array
     * @param dimension the dimension to remove
     * @return the array with dimension removed
     */
    public static INDArray squeeze(INDArray input, int dimension) {
        if (dimension < 0){
            dimension += input.rank();
        }
        long[] shape = input.shape();
        Preconditions.checkState(shape[dimension] == 1, String.format("Squeeze: Only dimension of size 1 can be squeezed. " +
                "Attempted to squeeze dimension %d of array with shape %s (size %d).", dimension, ArrayUtils.toString(shape), shape[dimension]));

        long[] newShape = ArrayUtil.removeIndex(shape, dimension);
        return input.reshape(input.ordering(), newShape);
    }

    /**
     * Backend specific:
     * Returns whether specifying the order
     * for the blas impl is allowed (cblas)
     * @return true if the blas impl
     * can support specifying array order
     */
    public static boolean allowsSpecifyOrdering() {
        return allowsOrder;
    }

    /**
     * In place shuffle of an ndarray
     * along a specified set of dimensions
     * @param toShuffle the ndarray to shuffle
     * @param random the random to use
     * @param dimension the dimension to do the shuffle
     * @return
     */
    public static void shuffle(INDArray toShuffle, Random random, @NonNull int... dimension) {
        INSTANCE.shuffle(toShuffle, random, dimension);
    }

    /**
     * In place shuffle of an ndarray
     * along a specified set of dimensions
     * @param toShuffle the ndarray to shuffle
     * @param dimension the dimension to do the shuffle
     * @return
     */
    public static void shuffle(INDArray toShuffle, @NonNull int... dimension) {
        //shuffle(toShuffle, new Random(), dimension);
        INSTANCE.shuffle(toShuffle, new Random(), dimension);
    }

    /**
     * Symmetric in place shuffle of an ndarray
     * along a specified set of dimensions
     * @param toShuffle the ndarray to shuffle
     * @param dimension the dimension to do the shuffle
     * @return
     */
    public static void shuffle(Collection<INDArray> toShuffle, @NonNull int... dimension) {
        //shuffle(toShuffle, new Random(), dimension);
        INSTANCE.shuffle(toShuffle, new Random(), dimension);
    }

    /**
     * Symmetric in place shuffle of an ndarray
     * along a specified set of dimensions
     * @param toShuffle the ndarray to shuffle
     * @param dimension the dimension to do the shuffle
     * @return
     */
    public static void shuffle(Collection<INDArray> toShuffle, Random rnd, @NonNull int... dimension) {
        //shuffle(toShuffle, new Random(), dimension);
        INSTANCE.shuffle(toShuffle, rnd, dimension);
    }

    /**
     * Symmetric in place shuffle of an ndarray
     * along a variable dimensions
     *
     * @param toShuffle the ndarray to shuffle
     * @param dimensions the dimension to do the shuffle. Please note - order matters here.
     * @return
     */
    public static void shuffle(List<INDArray> toShuffle, Random rnd, List<int[]> dimensions) {

        INSTANCE.shuffle(toShuffle, rnd, dimensions);
    }

    /**
     * The reference queue used for cleaning up
     * ndarrays
     *
     * @return the reference queue for cleaning up ndarrays
     */
    public static ReferenceQueue<INDArray> refQueue() {
        return referenceQueue;
    }

    /**
     * The reference queue used for cleaning up
     * databuffers
     *
     * @return the reference queue for cleaning up databuffers
     */
    public static ReferenceQueue<DataBuffer> bufferRefQueue() {
        return bufferQueue;
    }

    /**
     * Get the primary distributions
     * factory
     *
     * @return the primary distributions
     */
    public static DistributionFactory getDistributions() {
        return DISTRIBUTION_FACTORY;
    }

    // TODO: unused, uncommented method.
    public static void setNdArrayFactoryClazz(Class<? extends NDArrayFactory> clazz) {
        ndArrayFactoryClazz = clazz;
    }

    // TODO: unused, uncommented method.
    public static void setSparseNDArrayClazz(Class<? extends NDArrayFactory> clazz) {
        sparseNDArrayClazz = clazz;
    }

    /**
     * Get the current random generator
     *
     * @return the current random generator
     */
    public static org.nd4j.linalg.api.rng.Random getRandom() {
        return randomFactory.getRandom();
    }

    /**
     * This method returns RandomFactory instance
     *
     * @return
     */
    public static RandomFactory getRandomFactory() {
        return randomFactory;
    }

    /**
     * Get the convolution singleton
     *
     * @return the convolution singleton
     */
    public static ConvolutionInstance getConvolution() {
        return CONVOLUTION_INSTANCE;
    }

    /**
     * Set a convolution instance
     *
     * @param convolutionInstance
     */
    public static void setConvolution(ConvolutionInstance convolutionInstance) {
        if (convolutionInstance == null)
            throw new IllegalArgumentException("No null instances allowed");
        CONVOLUTION_INSTANCE = convolutionInstance;
    }

    /**
     * Returns the shape of the ndarray
     * @param arr the array to get the shape of
     * @return the shape of tihs ndarray
     */
    public static long[] shape(INDArray arr) {
        return arr.shape();
    }

    /**
     * Create an ndarray based on the given data
     * @param sliceShape the shape of each slice
     * @param arrays the arrays of data to create
     * @return the ndarray of the specified shape where
     * number of slices is equal to array length and each
     * slice is the specified shape
     */
    public static INDArray create(int[] sliceShape, float[]... arrays) {
        //TODO: Remove duplicate code.
        int slices = arrays.length;
        INDArray ret = Nd4j.create(ArrayUtil.combine(new int[] {slices}, sliceShape));
        for (int i = 0; i < ret.slices(); i++)
            ret.putSlice(i, Nd4j.create(arrays[i]).reshape(ArrayUtil.toLongArray(sliceShape)));
        return ret;
    }

    /**
     * @see #create(LongShapeDescriptor)
     */
    public static INDArray create(LongShapeDescriptor descriptor) {
        return create(descriptor, true);
    }

    /**
     * Create an ndarray based on the given description,
     * @param descriptor object with data for array creation.
     * @param initialize true/false creates initialized/uninitialized array.
     * @return the ndarray of the specified description.
     */
    public static INDArray create(LongShapeDescriptor descriptor, boolean initialize) {
        if(descriptor.isEmpty() && descriptor.rank() == 0){
            return Nd4j.empty(descriptor.dataType());
        }
        if (initialize)
            return create(descriptor.dataType(), descriptor.getShape(), descriptor.getStride(), descriptor.getOrder());
        else
            return createUninitialized(descriptor.dataType(), descriptor.getShape(), descriptor.getOrder());
    }

    /**
     * @see #create(int[], float[]...)
     */
    public static INDArray create(int[] sliceShape, double[]... arrays) {
        int slices = arrays.length;
        INDArray ret = Nd4j.create(ArrayUtil.combine(new int[] {slices}, sliceShape));
        for (int i = 0; i < ret.slices(); i++)
            ret.putSlice(i, Nd4j.create(arrays[i]).reshape(ArrayUtil.toLongArray(sliceShape)));
        return ret;
    }

    /**
     * Get the operation executioner instance.
     *
     * @return the operation executioner instance.
     */
    public static OpExecutioner getExecutioner() {
        return OP_EXECUTIONER_INSTANCE;
    }

    /**
     * Get the data buffer factory instance.
     *
     * @return the data buffer factory instance.
     */
    public static DataBufferFactory getDataBufferFactory() {
        return DATA_BUFFER_FACTORY_INSTANCE;
    }

    /**
     * Given a sequence of Iterators over a transform of matrices, fill in all of
     * the matrices with the entries in the theta vector.  Errors are
     * thrown if the theta vector does not exactly fill the matrices.
     *
     * TODO: unused method.
     */
    public static void setParams(INDArray theta, Collection<INDArray>... matrices) {
        int index = 0;
        for (Collection<INDArray> matrixCollection : matrices) {
            for (INDArray matrix : matrixCollection) {
                INDArray linear = matrix.reshape(-1);
                for (int i = 0; i < matrix.length(); i++) {
                    linear.putScalar(i, theta.getDouble(index));
                    index++;
                }
            }
        }

        if (index != theta.length()) {
            throw new AssertionError("Did not entirely use the theta vector");
        }
    }

    /**
     *  Roll the specified axis backwards,
     *  until it lies in a given position.
     *  Starting ends up being zero.
     *  See numpy's rollaxis
     * @param a the array to roll
     * @param axis the axis to roll backwards
     * @return the rolled ndarray
     */
    public static INDArray rollAxis(INDArray a, int axis) {
        return rollAxis(a, axis, 0);
    }

    /**
     * Get the maximum (argmax) or minimum (argmin) values for a dimension.
     * @param arr input array.
     * @param dimension the dimension along which to get the maximum
     * @return array of maximum values.
     */
    public static INDArray argMax(INDArray arr, @NonNull int... dimension) {
        IMax imax = new IMax(arr, dimension);
        return Nd4j.getExecutioner().exec(imax);
    }

    /**
     * @see #argMax(INDArray, int...)
     */
    public static INDArray argMin(INDArray arr, @NonNull int... dimension) {
        IMin imin = new IMin(arr, dimension);
        return Nd4j.getExecutioner().exec(imin);
    }

    /**
     *  Roll the specified axis backwards,
     *  until it lies in a given position.
     *  See numpy's rollaxis
     * @param a the array to roll
     * @param axis the axis to roll backwards
     * @param start the starting point
     * @return the rolled ndarray
     */
    public static INDArray rollAxis(INDArray a, int axis, int start) {
        if (axis < 0)
            axis += a.rank();
        if (start < 0)
            start += a.rank();
        if (axis == start)
            return a;
        if (axis < start)
            start--;
        if (!(axis >= 0 && axis < a.rank()))
            throw new IllegalArgumentException("Axis must be >= 0 && < start");
        if (!(start >= 0 && axis < a.rank() + 1))
            throw new IllegalArgumentException("Axis must be >= 0 && < start");

        List<Integer> range = new ArrayList<>(Ints.asList(ArrayUtil.range(0, a.rank())));
        range.remove(axis);
        range.add(start, axis);
        int[] newRange = Ints.toArray(range);
        return a.permute(newRange);

    }

    /**
     * Tensor matrix multiplication.
     * Both tensors must be the same rank
     *
     * @param a the left tensor
     * @param b the  right tensor
     * @param result the result array
     * @param axes the axes for each array to do matrix multiply along
     * @return
     */
    public static INDArray tensorMmul(INDArray a, INDArray b,INDArray result, int[][] axes) {
        int validationLength = Math.min(axes[0].length, axes[1].length);
        for (int i = 0; i < validationLength; i++) {
            if (a.size(axes[0][i]) != b.size(axes[1][i]))
                throw new IllegalArgumentException("Size of the given axes at each dimension must be the same size.");
            if (axes[0][i] < 0)
                axes[0][i] += a.rank();
            if (axes[1][i] < 0)
                axes[1][i] += b.rank();

        }

        List<Integer> listA = new ArrayList<>();
        for (int i = 0; i < a.rank(); i++) {
            if (!Ints.contains(axes[0], i))
                listA.add(i);
        }

        int[] newAxesA = Ints.concat(Ints.toArray(listA), axes[0]);

        List<Integer> listB = new ArrayList<>();
        for (int i = 0; i < b.rank(); i++) {
            if (!Ints.contains(axes[1], i))
                listB.add(i);
        }

        int[] newAxesB = Ints.concat(axes[1], Ints.toArray(listB));

        int n2 = 1;
        int aLength = Math.min(a.rank(), axes[0].length);
        for (int i = 0; i < aLength; i++) {
            n2 *= a.size(axes[0][i]);
        }

        //if listA and listB are empty these donot initialize.
        //so initializing with {1} which will then get overriden if not empty
        long[] newShapeA = {-1, n2};
        //TODO: remove duplicate code.
        long[] oldShapeA;
        if (listA.size() == 0) {
            oldShapeA = new long[] {1};
        } else {
            oldShapeA = Longs.toArray(listA);
            for (int i = 0; i < oldShapeA.length; i++)
                oldShapeA[i] = a.size((int) oldShapeA[i]);
        }

        int n3 = 1;
        int bNax = Math.min(b.rank(), axes[1].length);
        for (int i = 0; i < bNax; i++) {
            n3 *= b.size(axes[1][i]);
        }

        long[] newShapeB = {n3, -1};
        long[] oldShapeB;
        if (listB.size() == 0) {
            oldShapeB = new long[] {1};
        } else {
            oldShapeB = Longs.toArray(listB);
            for (int i = 0; i < oldShapeB.length; i++)
                oldShapeB[i] = b.size((int) oldShapeB[i]);
        }

        INDArray at = a.permute(newAxesA).reshape(newShapeA);
        INDArray bt = b.permute(newAxesB).reshape(newShapeB);
        INDArray ret = at.mmul(bt,result);

        long[] aPlusB = Longs.concat(oldShapeA, oldShapeB);
        return ret.reshape(aPlusB);
    }

    /**
     * Tensor matrix multiplication.
     * Both tensors must be the same rank
     *
     * @param a the left tensor
     * @param b the  right tensor
     * @param axes the axes for each array to do matrix multiply along
     * @return
     */
    public static INDArray tensorMmul(INDArray a, INDArray b, int[][] axes) {
        CustomOp op = DynamicCustomOp.builder("tensordot")
                .addInputs(a, b)
                .addIntegerArguments(axes[0].length)
                .addIntegerArguments(axes[0])
                .addIntegerArguments(axes[1].length)
                .addIntegerArguments(axes[1])
                .build();

        List<LongShapeDescriptor> l = op.calculateOutputShape();
        INDArray out = Nd4j.create(l.get(0).asDataType(a.dataType()));
        op.addOutputArgument(out);
        Nd4j.exec(op);

        return out;
    }

    /**
     * matrix multiply: implements op(a)*op(b)
     *
     * where op(x) means transpose x (or not) depending on
     * setting of arguments transposea and transposeb.<br>
     * so gemm(a,b,false,false) == a.mmul(b), gemm(a,b,true,false) == a.transpose().mmul(b) etc.
     * @param a first matrix
     * @param b second matrix
     * @param transposeA if true: transpose matrix a before mmul
     * @param transposeB if true: transpose matrix b before mmul
     * @return result
     */
    public static INDArray gemm(INDArray a,
                                INDArray b,
                                boolean transposeA,
                                boolean transposeB) {
        long cRows = (transposeA ? a.columns() : a.rows());
        long cCols = (transposeB ? b.rows() : b.columns());
        INDArray c = Nd4j.createUninitialized(a.dataType(), new long[] {cRows, cCols}, 'f');
        return gemm(a, b, c, transposeA, transposeB, 1.0, 0.0);
    }

    /**
     *  Matrix multiply: Implements c = alpha*op(a)*op(b) + beta*c where op(X) means transpose X (or not)
     * depending on setting of arguments transposeA and transposeB.<br>
     * Note that matrix c MUST be fortran order, have zero offset and have c.data().length == c.length().
     * i.e., the result array must not be a view. An exception will be thrown otherwise.<br>
     * (Note: some views are allowed, if and only if they have f order and are contiguous in the buffer other than an
     * offset. Put another way, they must be f order and have strides identical to a non-view/default array of the same shape)<br>
     * Don't use this unless you know about level 3 blas and NDArray storage orders.
     * @param a First matrix
     * @param b Second matrix
     * @param c result matrix. Used in calculation (assuming beta != 0) and result is stored in this. f order, and not a view only
     * @param transposeA if true: transpose matrix a before mmul
     * @param transposeB if true: transpose matrix b before mmul
     * @return result, i.e., matrix c is returned for convenience
     */
    public static INDArray gemm(INDArray a,
                                INDArray b,
                                INDArray c,
                                boolean transposeA,
                                boolean transposeB,
                                double alpha,
                                double beta) {
        //Note: some views have non-zero offset but 'default' strides (these are OK). And a 'c' order vector such as [10,1] is OK - same buffer as an 'f' order vector with same shape
        Preconditions.checkState(c.length() == 1 || c.ordering() == 'f' && Shape.hasDefaultStridesForShape(c) ||
                        c.isVectorOrScalar() && c.elementWiseStride() == 1,
                "C (result) array is not F order or is a view. Nd4j.gemm requires the result array to be F order " +
                        "and not a view. C (result) array: [%ndSInfo]", c);
        getBlasWrapper().level3().gemm(a, b, c, transposeA, transposeB, alpha, beta);
        return c;
    }

    /**
     * Matrix multiplication/dot product
     *
     * Depending on inputs dimensionality output result might be different.
     * matrix x matrix = BLAS gemm
     * vector x matrix = BLAS gemm
     * vector x vector = BLAS dot
     * vector x scalar = element-wise mul
     * scalar x vector = element-wise mul
     * tensor x tensor = matrix multiplication using the last two dimensions
     *
     * Transpose operations only available where applicable. In the
     * tensor x tensor case it will be applied only to the last two dimensions.
     *
     * @param a First tensor
     * @param b Second tensor
     * @param result result matrix.
     * @param transposeA if true: transpose matrix a before mmul
     * @param transposeB if true: transpose matrix b before mmul
     * @param transposeResult if true: result matrix will be transposed
     * @return result, i.e., result matrix is returned for convenience
     */
    public static INDArray matmul(INDArray a, INDArray b, INDArray result, boolean transposeA, boolean transposeB, boolean transposeResult){
        final Mmul op = new Mmul(a, b, result,
                MMulTranspose.builder()
                        .transposeA(transposeA)
                        .transposeB(transposeB)
                        .transposeResult(transposeResult).build());
        return exec(op)[0];
    }

    /**
     * Matrix multiplication/dot product
     *
     * @see #matmul(INDArray, INDArray, INDArray, boolean, boolean, boolean)
     */
    public static INDArray matmul(INDArray a, INDArray b, INDArray result){
        final Mmul op = new Mmul(a, b, result, null);
        return exec(op)[0];
    }

    /**
     * Matrix multiplication/dot product
     *
     * @see #matmul(INDArray, INDArray, INDArray, boolean, boolean, boolean)
     */
    public static INDArray matmul(INDArray a, INDArray b, boolean transposeA, boolean transposeB, boolean transposeResult){
       return matmul(a, b, null, transposeA, transposeB, transposeResult);
    }

    /**
     * Matrix multiplication/dot product
     *
     * @see #matmul(INDArray, INDArray, INDArray, boolean, boolean, boolean)
     */
    public static INDArray matmul(INDArray a, INDArray b){
        return matmul(a,b, null);
    }

    /**
     * Given a sequence of Iterators over a transform of matrices, fill in all of
     * the matrices with the entries in the theta vector.  Errors are
     * thrown if the theta vector does not exactly fill the matrices.
     * TODO: unused method.
     */
    public static void setParams(INDArray theta, Iterator<? extends INDArray>... matrices) {
        int index = 0;
        for (Iterator<? extends INDArray> matrixIterator : matrices) {
            while (matrixIterator.hasNext()) {
                INDArray matrix = matrixIterator.next().reshape(-1);
                for (int i = 0; i < matrix.length(); i++) {
                    matrix.putScalar(i, theta.getDouble(index));
                    index++;
                }
            }
        }

        if (index != theta.length()) {
            throw new AssertionError("Did not entirely use the theta vector");
        }
    }

    /**
     * The factory used for creating ndarrays
     *
     * @return the factory instance used for creating ndarrays
     */
    public static NDArrayFactory factory() {
        return INSTANCE;
    }

    /**
     * The factory used for creating sparse arrays.
     * @return the factory used for creating sparse arrays.
     */
    public static NDArrayFactory sparseFactory() {
        return SPARSE_INSTANCE;
    }

    // TODO: We forward the following methods with a default dimension argument. Doc should say something about that.
    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#cumsum(int)
     */
    public static INDArray cumsum(INDArray compute) {
        return compute.cumsum(Integer.MAX_VALUE);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#max(int...)
     */
    public static INDArray max(INDArray compute) {
        return compute.max(Integer.MAX_VALUE);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#min(int...)
     */
    public static INDArray min(INDArray compute) {
        return compute.min(Integer.MAX_VALUE);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#prod(int...)
     */
    public static INDArray prod(INDArray compute) {
        return compute.prod(Integer.MAX_VALUE);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#normmax(int...)
     */
    public static INDArray normmax(INDArray compute) {
        return compute.normmax(Integer.MAX_VALUE);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#norm2(int...)
     */
    public static INDArray norm2(INDArray compute) {
        return compute.norm2(Integer.MAX_VALUE);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#norm1(int...)
     */
    public static INDArray norm1(INDArray compute) {
        return compute.norm1(Integer.MAX_VALUE);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#std(int...)
     */
    public static INDArray std(INDArray compute) {
        return compute.std(Integer.MAX_VALUE);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#var(int...)
     */
    public static INDArray var(INDArray compute) {
        return compute.var(Integer.MAX_VALUE);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#sum(int...)
     */
    public static INDArray sum(INDArray compute) {
        return compute.sum(Integer.MAX_VALUE);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#mean(int...)
     */
    public static INDArray mean(INDArray compute) {
        return compute.mean(Integer.MAX_VALUE);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#cumsum(int)
     */
    public static INDArray cumsum(INDArray compute, int dimension) {
        return compute.cumsum(dimension);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#max(int...)
     */
    public static INDArray max(INDArray compute, int dimension) {
        return compute.max(dimension);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#min(int...)
     */
    public static INDArray min(INDArray compute, int dimension) {
        return compute.min(dimension);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#prod(int...)
     */
    public static INDArray prod(INDArray compute, int dimension) {
        return compute.prod(dimension);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#normmax(int...)
     */
    public static INDArray normmax(INDArray compute, int dimension) {
        return compute.normmax(dimension);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#norm2(int...)
     */
    public static INDArray norm2(INDArray compute, int dimension) {
        return compute.norm2(dimension);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#norm1(int...)
     */
    public static INDArray norm1(INDArray compute, int dimension) {
        return compute.norm1(dimension);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#std(int...)
     */
    public static INDArray std(INDArray compute, int dimension) {
        return compute.std(dimension);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#var(int...)
     */
    public static INDArray var(INDArray compute, int dimension) {
        return compute.var(dimension);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#sum(int...)
     */
    public static INDArray sum(INDArray compute, int dimension) {
        return compute.sum(dimension);
    }

    /**
     * @see org.nd4j.linalg.api.ndarray.INDArray#mean(int...)
     */
    public static INDArray mean(INDArray compute, int dimension) {
        return compute.mean(dimension);
    }

    /**
     * Create a view of a data buffer
     * Leverages the underlying storage of the buffer with a new view
     *
     * @param underlyingBuffer the underlying buffer
     * @param offset the offset for the view
     * @return the new view of the data buffer
     */
    public static DataBuffer createBuffer(DataBuffer underlyingBuffer, long offset, long length) {
        return DATA_BUFFER_FACTORY_INSTANCE.create(underlyingBuffer, offset, length);
    }

    /**
     * Create a buffer equal of length prod(shape)
     *
     * @param shape the shape of the buffer to create
     * @param type  the opType to create
     * @return the created buffer
     */
    public static DataBuffer createBuffer(int[] shape, DataType type, long offset) {
        int length = ArrayUtil.prod(shape);
        return type == DataType.DOUBLE ? createBuffer(new double[length], offset)
                : createBuffer(new float[length], offset);
    }

    /**
     * Creates a buffer of the specified opType and length with the given byte buffer.
     *
     * This will wrap the buffer as a reference (no copy)
     * if the allocation opType is the same.
     * @param buffer the buffer to create from
     * @param type the opType of buffer to create
     * @param length the length of the buffer
     * @return the created buffer
     */
    public static DataBuffer createBuffer(ByteBuffer buffer, DataType type, int length, long offset) {
        switch (type) {
            case INT:
                return DATA_BUFFER_FACTORY_INSTANCE.createInt(offset, buffer, length);
            case DOUBLE:
                return DATA_BUFFER_FACTORY_INSTANCE.createDouble(offset, buffer, length);
            case FLOAT:
                return DATA_BUFFER_FACTORY_INSTANCE.createFloat(offset, buffer, length);
            default:
                throw new IllegalArgumentException("Illegal opType " + type);
        }
    }

    /**
     * Create a buffer based on the data opType
     *
     * @param data the data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBuffer(byte[] data, int length, long offset) {
        DataBuffer ret;
        if (dataType() == DataType.DOUBLE)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createDouble(offset, data, length);
        else
            ret = DATA_BUFFER_FACTORY_INSTANCE.createFloat(offset, data, length);
        return ret;
    }

    /**
     * Create a buffer equal of length prod(shape)
     *
     * @param data the shape of the buffer to create
     * @return the created buffer
     */
    public static DataBuffer createBuffer(int[] data, long offset) {
        DataBuffer ret = createTypedBuffer(data, DataType.INT, Nd4j.getMemoryManager().getCurrentWorkspace());
        return ret;
    }

    /**
     * Creates a buffer of the specified length based on the data opType
     *
     * @param length the length of te buffer
     * @return the buffer to create
     */
    public static DataBuffer createBuffer(int length, long offset) {
        DataBuffer ret;
        if (dataType() == DataType.FLOAT)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createFloat(offset, length);
        else if (dataType() == DataType.INT)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createInt(offset, length);
        else if (dataType() == DataType.DOUBLE)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createDouble(offset, length);
        else if (dataType() == DataType.HALF)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createHalf(offset, length);
        else
            ret = null;

        return ret;
    }

    protected static Indexer getIndexerByType(Pointer pointer, DataType dataType) {
        switch (dataType) {
            case UINT64:
            case LONG:
                return LongIndexer.create((LongPointer) pointer);
            case UINT32:
            case INT:
                return IntIndexer.create((IntPointer) pointer);
            case UINT16:
                return UShortIndexer.create((ShortPointer) pointer);
            case SHORT:
                return ShortIndexer.create((ShortPointer) pointer);
            case BYTE:
                return ByteIndexer.create((BytePointer) pointer);
            case UBYTE:
                return UByteIndexer.create((BytePointer) pointer);
            case BOOL:
                return BooleanIndexer.create((BooleanPointer) pointer);
            case FLOAT:
                return FloatIndexer.create((FloatPointer) pointer);
            case BFLOAT16:
                return Bfloat16Indexer.create((ShortPointer) pointer);
            case HALF:
                return HalfIndexer.create((ShortPointer) pointer);
            case DOUBLE:
                return DoubleIndexer.create((DoublePointer) pointer);
            default:
                throw new UnsupportedOperationException();
        }
    }

    /**
     * Creates a buffer of the specified type and length with the given pointer.
     *
     * @param pointer pointer to data to create from.
     * @param length the length of the buffer
     * @param dataType the opType of buffer to create,
     * @return the created buffer
     */
    public static DataBuffer createBuffer(@NonNull Pointer pointer, long length, @NonNull DataType dataType) {
        Pointer nPointer = null;
        //TODO: remove dupplicate code.
        switch (dataType) {
            case LONG:
                nPointer =  new LongPointer(pointer);
                break;
            case INT:
                nPointer =  new IntPointer(pointer);
                break;
            case SHORT:
                nPointer =  new ShortPointer(pointer);
                break;
            case BYTE:
                nPointer =  new BytePointer(pointer);
                break;
            case UBYTE:
                nPointer =  new BytePointer(pointer);
                break;
            case BOOL:
                nPointer =  new BooleanPointer(pointer);
                break;
            case FLOAT:
                nPointer =  new FloatPointer(pointer);
                break;
            case HALF:
                nPointer =  new ShortPointer(pointer);
                break;
            case DOUBLE:
                nPointer =  new DoublePointer(pointer);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType);
        }

        return DATA_BUFFER_FACTORY_INSTANCE.create(nPointer, dataType, length, getIndexerByType(nPointer, dataType));
    }

    /**
     * Creates a buffer of the specified type and length with the given pointer at the specified device.
     * (This method is relevant only for a CUDA backend).
     *
     * @param pointer        pointer to data to create from.
     * @param devicePointer  pointer to device to create in (only implemented in the CUDA backend)
     * @param length         the length of the buffer
     * @param dataType       the opType of buffer to create,
     * @return               the created buffer
     */
    public static DataBuffer createBuffer(@NonNull Pointer pointer, @NonNull Pointer devicePointer, long length, @NonNull DataType dataType) {
        Pointer nPointer = null;
        switch (dataType) {
            case UINT64:
            case LONG:
                nPointer =  new LongPointer(pointer);
                break;
            case UINT32:
            case INT:
                nPointer =  new IntPointer(pointer);
                break;
            case UINT16:
            case SHORT:
                nPointer =  new ShortPointer(pointer);
                break;
            case BYTE:
                nPointer =  new BytePointer(pointer);
                break;
            case UBYTE:
                nPointer =  new BytePointer(pointer);
                break;
            case BOOL:
                nPointer =  new BooleanPointer(pointer);
                break;
            case BFLOAT16:
            case HALF:
                nPointer =  new ShortPointer(pointer);
                break;
            case FLOAT:
                nPointer =  new FloatPointer(pointer);
                break;
            case DOUBLE:
                nPointer =  new DoublePointer(pointer);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType);
        }

        return DATA_BUFFER_FACTORY_INSTANCE.create(nPointer, devicePointer, dataType, length, getIndexerByType(nPointer, dataType));
    }

    /**
     * Create a buffer based on the data opType
     *
     * @param data the data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBuffer(float[] data, long offset) {
        val ndata = Arrays.copyOfRange(data, (int) offset, data.length);
        DataBuffer ret = createTypedBuffer(ndata, DataType.FLOAT, Nd4j.getMemoryManager().getCurrentWorkspace());
        return ret;
    }

    /**
     * Create a buffer based on the data opType
     *
     * @param data the data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBuffer(double[] data, long offset) {
        val ndata = Arrays.copyOfRange(data, (int) offset, data.length);
        DataBuffer ret = createTypedBuffer(ndata, DataType.DOUBLE, Nd4j.getMemoryManager().getCurrentWorkspace());
        return ret;
    }

    /**
     * Create a buffer equal of length prod(shape)
     *
     * @param shape the shape of the buffer to create
     * @param type  the opType to create
     * @return the created buffer
     */
    public static DataBuffer createBuffer(int[] shape, DataType type) {
        long length = ArrayUtil.prodLong(shape);

        if (type == DataType.INT)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createInt(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createInt(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (type == DataType.LONG)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createLong(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createLong(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (type == DataType.HALF)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createHalf(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createHalf(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (type == DataType.DOUBLE)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createDouble(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createFloat(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Create a buffer equal of length prod(shape)
     *
     * @param shape the shape of the buffer to create
     * @param type the opType to create
     * @return
     */
    public static DataBuffer createBufferDetached(int[] shape, DataType type) {
        long length = ArrayUtil.prodLong(shape);
        switch (type){
            case DOUBLE:
                return DATA_BUFFER_FACTORY_INSTANCE.createDouble(length);
            case FLOAT:
                return DATA_BUFFER_FACTORY_INSTANCE.createFloat(length);
            case HALF:
                return DATA_BUFFER_FACTORY_INSTANCE.createHalf(length);
            case BFLOAT16:
                return DATA_BUFFER_FACTORY_INSTANCE.createBFloat16(length);
            case UINT64:
                return DATA_BUFFER_FACTORY_INSTANCE.createULong(length);
            case LONG:
                return DATA_BUFFER_FACTORY_INSTANCE.createLong(length);
            case UINT32:
                return DATA_BUFFER_FACTORY_INSTANCE.createUInt(length);
            case INT:
                return DATA_BUFFER_FACTORY_INSTANCE.createInt(length);
            case UINT16:
                return DATA_BUFFER_FACTORY_INSTANCE.createUShort(length);
            case SHORT:
                return DATA_BUFFER_FACTORY_INSTANCE.createShort(length);
            case UBYTE:
                return DATA_BUFFER_FACTORY_INSTANCE.createUByte(length);
            case BYTE:
                return DATA_BUFFER_FACTORY_INSTANCE.createByte(length);
            case BOOL:
                return DATA_BUFFER_FACTORY_INSTANCE.createBool(length);
            case UTF8:
            case COMPRESSED:
            case UNKNOWN:
            default:
                throw new UnsupportedOperationException("Cannot create type: " + type);
        }
    }

    /**
     * @see #createBuffer(int[], DataType)
     */
    public static DataBuffer createBuffer(long[] shape, DataType type) {
        long length = ArrayUtil.prodLong(shape);

        switch (type) {
            case BOOL:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createBool(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createBool(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case UBYTE:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createUByte(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createUByte(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case UINT16:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createUShort(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createUShort(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case UINT32:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createUInt(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createUInt(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case UINT64:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createULong(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createULong(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case BYTE:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createByte(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createByte(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case SHORT:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createShort(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createShort(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case INT:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createInt(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createInt(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case LONG:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createLong(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createLong(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case HALF:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createHalf(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createHalf(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case BFLOAT16:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createBFloat16(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createBFloat16(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case FLOAT:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createFloat(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case DOUBLE:
                return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createDouble(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
            case UTF8:
            case COMPRESSED:
            case UNKNOWN:
            default:
                throw new UnsupportedOperationException("Cannot create type: " + type);
        }
    }


    /**
     * @see #createBufferDetached(int[], DataType)
     */
    public static DataBuffer createBufferDetached(long[] shape, DataType type) {
        long length = ArrayUtil.prodLong(shape);
        switch (type){

            case DOUBLE:
                return DATA_BUFFER_FACTORY_INSTANCE.createDouble(length);
            case FLOAT:
                return DATA_BUFFER_FACTORY_INSTANCE.createFloat(length);
            case HALF:
                return DATA_BUFFER_FACTORY_INSTANCE.createHalf(length);
            case BFLOAT16:
                return DATA_BUFFER_FACTORY_INSTANCE.createBFloat16(length);
            case UINT64:
                return DATA_BUFFER_FACTORY_INSTANCE.createULong(length);
            case LONG:
                return DATA_BUFFER_FACTORY_INSTANCE.createLong(length);
            case UINT32:
                return DATA_BUFFER_FACTORY_INSTANCE.createUInt(length);
            case INT:
                return DATA_BUFFER_FACTORY_INSTANCE.createInt(length);
            case UINT16:
                return DATA_BUFFER_FACTORY_INSTANCE.createUShort(length);
            case SHORT:
                return DATA_BUFFER_FACTORY_INSTANCE.createShort(length);
            case UBYTE:
                return DATA_BUFFER_FACTORY_INSTANCE.createUByte(length);
            case BYTE:
                return DATA_BUFFER_FACTORY_INSTANCE.createByte(length);
            case BOOL:
                return DATA_BUFFER_FACTORY_INSTANCE.createBool(length);
            case UTF8:
            case COMPRESSED:
            case UNKNOWN:
            default:
                throw new UnsupportedOperationException("Cannot create type: " + type);
        }
    }

    /**
     * Creates a buffer of the specified opType
     * and length with the given byte buffer.
     *
     * This will wrap the buffer as a reference (no copy)
     * if the allocation opType is the same.
     * @param buffer the buffer to create from
     * @param type the opType of buffer to create
     * @param length the length of the buffer
     * @return
     */
    public static DataBuffer createBuffer(ByteBuffer buffer, DataType type, int length) {
        switch (type) {
            case INT:
                return DATA_BUFFER_FACTORY_INSTANCE.createInt(buffer, length);
            case LONG:
                return DATA_BUFFER_FACTORY_INSTANCE.createLong(buffer, length);
            case DOUBLE:
                return DATA_BUFFER_FACTORY_INSTANCE.createDouble(buffer, length);
            case FLOAT:
                return DATA_BUFFER_FACTORY_INSTANCE.createFloat(buffer, length);
            case HALF:
                return DATA_BUFFER_FACTORY_INSTANCE.createHalf(buffer, length);
            default:
                throw new IllegalArgumentException("Illegal opType " + type);
        }
    }

    /**
     * Create a buffer based on the data opType
     *
     * @param data the data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBuffer(byte[] data, int length) {
        DataBuffer ret;
        if (dataType() == DataType.DOUBLE)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createDouble(data, length);
        else if (dataType() == DataType.HALF)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createHalf(data, length);
        else
            ret = DATA_BUFFER_FACTORY_INSTANCE.createFloat(data, length);
        return ret;
    }

    /**
     * Create a buffer equal of length prod(shape)
     *
     * @param data the shape of the buffer to create
     * @return the created buffer
     */
    public static DataBuffer createBuffer(int[] data) {
        DataBuffer ret;
        ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createInt(data) : DATA_BUFFER_FACTORY_INSTANCE.createInt(data, Nd4j.getMemoryManager().getCurrentWorkspace());
        return ret;
    }

    /**
     * Create a buffer equal of length prod(shape)
     *
     * @param data the shape of the buffer to create
     * @return the created buffer
     */
    public static DataBuffer createBuffer(long[] data) {
        DataBuffer ret;
        ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createLong(data) : DATA_BUFFER_FACTORY_INSTANCE.createLong(data, Nd4j.getMemoryManager().getCurrentWorkspace());
        return ret;
    }

    /**
     * Create a buffer equal of length prod(shape). This method is NOT affected by workspaces
     *
     * @param data
     * @return
     */
    public static DataBuffer createBufferDetached(int[] data) {
        DataBuffer ret;
        ret = DATA_BUFFER_FACTORY_INSTANCE.createInt(data);
        return ret;
    }

    /**
     * Create a buffer equal of length prod(shape). This method is NOT affected by workspaces
     *
     * @param data
     * @return
     */
    public static DataBuffer createBufferDetached(long[] data) {
        DataBuffer ret;
        ret = DATA_BUFFER_FACTORY_INSTANCE.createLong(data);
        return ret;
    }

    /**
     * Creates a buffer of the specified length based on the data opType
     *
     * @param length the length of te buffer
     * @return the buffer to create
     */
    public static DataBuffer createBuffer(long length) {
        return createBuffer(length, true);
    }

    /**
     * Create a data buffer
     * based on a pointer
     * with the given opType and length
     * @param pointer the pointer to create the buffer for
     * @param type the opType of pointer
     * @param length the length of the buffer
     * @param  indexer the indexer to use
     * @return the data buffer based on the given parameters
     */
    public static DataBuffer createBuffer(Pointer pointer, DataType type, long length, Indexer indexer) {
        return DATA_BUFFER_FACTORY_INSTANCE.create(pointer, type, length, indexer);
    }

    /**
     * @see #createBuffer(DataType dataType, long length, boolean initialize)
     */
    public static DataBuffer createBuffer(long length, boolean initialize) {
        DataBuffer ret = createBuffer(Nd4j.dataType(), length, initialize);

        return ret;
    }

    /**
     * Create a data buffer based on datatype.
     * @param dataType the type of buffer to create
     * @param length  the length of the buffer
     * @param initialize  flag to leave the underlying memory (false) or initialize with zero (true).
     * @return the created buffer.
     */
    public static DataBuffer createBuffer(DataType dataType, long length, boolean initialize) {
        return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, length, initialize) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType,length, initialize, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Create a data buffer based on datatype, workspace.
     * @param dataType the type of buffer to create
     * @param length  the length of the buffer
     * @param initialize  flag to leave the underlying memory (false) or initialize with zero (true).
     * @param workspace workspace to use for buffer creation.
     * @return the created buffer.
     */
    public static DataBuffer createBuffer(DataType dataType, long length, boolean initialize, MemoryWorkspace workspace) {
        return workspace == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, length, initialize) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType,length, initialize, workspace);
    }

    /**
     * Create a buffer based on the data opType
     * @param data the data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBuffer(float[] data) {
        return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createFloat(data) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(data, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Create a buffer based on underlying array.
     * @param data data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBufferDetached(float[] data) {
        return DATA_BUFFER_FACTORY_INSTANCE.createFloat(data);
    }

    /**
     * @see #createBufferDetached(float[])
     */
    public static DataBuffer createBufferDetached(double[] data) {
        return DATA_BUFFER_FACTORY_INSTANCE.createDouble(data);
    }

    /**
     * @see #createBuffer(float[])
     */
    public static DataBuffer createBuffer(double[] data) {
        return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(data) : DATA_BUFFER_FACTORY_INSTANCE.createDouble(data, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Create a buffer based on the data of the underlying java array with the specified type..
     * @param data underlying java array
     * @param dataType specified type.
     * @return created buffer,
     */
    public static DataBuffer createTypedBuffer(double[] data, DataType dataType) {
        val buffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, Nd4j.getMemoryManager().getCurrentWorkspace());
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType)
     */
    public static DataBuffer createTypedBuffer(float[] data, DataType dataType) {
        val buffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, Nd4j.getMemoryManager().getCurrentWorkspace());
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType)
     */
    public static DataBuffer createTypedBuffer(int[] data, DataType dataType) {
        val buffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, Nd4j.getMemoryManager().getCurrentWorkspace());
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType)
     */
    public static DataBuffer createTypedBuffer(long[] data, DataType dataType) {
        val buffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, Nd4j.getMemoryManager().getCurrentWorkspace());
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType)
     */
    public static DataBuffer createTypedBuffer(short[] data, DataType dataType) {
        val buffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, Nd4j.getMemoryManager().getCurrentWorkspace());
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType)
     */
    public static DataBuffer createTypedBuffer(byte[] data, DataType dataType) {
        val buffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, Nd4j.getMemoryManager().getCurrentWorkspace());
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType)
     */
    public static DataBuffer createTypedBuffer(boolean[] data, DataType dataType) {
        val buffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, Nd4j.getMemoryManager().getCurrentWorkspace());
        buffer.setData(data);
        return buffer;
    }

    /**
     * Create a buffer based on the data of the underlying java array, specified type and workspace
     * @param data underlying java array
     * @param dataType specified type.
     * @param workspace specified workspace.
     * @return created buffer,
     */
    public static DataBuffer createTypedBuffer(double[] data, DataType dataType, MemoryWorkspace workspace) {
        val buffer = workspace == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, workspace);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType, MemoryWorkspace)
     */
    public static DataBuffer createTypedBuffer(float[] data, DataType dataType, MemoryWorkspace workspace) {
        val buffer = workspace == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, workspace);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType, MemoryWorkspace)
     */
    public static DataBuffer createTypedBuffer(int[] data, DataType dataType, MemoryWorkspace workspace) {
        val buffer = workspace == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, workspace);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType, MemoryWorkspace)
     */
    public static DataBuffer createTypedBuffer(long[] data, DataType dataType, MemoryWorkspace workspace) {
        val buffer = workspace == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, workspace);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType, MemoryWorkspace)
     */
    public static DataBuffer createTypedBuffer(short[] data, DataType dataType, MemoryWorkspace workspace) {
        val buffer = workspace == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, workspace);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType, MemoryWorkspace)
     */
    public static DataBuffer createTypedBuffer(byte[] data, DataType dataType, MemoryWorkspace workspace) {
        val buffer = workspace == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, workspace);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBuffer(float[], DataType, MemoryWorkspace)
     */
    public static DataBuffer createTypedBuffer(boolean[] data, DataType dataType, MemoryWorkspace workspace) {
        val buffer = workspace == null ? DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false) : DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false, workspace);
        buffer.setData(data);
        return buffer;
    }

    /**
     *  Create am uninitialized  buffer based on the data of the underlying java array and specified type.
     * @param data underlying java array
     * @param dataType specified type.
     * @return
     */
    public static DataBuffer createTypedBufferDetached(double[] data, DataType dataType) {
        val buffer = DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBufferDetached(double[], DataType)
     */
    public static DataBuffer createTypedBufferDetached(float[] data, DataType dataType) {
        val buffer = DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBufferDetached(double[], DataType)
     */
    public static DataBuffer createTypedBufferDetached(int[] data, DataType dataType) {
        val buffer = DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBufferDetached(double[], DataType)
     */
    public static DataBuffer createTypedBufferDetached(long[] data, DataType dataType) {
        val buffer = DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBufferDetached(double[], DataType)
     */
    public static DataBuffer createTypedBufferDetached(short[] data, DataType dataType) {
        val buffer = DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBufferDetached(double[], DataType)
     */
    public static DataBuffer createTypedBufferDetached(byte[] data, DataType dataType) {
        val buffer = DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false);
        buffer.setData(data);
        return buffer;
    }

    /**
     * @see #createTypedBufferDetached(double[], DataType)
     */
    public static DataBuffer createTypedBufferDetached(boolean[] data, DataType dataType) {
        val buffer = DATA_BUFFER_FACTORY_INSTANCE.create(dataType, data.length, false);
        buffer.setData(data);
        return buffer;
    }

    /**
     * Set the factory instance for INDArray creation.
     * @param factory new INDArray factory
     */
    public static void setFactory(NDArrayFactory factory) {
        INSTANCE = factory;
    }

    /**
     * Set the factory instance for sparse INDArray creation.
     * @param factory new INDArray factory
     */
    public static void setSparseFactory(NDArrayFactory factory) {
        SPARSE_INSTANCE = factory;
    }

    /**
     * Returns the ordering of the ndarrays
     *
     * @return the ordering of the ndarrays
     */
    public static Character order() {
        return factory().order();
    }

    /**
     * Returns the data opType used for the runtime
     *
     * @return the datatype used for the runtime
     */
    public static DataType dataType() {
        return DataTypeUtil.getDtypeFromContext();
    }

    /**
     * DEPRECATED - use {@link #setDefaultDataTypes(DataType, DataType)}
     * This method sets dataType for the current JVM.
     * @param dtype Data type to set
     * @deprecated use {@link #setDefaultDataTypes(DataType, DataType)}. Equivalent to {@code setDefaultDataTypes(dtype, (dtype.isFPType() ? dtype : defaultFloatingPointType()))}
     */
    @Deprecated
    public static void setDataType(@NonNull DataType dtype) {
        setDefaultDataTypes(dtype, (dtype.isFPType() ? dtype : defaultFloatingPointType()));
    }

    /**
     * Set the default data types.<br>
     * The default data types are used for array creation methods where no data type is specified.<br>
     * When the user explicitly provides a datatype (such as in Nd4j.ones(DataType.FLOAT, 1, 10)) these default values
     * will not be used.<br>
     * defaultType: used in methods such as Nd4j.ones(1,10) and Nd4j.zeros(10).<br>
     * defaultFloatingPointType: used internally where a floating point array needs to be created, but no datatype is specified.
     * defaultFloatingPointType must be one of DOUBLE, FLOAT or HALF
     *
     * @param defaultType              Default datatype for new arrays (used when no type is specified).
     * @param defaultFloatingPointType Default datatype for new floating point arrays (used when no type is specified. Must be one of DOUBLE, FLOAT or HALF
     */
    public static void setDefaultDataTypes(@NonNull DataType defaultType, @NonNull DataType defaultFloatingPointType){
        Preconditions.checkArgument(defaultFloatingPointType.isFPType(), "Invalid default floating point type: %s is not a floating point type", defaultFloatingPointType);
        DataTypeUtil.setDTypeForContext(defaultType);
        Nd4j.defaultFloatingPointDataType.set(defaultFloatingPointType);
    }

    /**
     * Retrieve the Nd4J backend.
     * @return the Nd4J backend.
     */
    public static Nd4jBackend getBackend() {
        return backend;
    }

    /**
     * Retrieve the BLAS wrapper.
     * @return the BLAS wrapper.
     */
    public static BlasWrapper getBlasWrapper() {
        return BLAS_WRAPPER_INSTANCE;
    }

    /**
     * Retreive the sparse BLAS wrapper.
     * @return the sparse BLAS wrapper.
     */
    public static BlasWrapper getSparseBlasWrapper() {
        return SPARSE_BLAS_WRAPPER_INSTANCE;
    }

    /**
     * Sets the global blas wrapper
     *
     * @param factory
     */
    public static void setBlasWrapper(BlasWrapper factory) {
        BLAS_WRAPPER_INSTANCE = factory;
    }

    /**
     * Sort an ndarray along a particular dimension.<br>
     * Note that the input array is modified in-place.
     *
     * @param ndarray   the ndarray to sort
     * @param dimension the dimension to sort
     * @return the indices and the sorted ndarray (the original array, modified in-place)
     */
    public static INDArray[] sortWithIndices(INDArray ndarray, int dimension, boolean ascending) {
        INDArray indices = Nd4j.create(ndarray.shape());
        INDArray[] ret = new INDArray[2];

        long nV = ndarray.vectorsAlongDimension(dimension);
        for (int i = 0; i < nV; i++) {
            INDArray vec = ndarray.vectorAlongDimension(i, dimension);
            INDArray indexVector = indices.vectorAlongDimension(i, dimension);
            final Double[] data = new Double[(int) vec.length()];
            final Double[] index = new Double[(int) vec.length()];

            for (int j = 0; j < vec.length(); j++) {
                data[j] = vec.getDouble(j);
                index[j] = (double) j;
            }

            /**
             * Inject a comparator that sorts indices relative to
             * the actual values in the data.
             * This allows us to retain the indices
             * and how they were rearranged.
             */


            Arrays.sort(index, new Comparator<Double>() {
                @Override
                public int compare(Double o1, Double o2) {
                    int o = (int) o1.doubleValue();
                    int oo2 = (int) o2.doubleValue();
                    return Double.compare(data[o], data[oo2]);
                }
            });

            if (ascending)
                for (int j = 0; j < vec.length(); j++) {
                    vec.putScalar(j, data[(int) index[j].doubleValue()]);
                    indexVector.putScalar(j, index[j]);
                }
            else {
                int count = data.length - 1;
                for (int j = 0; j < vec.length(); j++) {
                    int currCount2 = count;
                    count--;
                    vec.putScalar(j, data[(int) index[currCount2].doubleValue()]);
                    indexVector.putScalar(j, index[currCount2]);
                }
            }


        }

        ret[0] = indices;
        ret[1] = ndarray;

        return ret;
    }

    /**
     * Sort all elements of an array.
     *
     * sorts all elements of an array. For multi dimansional arrays the result depends on the array ordering]
     *
     * Nd4j.factory().setOrder('f');
     * INDArray x = Nd4j.arange(4).reshape(2,2);
     * Nd4j.sort(x, true);
     * gives: [[         0,    2.0000], [    1.0000,    3.0000]]
     *
     * The same ode with .setOrder('c')
     * [[         0,    1.0000], [    2.0000,    3.0000]]
     *
     * @param ndarray array to sort
     * @param ascending true for ascending, false for descending
     * @return
     */
    public static INDArray sort(INDArray ndarray, boolean ascending) {
        return getNDArrayFactory().sort(ndarray, !ascending);
    }

    /**
     * Sort an ndarray along a particular dimension<br>
     * Note that the input array is modified in-place.
     *
     * @param ndarray   the ndarray to sort
     * @param dimension the dimension to sort
     * @return the sorted ndarray
     */
    public static INDArray sort(INDArray ndarray, int dimension, boolean ascending) {
        return getNDArrayFactory().sort(ndarray, !ascending, dimension);
    }

    /**Sort (shuffle) the rows of a 2d array according to the value at a specified column.
     * Other than the order of the rows, each row is unmodified. Copy operation: original
     * INDArray is unmodified<br>
     * So if sorting the following on values of column 2 (ascending):<br>
     * [a b 2]<br>
     * [c d 0]<br>
     * [e f -3]<br>
     * Then output is<br>
     * [e f -3]<br>
     * [c d 0]<br>
     * [a b 2]<br>
     * @param in 2d array to sort
     * @param colIdx The column to sort on
     * @param ascending true if smallest-to-largest; false if largest-to-smallest
     * @return
     */
    public static INDArray sortRows(final INDArray in, final int colIdx, final boolean ascending) {
        if (in.rank() != 2)
            throw new IllegalArgumentException("Cannot sort rows on non-2d matrix");
        if (colIdx < 0 || colIdx >= in.columns())
            throw new IllegalArgumentException("Cannot sort on values in column " + colIdx + ", nCols=" + in.columns());

        if (in.rows() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        INDArray out = Nd4j.create(in.dataType(), in.shape());
        int nRows = (int) in.rows();
        ArrayList<Integer> list = new ArrayList<Integer>(nRows);
        for (int i = 0; i < nRows; i++)
            list.add(i);
        Collections.sort(list, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                if (ascending)
                    return Double.compare(in.getDouble(o1, colIdx), in.getDouble(o2, colIdx));
                else
                    return -Double.compare(in.getDouble(o1, colIdx), in.getDouble(o2, colIdx));
            }
        });
        for (int i = 0; i < nRows; i++) {
            out.putRow(i, in.getRow(list.get(i)));
        }
        return out;
    }

    /**Sort (shuffle) the columns of a 2d array according to the value at a specified row.
     * Other than the order of the columns, each column is unmodified. Copy operation: original
     * INDArray is unmodified<br>
     * So if sorting the following on values of row 1 (ascending):<br>
     * [a b c]<br>
     * [1 -1 0]<br>
     * [d e f]<br>
     * Then output is<br>
     * [b c a]<br>
     * [-1 0 1]<br>
     * [e f d]<br>
     * @param in 2d array to sort
     * @param rowIdx The row to sort on
     * @param ascending true if smallest-to-largest; false if largest-to-smallest
     * @return
     */
    public static INDArray sortColumns(final INDArray in, final int rowIdx, final boolean ascending) {
        if (in.rank() != 2)
            throw new IllegalArgumentException("Cannot sort columns on non-2d matrix");
        if (rowIdx < 0 || rowIdx >= in.rows())
            throw new IllegalArgumentException("Cannot sort on values in row " + rowIdx + ", nRows=" + in.rows());

        if (in.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        INDArray out = Nd4j.create(in.shape());
        int nCols = (int) in.columns();
        ArrayList<Integer> list = new ArrayList<>(nCols);
        for (int i = 0; i < nCols; i++)
            list.add(i);
        Collections.sort(list, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                if (ascending)
                    return Double.compare(in.getDouble(rowIdx, o1), in.getDouble(rowIdx, o2));
                else
                    return -Double.compare(in.getDouble(rowIdx, o1), in.getDouble(rowIdx, o2));
            }
        });
        for (int i = 0; i < nCols; i++) {
            out.putColumn(i, in.getColumn(list.get(i)));
        }
        return out;
    }

    /**
     * Create an n x (shape)
     * ndarray where the ndarray is repeated num times
     *
     * @param n   the ndarray to replicate
     * @param num the number of copies to repeat
     * @return the repeated ndarray
     */
    public static INDArray repeat(INDArray n, int num) {
        List<INDArray> list = new ArrayList<>();
        for (int i = 0; i < num; i++)
            list.add(n.dup());
        long[] nShape = n.shape();
        long[] shape = n.isColumnVector() ? new long[] {n.shape()[0]} : nShape;
        long[] retShape = Longs.concat(new long[] {num}, shape);
        return Nd4j.create(list, retShape);
    }

    /**
     * Generate a linearly spaced vector
     *
     * @param lower upper bound
     * @param num   number of items in returned vector
     * @param step  the step (incompatible with <b>upper</b>)
     * @return the linearly spaced vector
     */
    public static INDArray linspace(@NonNull DataType dtype, long lower, long num, long step) {
        // for now we'll temporarily keep original impl
        if(num == 1) {
            return Nd4j.scalar(dtype, lower);
        }

        if (dtype.isIntType()) {
            long upper = lower + num * step;
            return linspaceWithCustomOpByRange((long) lower, upper, num, (long) step, dtype);
        } else if (dtype.isFPType()) {
            return Nd4j.getExecutioner().exec(new Linspace((double) lower, num, (double)step, dtype));
        }
        else {
            throw new IllegalStateException("Illegal data type for linspace: " + dtype.toString());
        }
    }

    /**
     * Generate a linearly spaced vector with default data type
     *
     * @param lower lower bound
     * @param upper upper bound
     * @param num   number of items in returned vector
     * @return the linearly spaced vector
     */
    public static INDArray linspace(long lower, long upper, long num) {
        return linspace(lower, upper, num, Nd4j.dataType());
    }


    /**
     * Generate a linearly spaced vector
     *
     * @param lower lower bound
     * @param upper upper bound
     * @param num   number of items in returned vector
     * @return the linearly spaced vector
     */
    public static INDArray linspace(long lower, long upper, long num, @NonNull DataType dtype) {
        // for now we'll temporarily keep original impl
        if(lower == upper && num == 1) {
            return Nd4j.scalar(dtype, lower);
        }
        if (dtype.isIntType()) {
            return linspaceWithCustomOp((long)lower, (long)upper, (int)num, dtype);
        } else if (dtype.isFPType()) {
            return linspace((double) lower, (double)upper, (int) num, dtype);
        }
        else {
            throw new IllegalStateException("Illegal data type for linspace: " + dtype.toString());
        }
    }

    /**
     * Generate a linearly spaced 1d vector of the specified datatype
     *
     * @param lower lower bound
     * @param step step between items
     * @param num   number of resulting items
     * @return the linearly spaced vector
     */
    public static INDArray linspace(@NonNull DataType dataType, double lower, double step, long num) {
        Preconditions.checkState(dataType.isFPType());
        return Nd4j.getExecutioner().exec(new Linspace(lower, num, step, dataType));
    }

    /**
     * Generate a linearly spaced 1d vector of the specified datatype
     *
     * @param lower lower bound
     * @param upper upper bound
     * @param num   number of resulting items
     * @return the linearly spaced vector
     */
    public static INDArray linspace( double lower, double upper, long num, @NonNull DataType dataType) {
        Preconditions.checkState(dataType.isFPType());
        return Nd4j.getExecutioner().exec(new Linspace(lower, upper, num, dataType));
    }

    private static INDArray linspaceWithCustomOp(long lower, long upper, int num, DataType dataType) {

        INDArray result = Nd4j.createUninitialized(dataType, new long[] {num}, Nd4j.order());

        val op = DynamicCustomOp.builder("lin_space")
                .addInputs(Nd4j.scalar(lower), Nd4j.scalar(upper), Nd4j.scalar(num))
                .addOutputs(result)
                .build();

        Nd4j.getExecutioner().execAndReturn(op);
        return result;
    }

    private static INDArray linspaceWithCustomOpByRange(long lower, long upper, long num, long step, DataType dataType) {

        INDArray result = Nd4j.createUninitialized(dataType, new long[] {num}, Nd4j.order());

        val op = DynamicCustomOp.builder("range")
                .addInputs(Nd4j.scalar(lower), Nd4j.scalar(upper), Nd4j.scalar(step))
                .addOutputs(result)
                .build();

        Nd4j.getExecutioner().execAndReturn(op);
        return result;
    }

    /**
     * Meshgrid op. Returns a pair of arrays where values are broadcast on a 2d grid.<br>
     * For example, if x = [1,2,3,4] and y = [5,6,7], then:<br>
     * out[0] =<br>
     * [1,2,3,4]<br>
     * [1,2,3,4]<br>
     * [1,2,3,4]<br>
     * <br>
     * out[1] =<br>
     * [5,5,5,5]<br>
     * [6,6,6,6]<br>
     * [7,7,7,7]<br>
     * <br>
     *
     * @param x X array input
     * @param y Y array input
     * @return INDArray[] of length 2, shape [y.length, x.length]
     */
    public static INDArray[] meshgrid(@NonNull INDArray x, @NonNull INDArray y){
        Preconditions.checkArgument(x.isVectorOrScalar(), "X must be a vector");
        Preconditions.checkArgument(y.isVectorOrScalar(), "Y must be a vector");
        if(y.dataType() != x.dataType())
            y = y.castTo(x.dataType());

        INDArray xOut = Nd4j.createUninitialized(x.dataType(), y.length(), x.length());
        INDArray yOut = Nd4j.createUninitialized(x.dataType(), y.length(), x.length());

        CustomOp op = DynamicCustomOp.builder("meshgrid")
                .addInputs(x, y)
                .addOutputs(xOut, yOut)
                .build();
        Nd4j.getExecutioner().execAndReturn(op);

        return new INDArray[]{xOut, yOut};
    }


    /**
     * Create a long row vector of all of the given ndarrays
     * @param matrices the matrices to create the flattened ndarray for
     * @return the flattened representation of
     * these ndarrays
     */
    public static INDArray toFlattened(Collection<INDArray> matrices) {
        INDArray ret = INSTANCE.toFlattened(matrices);
        return ret;
    }

    /**
     * Create a long row vector of all of the given ndarrays
     * @param order the order in which to flatten the matrices
     * @param matrices the matrices to create the flattened ndarray for
     * @return the flattened representation of
     * these ndarrays
     */
    public static INDArray toFlattened(char order, Collection<INDArray> matrices) {
        INDArray ret = INSTANCE.toFlattened(order, matrices);
        return ret;
    }

    /**
     * Create a long row vector of all of the given ndarrays
     * @param matrices the matrices to create the flattened ndarray for
     * @return the flattened representation of
     * these ndarrays
     */
    public static INDArray toFlattened(int length, Iterator<? extends INDArray>... matrices) {
        INDArray ret = INSTANCE.toFlattened(length, matrices);
        return ret;
    }

    /**
     * Returns a column vector where each entry is the nth bilinear
     * product of the nth slices of the two tensors.
     */
    public static INDArray bilinearProducts(INDArray curr, INDArray in) {
        return INSTANCE.bilinearProducts(curr, in);
    }

    /**
     * Create a long row vector of all of the given ndarrays
     * @param matrices the matrices to create the flattened ndarray for
     * @return the flattened representation of
     * these ndarrays
     */
    public static INDArray toFlattened(@NonNull INDArray... matrices) {
        return INSTANCE.toFlattened(matrices);
    }

    /**
     * Create a long row vector of all of the given ndarrays/
     * @param order order in which to flatten ndarrays
     * @param matrices the matrices to create the flattened ndarray for

     * @return the flattened representation of
     * these ndarrays
     */
    public static INDArray toFlattened(char order, @NonNull INDArray... matrices) {
        return INSTANCE.toFlattened(order, matrices);
    }

    /**
     * Create the identity ndarray
     *
     * @param n the number for the identity
     * @return
     */
    public static INDArray eye(long n) {
        INDArray ret = INSTANCE.eye(n);
        return ret;
    }

    /**
     * Rotate a matrix 90 degrees
     *
     * @param toRotate the matrix to rotate
     * @return the rotated matrix
     */
    public static void rot90(INDArray toRotate) {
        INSTANCE.rot90(toRotate);

    }

    /**
     * Write NDArray to a text file
     *
     * @param filePath
     * @param split    the split separator, defaults to ","
     * @deprecated custom col separators are no longer supported; uses ","
     * @param precision digits after the decimal point
     * @deprecated Precision is no longer used.
     * Defaults to scientific notation with 18 digits after the decimal
     * Use {@link #writeTxt(INDArray, String)}
     */
    public static void writeTxt(INDArray write, String filePath, String split, int precision) {
        writeTxt(write,filePath);
    }

    /**
     * Write NDArray to a text file
     *
     * @param write
     * @param filePath
     * @param precision
     * @deprecated Precision is no longer used.
     * Defaults to scientific notation with 18 digits after the decimal
     * Use {@link #writeTxt(INDArray, String)}
     */
    public static void writeTxt(INDArray write, String filePath, int precision) {
        writeTxt(write, filePath);
    }

    /**
     * Write NDArray to a text file
     *
     * @param write
     * @param filePath
     * @param split
     * @deprecated custom col and higher dimension separators are no longer supported; uses ","
     * Use {@link #writeTxt(INDArray, String)}
     */
    public static void writeTxt(INDArray write, String filePath, String split) {
        writeTxt(write,filePath);
    }

    /**
     * Write NDArray to a text file
     *
     * @param write Array to write
     * @param filePath path to write to
     */
    public static void writeTxt(INDArray write, String filePath) {
        try {
            String toWrite = writeStringForArray(write, "0.000000000000000000E0");
            FileUtils.writeStringToFile(new File(filePath), toWrite);
        } catch (IOException e) {
            throw new RuntimeException("Error writing output", e);
        }
    }

    /**
     * @deprecated custom col separators are no longer supported; uses ","
     * @deprecated precision can no longer be specified. The array is written in scientific notation.
     * @see #writeTxtString(INDArray, OutputStream)
     */
    public static void writeTxtString(INDArray write, OutputStream os, String split, int precision) {
        writeTxtString(write,os);
    }

    /**
     * @deprecated precision can no longer be specified. The array is written in scientific notation.
     * @see #writeTxtString(INDArray, OutputStream)
     */
    @Deprecated
    public static void writeTxtString(INDArray write, OutputStream os, int precision) {
        writeTxtString(write,os);
    }

    /**
     * @deprecated column separator can longer be specified; Uses ","
     * @see #writeTxtString(INDArray, OutputStream)
     */
    @Deprecated
    public static void writeTxtString(INDArray write, OutputStream os, String split) {
        writeTxtString(write, os);
    }

    /**
     * Write ndarray as text to output stream
     * @param write Array to write
     * @param os    stream to write too.
     */
    public static void writeTxtString(INDArray write, OutputStream os) {
        try {
            // default format is "0.000000000000000000E0"
            String toWrite = writeStringForArray(write, "0.000000000000000000E0");
            os.write(toWrite.getBytes());
        } catch (IOException e) {
            throw new RuntimeException("Error writing output", e);
        }
    }

    private static String writeStringForArray(INDArray write, String format) {
        if(write.isView() || !Shape.hasDefaultStridesForShape(write))
            write = write.dup();
        if (format.isEmpty()) format = "0.000000000000000000E0";
        String lineOne = "{\n";
        String lineTwo = "\"filefrom\": \"dl4j\",\n";
        String lineThree = "\"ordering\": \"" + write.ordering() + "\",\n";
        String lineFour = "\"shape\":\t" + java.util.Arrays.toString(write.shape()) + ",\n";
        String lineFive = "\"data\":\n";
        String fileData = new NDArrayStrings(",", format).format(write, false);
        String fileEnd = "\n}\n";
        String fileBegin = lineOne + lineTwo + lineThree + lineFour + lineFive;
        String fileContents = fileBegin + fileData + fileEnd;
        return fileContents;
    }



    /**Y
     * Write an ndarray to a writer
     * @param writer the writer to write to
     * @param write the ndarray to write
     * @throws IOException
     */
    public static void write(OutputStream writer, INDArray write) throws IOException {
        DataOutputStream stream = new DataOutputStream(writer);
        write(write, stream);
        stream.close();
    }


    /**
     * Convert an ndarray to a byte array
     * @param arr the array to convert
     * @return the converted byte array
     * @throws IOException
     */
    public static byte[] toByteArray(INDArray arr) throws IOException {
        if (arr.length() * arr.data().getElementSize() >  Integer.MAX_VALUE)
            throw new ND4JIllegalStateException("");

        ByteArrayOutputStream bos = new ByteArrayOutputStream((int) (arr.length() * arr.data().getElementSize()));
        DataOutputStream dos = new DataOutputStream(bos);
        write(arr, dos);
        byte[] ret = bos.toByteArray();
        return ret;
    }

    /**
     * Read an ndarray from a byte array
     * @param arr the array to read from
     * @return the deserialized ndarray
     * @throws IOException
     */
    public static INDArray fromByteArray(byte[] arr) throws IOException {
        ByteArrayInputStream bis = new ByteArrayInputStream(arr);
        INDArray ret = read(bis);
        return ret;
    }


    /**
     * Read line via input streams
     *
     * @param filePath the input stream ndarray
     * @param split    the split separator
     * @return the read txt method
     */
    public static INDArray readNumpy(@NonNull InputStream filePath, @NonNull String split) throws IOException {
        return readNumpy(DataType.FLOAT, filePath, split, StandardCharsets.UTF_8);
    }

    /**
     * Read array from input stream.
     *
     * @param dataType datatype of array
     * @param filePath the input stream
     * @param split    the split separator
     * @param charset the  charset
     * @return the deserialized array.
     * @throws IOException
     */
    public static INDArray readNumpy(@NonNull DataType dataType, @NonNull InputStream filePath, @NonNull String split, @NonNull Charset charset) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(filePath, charset));
        String line;
        List<float[]> data2 = new ArrayList<>();
        int numColumns = -1;
        INDArray ret;
        while ((line = reader.readLine()) != null) {
            String[] data = line.trim().split(split);
            if (numColumns < 0) {
                numColumns = data.length;
            } else
                Preconditions.checkState(data.length == numColumns,
                        "Data has inconsistent number of columns: data length %s, numColumns %s", data.length, numColumns);
            data2.add(readSplit(data));


        }
        ret = Nd4j.create(dataType, data2.size(), numColumns);
        for (int i = 0; i < data2.size(); i++) {
            float[] row = data2.get(i);
            INDArray arr = Nd4j.create(row, new long[]{1, row.length}, dataType);
            ret.putRow(i, arr);
        }
        return ret;
    }

    private static float[] readSplit(String[] split) {
        float[] ret = new float[split.length];
        for (int i = 0; i < split.length; i++) {
            try {
                ret[i] = Float.parseFloat(split[i]);
            } catch (NumberFormatException e) {
                if (split[i].equalsIgnoreCase("inf")) {
                    ret[i] = Float.POSITIVE_INFINITY;
                } else if (split[i].equalsIgnoreCase("-inf")) {
                    ret[i] = Float.NEGATIVE_INFINITY;
                } else if (split[i].equalsIgnoreCase("nan")) {
                    ret[i] = Float.NaN;
                } else
                    throw new RuntimeException(e);

            }
        }
        return ret;
    }

    /**
     * Read line via input streams
     *
     * @param filePath the input stream ndarray
     * @param split    the split separator
     * @return the read txt method
     */
    public static INDArray readNumpy(String filePath, String split) throws IOException {
        return readNumpy(DataType.FLOAT, filePath, split);
    }

    /**
     * Read array via input stream using standard UTF-8 encoding.
     *
     * @see #readNumpy(DataType, InputStream, String , Charset)
     */
    public static INDArray readNumpy(DataType dataType, String filePath, String split) throws IOException {
        try(InputStream is = new FileInputStream(filePath)) {
            return readNumpy(dataType, is, split, StandardCharsets.UTF_8);
        }
    }

    /**
     * Read line via input streams
     *
     * @param filePath the input stream ndarray
     * @return the read txt method
     */
    public static INDArray readNumpy(String filePath) throws IOException {
        return readNumpy(DataType.FLOAT, filePath);
    }

    /**
     * Read array with default split and UTF-8 encoding.
     *
     * @see #readNumpy(DataType, InputStream, String , Charset)
     */
    public static INDArray readNumpy(DataType dataType, String filePath) throws IOException {
        return readNumpy(dataType, filePath, " ");
    }

    /**
     * Raad an ndarray from an input stream
     *
     * @see #read(DataInputStream)
     */
    public static INDArray read(InputStream reader) throws IOException {
        return read(new DataInputStream(reader));
    }

    /**
     * Read line via input streams
     *
     * @param ndarray the input stream ndarray
     * @return NDArray
     */
    public static INDArray readTxtString(InputStream ndarray) {
        String sep = ",";
        /*
         We could dump an ndarray to a file with the tostring (since that is valid json) and use put/get to parse it as json
         But here we leverage our information of the tostring method to be more efficient
         With our current toString format we use tads along dimension (rank-1,rank-2) to write to the array in two dimensional chunks at a time.
         This is more efficient than setting each value at a time with putScalar.
         This also means we can read the file one line at a time instead of loading the whole thing into memory
        */
        INDArray newArr = null;
        BufferedReader reader = new BufferedReader(new InputStreamReader(ndarray));
        LineIterator it = IOUtils.lineIterator(reader);
        DecimalFormat format = (DecimalFormat) NumberFormat.getInstance(Locale.US);
        format.setParseBigDecimal(true);
        try {
            int lineNum = 0;
            int tensorNum = 0;
            char theOrder = 'c';
            int rank = 0;
            long[] theShape = null;
            double[] subsetArr = null;
            while (it.hasNext()) {
                String line = it.nextLine();
                lineNum++;
                line = line.replaceAll("\\s", "");
                if (line.equals("") || line.equals("}"))
                    continue;
                // is it from dl4j?
                if (lineNum == 2) {
                    String[] lineArr = line.split(":");
                    String fileSource = lineArr[1].replaceAll("\\W", "");
                    if (!fileSource.equals("dl4j"))
                        throw new IllegalArgumentException("Only files written out from Nd4j.writeTxT/writeTxtString can be read with the readTxt/readTxtString methods");
                }
                // parse ordering
                if (lineNum == 3) {
                    String[] lineArr = line.split(":");
                    theOrder = lineArr[1].replaceAll("\\W", "").charAt(0);
                    continue;
                }
                // parse shape
                if (lineNum == 4) {
                    String shapeString = line.split(":")[1].replace("[", "").replace("],", "");
                    if (shapeString.isEmpty()) {
                        newArr = Nd4j.scalar(Nd4j.defaultFloatingPointType(), 0);
                    } else {
                        String[] shapeArr = shapeString.split(",");
                        rank = shapeArr.length;
                        theShape = new long[rank];
                        for (int i = 0; i < rank; i++) {
                            theShape[i] = Integer.parseInt(shapeArr[i]);
                        }
                        if (theOrder == 'f' && theShape[rank-1] == 1) {
                            //Hack fix for tad issue with 'f' order and rank-1 dim shape == 1
                            newArr = Nd4j.create(Nd4j.defaultFloatingPointType(), theShape, 'c');
                        }
                        else {
                            newArr = Nd4j.create(Nd4j.defaultFloatingPointType(), theShape, theOrder);
                        }
                        subsetArr = new double[(int) theShape[rank - 1]];
                    }
                    continue;
                }
                //parse data
                if (lineNum > 5) {
                    String[] entries = line.replace("\\],", "").replaceAll("\\]", "").replaceAll("\\[", "").split(sep);
                    if (rank == 0) {
                        try {
                            newArr.addi((format.parse(entries[0])).doubleValue());
                        } catch (ParseException e) {
                            e.printStackTrace();
                        }
                    } else {
                        Preconditions.checkState(entries.length == theShape[rank-1], "Invalid number of entries - format does not match expected shape." +
                                "Expected %s values per line, got %s at line %s", theShape[rank-1], entries.length, lineNum );
                        for (int i = 0; i < theShape[rank - 1]; i++) {
                            try {
                                BigDecimal number = (BigDecimal) format.parse(entries[i]);
                                subsetArr[i] = number.doubleValue();
                            } catch (ParseException e) {
                                e.printStackTrace();
                            }
                        }
                        INDArray subTensor = Nd4j.create(subsetArr, new long[]{subsetArr.length}, Nd4j.defaultFloatingPointType());
                        newArr.tensorAlongDimension(tensorNum, rank - 1).addi(subTensor);
                        tensorNum++;
                    }
                }
            }
            //Hack fix for tad issue with 'f' order and rank-1 dim shape == 1
            if (theOrder == 'f' && rank > 1 && theShape[rank-1] == 1) {
                newArr = newArr.dup('f');
            }

        } finally {
            LineIterator.closeQuietly(it);
        }

        if(newArr == null){
            throw new IllegalStateException("Cannot parse file: file does not appear to represent a text serialized INDArray file");
        }

        return newArr;
    }

    /**
     * Read line via input streams
     *
     * @param filePath the input stream ndarray
     * @return NDArray
     */
    public static INDArray readTxt(String filePath) {
        String sep = ",";
        File file = new File(filePath);
        InputStream is = null;
        try {
            is = new FileInputStream(file);
            return readTxtString(is);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } finally {
            if (is != null) {
                try {
                    is.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private static int[] toIntArray(int length, DataBuffer buffer) {
        int[] ret = new int[length];
        for (int i = 0; i < length; i++) {
            ret[i] = buffer.getInt(i);
        }
        return ret;
    }

    /**
     * Create array based in data buffer and shape info,
     *
     * @param data Data buffer.
     * @param shapeInfo shape information.
     * @return new INDArray.
     */
    public static INDArray createArrayFromShapeBuffer(DataBuffer data, DataBuffer shapeInfo) {
        val jvmShapeInfo = shapeInfo.asLong();
        val dataType = ArrayOptionsHelper.dataType(jvmShapeInfo);
        val shape = Shape.shape(jvmShapeInfo);
        val strides = Shape.stridesOf(jvmShapeInfo);
        val order = Shape.order(jvmShapeInfo);
        INDArray result = Nd4j.create(data, shape, strides, 0, order, dataType);
        if (data instanceof CompressedDataBuffer)
            result.markAsCompressed(true);

        return result;
    }

    /**
     * Create array based in data buffer and shape info,
     *
     * @param data data buffer.
     * @param shapeInfo shape information.
     * @return new INDArray.
     */
    public static INDArray createArrayFromShapeBuffer(DataBuffer data, Pair<DataBuffer, long[]> shapeInfo) {
        int rank = Shape.rank(shapeInfo.getFirst());
        long offset = Shape.offset(shapeInfo.getFirst());
        INDArray result = Nd4j.create(data, toIntArray(rank, Shape.shapeOf(shapeInfo.getFirst())),
                toIntArray(rank, Shape.stride(shapeInfo.getFirst())), offset, Shape.order(shapeInfo.getFirst()));
        if (data instanceof CompressedDataBuffer)
            result.markAsCompressed(true);

        return result;
    }

    /**
     * Read in an ndarray from a data input stream
     *
     * @param dis the data input stream to read from
     * @return the ndarray
     * @throws IOException
     */
    public static INDArray read(DataInputStream dis) throws IOException {
        val headerShape = BaseDataBuffer.readHeader(dis);

        var shapeInformation = Nd4j.createBufferDetached(new long[]{headerShape.getMiddle().longValue()}, headerShape.getRight());
        shapeInformation.read(dis, headerShape.getLeft(), headerShape.getMiddle(), headerShape.getThird());
        val length = Shape.length(shapeInformation);
        DataType type = null;
        DataBuffer data = null;

        val headerData = BaseDataBuffer.readHeader(dis);
        try {
            // current version contains dtype in extras
            data = CompressedDataBuffer.readUnknown(dis, headerData.getFirst(), headerData.getMiddle(), headerData.getRight());
            type = ArrayOptionsHelper.dataType(shapeInformation.asLong());
        } catch (ND4JUnknownDataTypeException e) {
            // manually setting data type
            type = headerData.getRight();
            long extras = ArrayOptionsHelper.setOptionBit(0L, type);
            shapeInformation.put(shapeInformation.length() - 3, extras);
        }

        return createArrayFromShapeBuffer(data, shapeInformation);
    }

    /**
     * Write an ndarray to the specified outputstream
     *
     * @param arr              the array to write
     * @param dataOutputStream the data output stream to write to
     * @throws IOException
     */
    public static void write(INDArray arr, DataOutputStream dataOutputStream) throws IOException {
        //BaseDataBuffer.write(...) doesn't know about strides etc, so dup (or equiv. strategy) is necessary here
        //Furthermore, because we only want to save the *actual* data for a view (not the full data), the shape info
        // (mainly strides, offset, element-wise stride) may be different in the duped array vs. the view array
        if (arr.isView())
            arr = arr.dup();

        arr.shapeInfoDataBuffer().write(dataOutputStream);
        arr.data().write(dataOutputStream);
    }

    /**
     * Save an ndarray to the given file
     * @param arr the array to save
     * @param saveTo the file to save to
     * @throws IOException
     */
    public static void saveBinary(INDArray arr, File saveTo) throws IOException {
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(saveTo));
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(arr, dos);
        dos.flush();
        dos.close();
        bos.close();
    }

    /**
     * Read a binary ndarray from the given file
     * @param read the nd array to read
     * @return the loaded ndarray
     * @throws IOException
     */
    public static INDArray readBinary(File read) throws IOException {
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(read));
        DataInputStream dis = new DataInputStream(bis);
        INDArray ret = Nd4j.read(dis);
        dis.close();
        return ret;
    }

    /**
     * Clear nans from an ndarray
     *
     * @param arr the array to clear
     */
    public static void clearNans(INDArray arr) {
        //BooleanIndexing.applyWhere(arr, Conditions.isNan(), new Value(Nd4j.EPS_THRESHOLD));
        getExecutioner().exec(new ReplaceNans(arr, Nd4j.EPS_THRESHOLD));
    }

    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     *
     * @param reverse the matrix to reverse
     * @return the reversed matrix
     */
    public static INDArray rot(INDArray reverse) {
        INDArray ret = INSTANCE.rot(reverse);
        return ret;
    }

    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     *
     * @param reverse the matrix to reverse
     * @return the reversed matrix
     */
    public static INDArray reverse(INDArray reverse) {
        //INDArray ret = INSTANCE.reverse(reverse);
        //logCreationIfNecessary(ret);
        return Nd4j.getExecutioner().exec(new OldReverse(reverse));
    }

    /**
     * Create a 1D array of evenly spaced values between {@code begin} (inclusive) and {@code end} (exclusive)
     * with a step size.
     *
     * @param begin the begin of the range (inclusive)
     * @param end   the end of the range (exclusive)
     * @param step spacing between values. Default value is 1.
     * @return the 1D range vector
     */
    public static INDArray arange(double begin, double end, double step) {
        INDArray ret = INSTANCE.arange(begin, end, step);
        return ret;
    }

    /**
     * Create a 1D array of evenly spaced values between {@code begin} (inclusive) and {@code end} (exclusive)
     * with a step size of 1
     *
     * @see #arange(double, double, double)
     */
    public static INDArray arange(double begin, double end) {
        INDArray ret = INSTANCE.arange(begin, end, 1);
        return ret;
    }

    /**
     * Create a 1D array of evenly spaced values between 0 (inclusive) and {@code end} (exclusive)
     * with a step size of 1
     *
     * @see #arange(double, double, double)
     */
    public static INDArray arange(double end) {
        return arange(0, end);
    }

    /**
     * Copy a to b
     *
     * @param a the origin matrix
     * @param b the destination matrix
     */
    public static void copy(INDArray a, INDArray b) {
        INSTANCE.copy(a, b);
    }

    /**
     * Creates a new matrix where the values of the given vector are the diagonal values of
     * the matrix if a vector is passed in, if a matrix is returns the kth diagonal
     * in the matrix
     *
     * @param x the diagonal values
     * @param k the kth diagonal to get
     * @return new matrix
     */
    public static INDArray diag(INDArray x, int k) {
        INDArray ret;
        if(x.isVectorOrScalar() || x.isRowVector() || x.isColumnVector()) {
            ret = Nd4j.create(new long[]{x.length(), x.length()});
            Nd4j.getExecutioner().execAndReturn(new Diag(new INDArray[]{x},new INDArray[]{ret}));
        } else {
            ret = Nd4j.createUninitialized(new long[]{Math.min(x.size(0), x.size(1))});
            Nd4j.getExecutioner().execAndReturn(new DiagPart(x,ret));
        }
        return ret;
    }

    /**
     * Creates a new matrix where the values of the given vector are the diagonal values of
     * the matrix if a vector is passed in, if a matrix is returns the kth diagonal
     * in the matrix
     *
     * @param x the diagonal values
     * @return new matrix
     */
    public static INDArray diag(INDArray x) {
        return diag(x, 0);
    }

    /**
     * This method samples value from Source array to Target, with probabilites provided in Probs argument
     *
     * @param source source array.
     * @param probs array with probabilities.
     * @param target destination array.
     * @param rng Random number generator.
     * @return the destination (target) array.
     */
    public static INDArray choice(@NonNull INDArray source, @NonNull INDArray probs, @NonNull INDArray target,
                                  @NonNull org.nd4j.linalg.api.rng.Random rng) {
        if (source.length() != probs.length())
            throw new ND4JIllegalStateException("Nd4j.choice() requires lengths of Source and Probs to be equal");

        return Nd4j.getExecutioner().exec(new Choice(source, probs, target), rng);
    }

    /**
     * This method samples value from Source array to Target,the default random number generator.
     *
     * @see #choice(INDArray, INDArray, INDArray, org.nd4j.linalg.api.rng.Random)
     */
    public static INDArray choice(INDArray source, INDArray probs, INDArray target) {
        return choice(source, probs, target, Nd4j.getRandom());
    }

    /**
     *
     *
     * @param source
     * @param probs
     * @param numSamples
     * @return
     */

    /**
     * This method returns new INDArray instance, sampled from Source array with probabilities given in Probs.
     *
     * @param numSamples number of samples to take. (size of the new NDArray).
     * @see #choice(INDArray, INDArray, int, org.nd4j.linalg.api.rng.Random)
     */
    public static INDArray choice(INDArray source, INDArray probs, int numSamples,
                                  @NonNull org.nd4j.linalg.api.rng.Random rng) {
        if (numSamples < 1)
            throw new ND4JIllegalStateException("Nd4j.choice() numSamples must be positive value");

        return choice(source, probs, createUninitialized(numSamples), rng);
    }

    /**
     * This method returns new INDArray instance, sampled from Source array with probabilities given in Probs
     * using the default random number generator.
     *
     * @see #choice(INDArray, INDArray, int, org.nd4j.linalg.api.rng.Random)
     */
    public static INDArray choice(INDArray source, INDArray probs, int numSamples) {
        return choice(source, probs, numSamples, Nd4j.getRandom());
    }

    public static INDArray appendBias(@NonNull INDArray... vectors) {
        INDArray ret = INSTANCE.appendBias(vectors);
        return ret;
    }

    /**
     * Perform an operation along a diagonal
     *
     * @param x    the ndarray to perform the operation on
     * @param func the operation to perform
     */
    public static void doAlongDiagonal(INDArray x, Function<Number, Number> func) {
        if (x.isMatrix())
            for (int i = 0; i < x.rows(); i++)
                x.put(i, i, func.apply(x.getDouble(i, i)));
    }


    ////////////////////// RANDOM ///////////////////////////////

    /**
     * Create a random ndarray with values from a uniform distribution over (0, 1) with the given shape
     *
     * @param shape the shape of the array
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(@NonNull int... shape) {
        INDArray ret = createUninitialized(shape, order()); //INSTANCE.rand(shape, Nd4j.getRandom());
        return rand(ret);
    }

    /**
     * @see #rand(int[])
     */
    public static INDArray rand(@NonNull long... shape) {
        INDArray ret = createUninitialized(shape, order()); //INSTANCE.rand(shape, Nd4j.getRandom());
        return rand(ret);
    }

    /**
     * Create a random ndarray with values from a uniform distribution over (0, 1) with the given shape and data type
     *
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(@NonNull DataType dataType, @NonNull long... shape) {
        Preconditions.checkArgument(dataType.isFPType(),
                "Can't create a random array of a non-floating point data type");
        INDArray ret = createUninitialized(dataType, shape, order()); //INSTANCE.rand(shape, Nd4j.getRandom());
        return rand(ret);
    }

    /**
     * Create a random ndarray with the given shape and array order
     *
     * Values are sampled from a uniform distribution over (0, 1)
     *
     * @param order the order of the ndarray to return
     * @param shape the shape of the array
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(char order, @NonNull int... shape) {
        INDArray ret = Nd4j.createUninitialized(shape, order); //INSTANCE.rand(order, shape);
        return rand(ret);
    }

    /**
     * @deprecated use {@link Nd4j#rand(org.nd4j.linalg.api.buffer.DataType, char, long...))
     */
    @Deprecated
    public static INDArray rand(@NonNull DataType dataType, int[] shape, char order) {
        return rand(dataType, order, ArrayUtil.toLongArray(shape));
    }

    /**
     * @deprecated use {@link org.nd4j.linalg.factory.Nd4j#rand(org.nd4j.linalg.api.buffer.DataType, char, long...)}
     */
    @Deprecated
    public static INDArray rand(@NonNull DataType dataType, char order, @NonNull int... shape) {
        return rand(dataType, order, ArrayUtil.toLongArray(shape));
    }

    /**
     * Create a random ndarray with the given shape, data type, and array order
     *
     * Values are sampled from a uniform distribution over (0, 1)
     *
     * @param order the order of the ndarray to return
     * @param shape the shape of the ndarray
     * @param dataType the data type of the ndarray
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(@NonNull DataType dataType, char order, @NonNull long... shape) {
        INDArray ret = Nd4j.createUninitialized(dataType, shape, order); //INSTANCE.rand(order, shape);
        return rand(ret);
    }


    /**
     * Create a random ndarray with the given shape and data type
     *
     * Values are sampled from a uniform distribution over (0, 1)
     *
     * @param shape the shape of the ndarray
     * @param dataType the data type of the ndarray
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(@NonNull DataType dataType, @NonNull int... shape) {
        INDArray ret = Nd4j.createUninitialized(dataType, ArrayUtil.toLongArray(shape), Nd4j.order()); //INSTANCE.rand(order, shape);
        return rand(ret);
    }

    /**
     * Create a random ndarray with values from a uniform distribution over (0, 1) with the given shape
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows, int columns) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        INDArray ret = createUninitialized(new int[] {rows, columns}, Nd4j.order());//INSTANCE.rand(rows, columns, Nd4j.getRandom());
        return rand(ret);
    }

    /**
     * Create a random ndarray with the given shape and output order
     *
     * Values are sampled from a uniform distribution over (0, 1)
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(char order, int rows, int columns) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        INDArray ret = createUninitialized(new int[] {rows, columns}, order);//INSTANCE.rand(order, rows, columns);
        return rand(ret);
    }

    /**
     * Create a random ndarray with values from a uniform distribution over (0, 1) with the given shape
     * using given seed
     *
     * @param shape the shape of the array
     * @param seed  the  seed to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(long seed, @NonNull long... shape) {
        INDArray ret = createUninitialized(shape, Nd4j.order());//;INSTANCE.rand(shape, seed);
        return rand(ret, seed);
    }

    /**
     * @deprecated use {@link Nd4j#rand(long, long...)}
     */
    @Deprecated
    public static INDArray rand(int[] shape, long seed) {
        INDArray ret = createUninitialized(shape, Nd4j.order());//;INSTANCE.rand(shape, seed);
        return rand(seed, ArrayUtil.toLongArray(shape));
    }


    /**
     * Create a random ndarray with values from a uniform distribution over (0, 1) with the given shape
     * using the given seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param seed    the  seed to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows, int columns, long seed) {
        INDArray ret = createUninitialized(new int[] {rows, columns}, Nd4j.order());
        return rand(ret, seed);
    }

    /**
     * @deprecated use {@link Nd4j#rand(org.nd4j.linalg.api.rng.Random, long...)}
     */
    @Deprecated
    public static INDArray rand(int[] shape, @NonNull org.nd4j.linalg.api.rng.Random rng) {
        return rand(rng, ArrayUtil.toLongArray(shape));
    }

    /**
     * Create a random ndarray with the given shape using the given RandomGenerator
     *
     * @param shape the shape of the array
     * @param rng     the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(@NonNull org.nd4j.linalg.api.rng.Random rng, @NonNull long... shape) {
        INDArray ret = createUninitialized(shape, Nd4j.order()); //INSTANCE.rand(shape, rng);
        return rand(ret, rng);
    }

    /**
     * @deprecated use {@link Nd4j#rand(org.nd4j.linalg.api.rng.distribution.Distribution, long...)}
     */
    @Deprecated
    public static INDArray rand(int[] shape, @NonNull Distribution dist) {
        return rand(dist, ArrayUtil.toLongArray(shape));
    }

    /**
     * @deprecated use
     * {@link org.nd4j.linalg.factory.Nd4j#rand(org.nd4j.linalg.api.rng.distribution.Distribution, long...)}
     */
    @Deprecated
    public static INDArray rand(long[] shape, @NonNull Distribution dist) {
        return rand(dist, shape);
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the array
     * @param dist  distribution to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(@NonNull Distribution dist, @NonNull long... shape) {
        //INDArray ret = INSTANCE.rand(shape, dist);
        //logCreationIfNecessary(ret);
        return dist.sample(shape);
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param rng       the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows, int columns, @NonNull org.nd4j.linalg.api.rng.Random rng) {
        INDArray ret = createUninitialized(new int[] {rows, columns}, order());//INSTANCE.rand(rows, columns, rng);
        return rand(ret, rng);
    }

    /**
     * @deprecated use {@link Nd4j#rand(double, double, org.nd4j.linalg.api.rng.Random, long...)}
     */
    @Deprecated
    public static INDArray rand(int[] shape, double min, double max, @NonNull org.nd4j.linalg.api.rng.Random rng) {
        return rand(min, max, rng, ArrayUtil.toLongArray(shape));
    }

    /**
     * @deprecated use {@link Nd4j#rand(double, double, org.nd4j.linalg.api.rng.Random, long...)}
     */
    @Deprecated
    public static INDArray rand(long[] shape, double min, double max, @NonNull org.nd4j.linalg.api.rng.Random rng) {
        INDArray ret = createUninitialized(shape, order()); //INSTANCE.rand(shape, min, max, rng);
        return rand(ret, min, max, rng);
    }

    /**
     * Generates a random matrix between min and max
     *
     * @param shape the number of rows of the matrix
     * @param min   the minimum number
     * @param max   the maximum number
     * @param rng   the rng to use
     * @return a random matrix of the specified shape and range
     */
    public static INDArray rand(double min, double max, @NonNull org.nd4j.linalg.api.rng.Random rng, @NonNull long... shape) {
        INDArray ret = createUninitialized(shape, order()); //INSTANCE.rand(shape, min, max, rng);
        return rand(ret, min, max, rng);
    }

    /**
     * Generates a random matrix between min and max
     *
     * @param rows    the number of rows of the matrix
     * @param columns the number of columns in the matrix
     * @param min     the minimum number
     * @param max     the maximum number
     * @param rng     the rng to use
     * @return a drandom matrix of the specified shape and range
     */
    public static INDArray rand(int rows, int columns, double min, double max, @NonNull org.nd4j.linalg.api.rng.Random rng) {
        INDArray ret = createUninitialized(rows, columns);//INSTANCE.rand(rows, columns, min, max, rng);
        return rand(ret, min, max, rng);
    }

    /**
     * Fill the given ndarray with random numbers drawn from a normal distribution
     *
     * @param target  target array
     * @return the given target array
     */
    public static INDArray randn(INDArray target) {
        return getExecutioner().exec(new GaussianDistribution(target), Nd4j.getRandom());
    }

    /**
     * Create a ndarray of the given shape with values from N(0,1)
     *
     * @param shape the shape of the array
     * @return new array with random values
     */
    public static INDArray randn(@NonNull int... shape) {
        return randn(ArrayUtil.toLongArray(shape));
    }


    /**
     * Create a ndarray of the given shape and data type with values from N(0,1)
     *
     * @param shape the shape of the ndarray
     * @return
     */
    public static INDArray randn(@NonNull DataType dataType, @NonNull int... shape) {
        return randn(dataType, ArrayUtil.toLongArray(shape));
    }

    /**
     * Create a ndarray of the given shape and data type with values from N(0,1)
     *
     * @param dataType datatype to use, must be a float type datatype.
     * @param shape shape for the new array.
     * @return new array with random values
     */
    public static INDArray randn(@NonNull DataType dataType, @NonNull long... shape) {
        INDArray ret = Nd4j.createUninitialized(dataType, shape, order());
        return randn(ret);
    }


    /**
     * Create a ndarray of the given shape with values from N(0,1).
     * Defaults to FLOAT and c-order.
     *
     * @param shape shape for the new array.
     * @return new array with random values
     */
    public static INDArray randn(@NonNull long... shape) {
        INDArray ret = Nd4j.createUninitialized(shape, order());
        return randn(ret);
    }

    /**
     * Random normal N(0,1) with the specified shape and array order
     *
     * @param order order of the output ndarray
     * @param shape the shape of the array
     * @return new array with random values
     */
    public static INDArray randn(char order, @NonNull int... shape) {
        INDArray ret = Nd4j.createUninitialized(shape, order);
        return randn(ret);
    }

    /**
     * Random normal N(0,1) with the specified shape and array order
     *
     * @param order order of the output ndarray
     * @param shape the shape of the array
     * @return new array with random values
     */
    public static INDArray randn(char order, @NonNull long... shape) {
        INDArray ret = Nd4j.createUninitialized(shape, order);
        return randn(ret);
    }


    /**
     * Random normal N(0,1) with the specified shape and array order
     *
     * @param order order of the output ndarray
     * @param shape the shape of the ndarray
     * @param dataType the data type of the ndarray
     */
    public static INDArray randn(@NonNull DataType dataType, char order, @NonNull long... shape) {
        INDArray ret = Nd4j.createUninitialized(dataType, shape, order);
        return randn(ret);
    }

    /**
     * @deprecated use {@link Nd4j#randn(long, long...)}
     */
    @Deprecated
    public static INDArray randn(int[] shape, long seed) {
        return randn(seed, ArrayUtil.toLongArray(shape));
    }

    /**
     * Random normal N(0, 1) using the specified seed
     *
     * @param shape the shape of the array
     * @return
     */
    public static INDArray randn(long seed, @NonNull long... shape) {
        INDArray ret = Nd4j.createUninitialized(shape, order());
        return randn(ret, seed);
    }

    /**
     * Random normal N(0, 1)
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    public static INDArray randn(long rows, long columns) {
        INDArray ret = Nd4j.createUninitialized(new long[]{rows, columns}, order());
        return randn(ret);
    }

    /**
     * Random normal N(0,1) with the specified shape and array order
     *
     * @param order   the order of the output array
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     */
    public static INDArray randn(char order, long rows, long columns) {
        INDArray ret = Nd4j.createUninitialized(new long[]{rows, columns}, order);
        return randn(ret);
    }

    /**
     * Random normal using the specified seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    public static INDArray randn(long rows, long columns, long seed) {
        INDArray ret = Nd4j.createUninitialized(new long[]{rows, columns}, order());
        return randn(ret, seed);
    }

    /**
     * Random normal using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r       the random generator to use
     * @return
     */
    public static INDArray randn(long rows, long columns, @NonNull org.nd4j.linalg.api.rng.Random r) {
        INDArray ret = Nd4j.createUninitialized(new long[]{rows, columns}, order());
        return randn(ret, r);
    }

    /**
     * @deprecated use {@link Nd4j#randn(org.nd4j.linalg.api.rng.Random, long...)}
     */
    @Deprecated
    public static INDArray randn(int[] shape, @NonNull org.nd4j.linalg.api.rng.Random r) {
        return randn(r, ArrayUtil.toLongArray(shape));
    }

    /**
     * @deprecated use {@link Nd4j#randn(org.nd4j.linalg.api.rng.Random, long...)}
     */
    @Deprecated
    public static INDArray randn(long[] shape, @NonNull org.nd4j.linalg.api.rng.Random r) {
        return randn(r, shape);
    }

    /**
     * Random normal using the given rng
     *
     * @param shape the shape of the array
     * @param r     the random generator to use
     * @return
     */
    public static INDArray randn(@NonNull org.nd4j.linalg.api.rng.Random r, @NonNull long... shape) {
        final INDArray ret = Nd4j.createUninitialized(shape, order());
        return randn(ret, r);
    }

    /**
     * Fill the given ndarray with random numbers drawn from a uniform distribution
     *
     * @param target  target array
     * @return the given target array
     */
    public static INDArray rand(INDArray target) {
        return getExecutioner().exec(new UniformDistribution(target), Nd4j.getRandom());
    }

    /**
     * Fill the given ndarray with random numbers drawn from a uniform distribution
     *
     * @param target  target array
     * @param seed the  seed to use
     * @return the given target array
     */
    public static INDArray rand(INDArray target, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return getExecutioner().exec(new UniformDistribution(target), Nd4j.getRandom());
    }

    /**
     * Fill the given ndarray with random numbers drawn from a uniform distribution using the given RandomGenerator
     *
     * @param target  target array
     * @param rng     the random generator to use
     * @return the given target array
     */
    public static INDArray rand(INDArray target, @NonNull org.nd4j.linalg.api.rng.Random rng) {
        return getExecutioner().exec(new UniformDistribution(target), rng);
    }

    /**
     * Fill the given ndarray with random numbers drawn from the given distribution
     *
     * @param target  target array
     * @param dist  distribution to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(INDArray target, @NonNull Distribution dist) {
        return dist.sample(target);
    }

    /**
     * Fill the given ndarray with random numbers drawn from a uniform distribution using the given RandomGenerator
     *
     * @param target  target array
     * @param min   the minimum number
     * @param max   the maximum number
     * @param rng     the random generator to use
     * @return the given target array
     */
    public static INDArray rand(INDArray target,  double min, double max, @NonNull org.nd4j.linalg.api.rng.Random rng) {
        if (min > max)
            throw new IllegalArgumentException("the maximum value supplied is smaller than the minimum");
        return getExecutioner().exec(new UniformDistribution(target, min, max), rng);
    }

    /**
     * Fill the given ndarray with random numbers drawn from a normal distribution
     *
     * @param target  target array
     * @return the given target array
     */
    public static INDArray randn(INDArray target, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return getExecutioner().exec(new GaussianDistribution(target), Nd4j.getRandom());
    }

    /**
     * Fill the given ndarray with random numbers drawn from a normal distribution utilizing the given random generator
     *
     * @param target  target array
     * @param rng     the random generator to use
     * @return the given target array
     */
    public static INDArray randn(INDArray target, @NonNull org.nd4j.linalg.api.rng.Random rng) {
        return getExecutioner().exec(new GaussianDistribution(target), rng);
    }

    /**
     * Generate a random array according to a binomial distribution with probability p: i.e., values 0 with probability
     * (1-p) or value 1 with probability p
     *
     * @param p     Probability. Must be in range 0 to 1
     * @param shape Shape of the result array
     * @return Result array
     */
    public static INDArray randomBernoulli(double p, @NonNull long... shape) {
        return randomBernoulli(p, Nd4j.createUninitialized(shape));
    }

    /**
     * Fill the specified array with values generated according to a binomial distribution with probability p: i.e.,
     * values 0 with probability (1-p) or value 1 with probability p
     *
     * @param p      Probability. Must be in range 0 to 1
     * @param target Result array to place generated values in
     * @return Result array
     */
    public static INDArray randomBernoulli(double p, @NonNull INDArray target) {
        Preconditions.checkArgument(p >= 0 && p <= 1.0, "Invalid probability: must be in range 0 to 1, got %s", p);
        return Nd4j.getExecutioner().exec(new BernoulliDistribution(target, p));
    }

    /**
     * Generate an array with random values generated according to a binomial distribution with the specified
     * number of trials and probability
     *
     * @param nTrials Number of trials. Must be >= 0
     * @param p       Probability. Must be in range 0 to 1
     * @param shape   Shape of the result array
     * @return Result array
     */
    public static INDArray randomBinomial(int nTrials, double p, @NonNull long... shape) {
        return randomBinomial(nTrials, p, Nd4j.createUninitialized(shape));
    }

    /**
     * Fill the target array with random values generated according to a binomial distribution with the specified
     * number of trials and probability
     *
     * @param nTrials Number of trials. Must be >= 0
     * @param p       Probability. Must be in range 0 to 1
     * @param target  Result array
     * @return Result array
     */
    public static INDArray randomBinomial(int nTrials, double p, INDArray target) {
        Preconditions.checkArgument(p >= 0 && p <= 1.0, "Invalid probability: must be in range 0 to 1, got %s", p);
        Preconditions.checkArgument(nTrials >= 0, "Number of trials must be positive: got %s", nTrials);
        return Nd4j.getExecutioner().exec(new BinomialDistribution(target, nTrials, p));
    }

    /**
     * Exponential distribution: P(x) = lambda * exp(-lambda * x)
     *
     * @param lambda Must be > 0
     * @param shape  Shape of the array to generate
     */
    public static INDArray randomExponential(double lambda, long... shape) {
        return randomExponential(lambda, Nd4j.createUninitialized(shape));
    }

    /**
     * Exponential distribution: P(x) = lambda * exp(-lambda * x)
     *
     * @param lambda Must be > 0
     * @param target Array to hold the result
     */
    public static INDArray randomExponential(double lambda, INDArray target) {
        Preconditions.checkArgument(lambda > 0, "Lambda argument must be >= 0 - got %s", lambda);
        INDArray shapeArr = Nd4j.create(ArrayUtil.toDouble(target.shape()));
        Nd4j.getExecutioner().execAndReturn(new RandomExponential(shapeArr, target, lambda));
        return target;
    }

    ////////////////////// CREATE ///////////////////////////////

    /**
     * This method returns uninitialized 2D array of rows x columns
     *
     * PLEASE NOTE: memory of underlying array will be NOT initialized, and won't be set to 0.0
     *
     * @param rows
     * @param columns
     * @return
     */
    public static INDArray createUninitialized(long rows, long columns) {
        return createUninitialized(new long[] {rows, columns});
    }

    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(float[] data) {
        return create(data, order());
    }

    /**
     * Create a vector based on a java boolean array.
     * @param data java boolean array
     * @return the created ndarray.
     */
    public static INDArray create(boolean[] data) {
        return INSTANCE.create(data, new long[]{data.length}, new long[]{1}, DataType.BOOL, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Creates a row vector with the data
     *
     * @param list the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(List<? extends Number> list) {
        INDArray array = create(list.size());
        int cnt = 0;
        if (dataType() == DataType.DOUBLE) {
            for (Number element: list) {
                array.putScalar(cnt++,element.doubleValue());
            }
        } else {
            for (Number element : list) {
                array.putScalar(cnt++,element.floatValue());
            }
        }
        return array;
    }

    /**
     * Create double array based on java double array.
     *
     * @param data java double array,
     * @return the created ndarray
     */
    public static INDArray create(double[] data) {
        return create(data, order());
    }

    /**
     * Create 2D float array based on java 2d float array.
     * @param data java 2d arrau.
     * @return the created ndarray.
     */
    public static INDArray create(float[][] data) {
        return INSTANCE.create(data);
    }

    /**
     * Create 2D float array based on java 2d float array and ordering.
     * @param data java 2d arrau.
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the created ndarray.
     */
    public static INDArray create(float[][] data, char ordering) {
        return INSTANCE.create(data, ordering);
    }

    /**
     * Create 2D double array based on java 2d double array. and ordering
     *
     * @param data the data to use
     * @return the created ndarray.
     */
    public static INDArray create(double[][] data) {
        return INSTANCE.create(data);
    }

    /**
     * Create 2D long array based on java 2d long array.
     * @param data java 2d long array
     * @return the created ndarray.
     */
    public static INDArray create(long[][] data) {
        val shape = new long[]{data.length, data[0].length};
        return INSTANCE.create(ArrayUtil.flatten(data), shape, getStrides(shape), DataType.LONG, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Create 2D boolean array based on java 2d boolean array.
     * @param data java 2d boolean array.
     * @return the created ndarray.
     */
    public static INDArray create(boolean[][] data) {
        val shape = new long[]{data.length, data[0].length};
        return INSTANCE.create(ArrayUtil.flatten(data), shape, getStrides(shape), DataType.BOOL, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Create a boolean array with given shape based on java 2d boolean array.
     * @param data java 2d boolean array.
     * @param shape desired shape of new array.
     * @return the created ndarray.
     */
    public static INDArray create(boolean[][] data, @NonNull long... shape) {
        return INSTANCE.create(ArrayUtil.flatten(data), shape, getStrides(shape), DataType.BOOL, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Create a 3D double array based on the 3D java double array.
     * @param data java 3d double array.
     * @return the created ndarray.
     */
    public static INDArray create(double[][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length});
    }

    /**
     * Create a 3D float array based on the 3D java float array.
     * @param data java 3d float array.
     * @return the created ndarray.
     */
    public static INDArray create(float[][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length});
    }

    /**
     * create 3D int array based on 3D java int array.
     * @param data java 3D i array.
     * @return the created ndarray.
     */
    public static INDArray create(int[][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length});
    }

    /**
     * Create 4D double array based on 4D java double array.
     * @param data java 4D double array.
     * @return the created ndarray.
     */
    public static INDArray create(double[][][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length, data[0][0][0].length});
    }

    /**
     * Create 4D float array based on 4D java float array.
     * @param data java 4D float array.
     * @return the created ndarray.
     */
    public static INDArray create(float[][][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length, data[0][0][0].length});
    }

    /**
     * Create 4D int array based on 4D java int array.
     * @param data java 4D int array.
     * @return the created ndarray.
     */
    public static INDArray create(int[][][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length, data[0][0][0].length});
    }


    /**
     * Create a 2D double array based on a 2D java double array with given ordering.
     * @param data java 2D double array.
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the created ndarray,
     */
    public static INDArray create(double[][] data, char ordering) {
        return INSTANCE.create(data, ordering);
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(int columns) {
        return create(columns, order());
    }

    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @param order Fortran 'f' or C/C++ 'c' ordering.
     * @return the created ndarray
     */
    public static INDArray create(float[] data, char order) {
        INDArray ret = INSTANCE.create(data, order);
        return ret;
    }

    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @param order Fortran 'f' or C/C++ 'c' ordering.
     * @return the created ndarray
     */
    public static INDArray create(double[] data, char order) {
        INDArray ret = INSTANCE.create(data, order);
        return ret;
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @param order Fortran 'f' or C/C++ 'c' ordering.
     * @return the created ndarray
     */
    public static INDArray create(int columns, char order) {
        INDArray ret = INSTANCE.create(new long[] {columns}, Nd4j.getStrides(new long[] {columns}, order), 0, order);
        return ret;
    }

    /**
     * Create a 1D float array in soecified order initialized with zero.
     * @param columns number of elements.
     * @param order Fortran 'f' or C/C++ 'c' ordering.
     * @return the created ndarray.
     */
    public static INDArray zeros(int columns, char order) {
        return Nd4j.create(columns, order);
    }

    /**
     * Create an array of the specified type and shape initialized with values from a java 1d array.
     * @param data java array used for initialisation. Must have at least the number of elements required.
     * @param shape desired shape of new array.
     * @param type Datatype of the new array. Does not need to match int. data will be converted.
     * @return the created ndarray.
     */
    public static INDArray create(int[] data, long[] shape, DataType type) {
        return INSTANCE.create(data, shape, Nd4j.getStrides(shape), type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], long[], DataType)
     */
    public static INDArray create(long[] data, long[] shape, DataType type) {
        return INSTANCE.create(data, shape, Nd4j.getStrides(shape), type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], long[], DataType)
     */
    public static INDArray create(double[] data, long[] shape, DataType type) {
        return INSTANCE.create(data, shape, Nd4j.getStrides(shape), type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], long[], DataType)
     */
    public static INDArray create(float[] data, long[] shape, DataType type) {
        return  INSTANCE.create(data, shape, Nd4j.getStrides(shape), type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], long[], DataType)
     */
    public static INDArray create(short[] data, long[] shape, DataType type) {
        return INSTANCE.create(data, shape, Nd4j.getStrides(shape), type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], long[], DataType)
     */
    public static INDArray create(byte[] data, long[] shape, DataType type) {
        return INSTANCE.create(data, shape, Nd4j.getStrides(shape), type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], long[], DataType)
     */
    public static INDArray create(boolean[] data, long[] shape, DataType type) {
        return INSTANCE.create(data, shape, Nd4j.getStrides(shape), type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    ////////////////////////////////////////////////

    /**
     * Create an array of the specified type, shape and stride initialized with values from a java 1d array.
     * @param data java array used for initialisation. Must have at least the number of elements required.
     * @param shape desired shape of new array.
     * @param strides stride, separation of elements in each dimension.
     * @param order Fortran 'f' or C/C++ 'c' ordering.
     * @param type Datatype of the new array. Does not need to match int. data will be converted.
     * @return the created ndarray.
     */
    public static INDArray create(int[] data, long[] shape, long[]strides, char order, DataType type) {
        val ret = INSTANCE.create(data, shape, strides, order, type, Nd4j.getMemoryManager().getCurrentWorkspace());
        return ret;
    }

    /**
     * @see #create(int[], long[], long[], char, DataType)
     */
    public static INDArray create(long[] data, long[] shape, long[]strides, char order, DataType type) {
        return INSTANCE.create(data, shape, strides, order, type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], long[], long[], char, DataType)
     */
    public static INDArray create(double[] data, long[] shape, long[]strides, char order, DataType type) {
        return INSTANCE.create(data, shape, strides, order, type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], long[], long[], char, DataType)
     */
    public static INDArray create(float[] data, long[] shape, long[]strides, char order, DataType type) {
        return INSTANCE.create(data, shape, strides, order, type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], long[], long[], char, DataType)
     */
    public static INDArray create(short[] data, long[] shape, long[]strides, char order, DataType type) {
        return INSTANCE.create(data, shape, strides, order, type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], long[], long[], char, DataType)
     */
    public static INDArray create(byte[] data, long[] shape, long[]strides, char order, DataType type) {
        return INSTANCE.create(data, shape, strides, order, type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], long[], long[], char, DataType)
     */
    public static INDArray create(boolean[] data, long[] shape, long[]strides, char order, DataType type) {
        return INSTANCE.create(data, shape, strides, order, type, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * This method creates new 0D INDArray, aka scalar.
     *
     * PLEASE NOTE: Temporary method, added to ensure backward compatibility
     * @param scalar
     * @return
     * @deprecated Use Nd4j.scalar methods, such as {@link #scalar(double)} or {@link #scalar(DataType, Number)}
     */
    @Deprecated
    public static INDArray trueScalar(Number scalar) {
        val ret = INSTANCE.trueScalar(scalar);
        return ret;
    }

    /**
     * @deprecated Use {@link #createFromArray(boolean...)}
     */
    @Deprecated
    public static INDArray trueVector(boolean[] data) {
        val ret = INSTANCE.trueVector(data);
        return ret;
    }

    /**
     * @deprecated Use {@link #createFromArray(long...)}
     */
    @Deprecated
    public static INDArray trueVector(long[] data) {
        val ret = INSTANCE.trueVector(data);
        return ret;
    }

    /**
     * @deprecated Use {@link #createFromArray(int...)}
     */
    @Deprecated
    public static INDArray trueVector(int[] data) {
        val ret = INSTANCE.trueVector(data);
        return ret;
    }

    /**
     * @deprecated Use {@link #createFromArray(float...)}
     */
    @Deprecated
    public static INDArray trueVector(float[] data) {
        val ret = INSTANCE.trueVector(data);
        return ret;
    }

    /**
     * @deprecated Use {@link #createFromArray(double...)}
     */
    @Deprecated
    public static INDArray trueVector(double[] data) {
        val ret = INSTANCE.trueVector(data);
        return ret;
    }

    /**
     * This method creates "empty" INDArray with datatype determined by {@link #dataType()}
     *
     * @return Empty INDArray
     */
    public static INDArray empty() {
        return empty(Nd4j.dataType());
    }

    /**
     * This method creates "empty" INDArray of the specified datatype
     *
     * @return Empty INDArray
     */
    public static INDArray empty(DataType type) {
        if(EMPTY_ARRAYS[type.ordinal()] == null){
            try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()){
                val ret = INSTANCE.empty(type);
                EMPTY_ARRAYS[type.ordinal()] = ret;
                    }
        }
        return EMPTY_ARRAYS[type.ordinal()];
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the array
     * @return the created ndarray
     */
    public static INDArray create(float[] data, int... shape) {
        if (shape.length == 0 && data.length == 1) {
            return scalar(data[0]);
        }

        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array doesn't match data length");
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape);
        return ret;
    }

    /**
     * @see #create(float[], int[])
     */
    public static INDArray create(float[] data, long... shape) {
        if (shape.length == 0 && data.length == 1) {
            return scalar(data[0]);
        }

        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array doesn't match data length");
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, Nd4j.getStrides(shape, Nd4j.order()), DataType.FLOAT, Nd4j.getMemoryManager().getCurrentWorkspace());
        return ret;
    }

    /**
     * @see #create(float[], int[])
     */
    public static INDArray create(double[] data, long... shape) {
        if (shape.length == 0 && data.length == 1) {
            return scalar(data[0]);
        }

        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array doesn't match data length");
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, Nd4j.getStrides(shape, Nd4j.order()), DataType.DOUBLE, Nd4j.getMemoryManager().getCurrentWorkspace());
        return ret;
    }

    /**
     * Create an array of the specified shape initialized with values from a java 1d array.
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the array
     * @return the created ndarray
     */
    public static INDArray create(double[] data, int... shape) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape) + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        val lshape = ArrayUtil.toLongArray(shape);
        INDArray ret = INSTANCE.create(data, lshape, Nd4j.getStrides(lshape, Nd4j.order()), DataType.DOUBLE, Nd4j.getMemoryManager().getCurrentWorkspace());
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * TODO: No usages found, candidate for deletion/depreciation.
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data, int[] shape, int[] stride, long offset) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, ArrayUtil.toLongArray(shape), ArrayUtil.toLongArray(stride), DataType.DOUBLE, Nd4j.getMemoryManager().getCurrentWorkspace());
        return ret;
    }


    /**
     * * Creates an ndarray
     *
     *TODO: used in one test. has unused paratmeter. Candidate for deletion/depreciation.
     *
     * @param data
     * @param shape
     * @param stride
     * @param offset
     * @param order
     * @return
     */
    public static INDArray create(double[] data, long[] shape, long[] stride, long offset, char order) {
        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, order, DataType.DOUBLE, Nd4j.getMemoryManager().getCurrentWorkspace());
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * TODO: No usages found, candidate for deletion/depreciation.
     *
     * @param data    the data to use with the ndarray
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(float[] data, int rows, int columns, int[] stride, long offset) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        INDArray ret = INSTANCE.create(data, rows, columns, stride, offset);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * TODO: No usages found, candidate for deletion/depreciation.
     *
     * @param data    the data to use with tne ndarray
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data, int rows, int columns, int[] stride, long offset) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        INDArray ret = INSTANCE.create(data, rows, columns, stride, offset);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * TODO: No usages found, candidate for deletion/depreciation.
     *
     * @param shape  the shape of the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(float[] data, int[] shape, long offset) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, offset, Nd4j.order());
        return ret;
    }

    /**
     * TODO: No usages found, candidate for deletion/depreciation.
     */
    public static INDArray create(float[] data, long[] shape, long offset) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, offset, Nd4j.order());
        return ret;
    }

    /**
     * Create an array.
     * Use specified shape and ordering initialized with values from a java 1d array starting at offset.
     *
     * @param data java array used for initialisation. Must have at least the number of elements required.
     * @param shape  desired shape of new array.
     * @param offset the offset of data array used for initialisation.
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the created ndarray.
     */
    public static INDArray create(double[] data, int[] shape, long offset, char ordering) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, Nd4j.getStrides(shape, ordering), offset, ordering);
        return ret;
    }

    /**
     * @see #create(double[], int[], long, char )
     */
    public static INDArray create(double[] data, long[] shape, long offset, char ordering) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, Nd4j.getStrides(shape, ordering), offset, ordering);
        return ret;
    }

    /**
     * Create an array of the specified type, shape and stride initialized with values from a java 1d array using offset.
     *
     * @param data java array used for initialisation. Must have at least the number of elements required.
     * @param shape desired shape of new array.
     * @param stride stride, separation of elements in each dimension.
     * @param offset the offset of data array used for initialisation.
     * @return the instance
     */
    public static INDArray create(float[] data, int[] shape, int[] stride, long offset) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset);
        return ret;
    }

    /**
     * Creates an array with the specified shape from a list of arrays.
     *
     * @param list list of arrays.
     * @param shape desired shape of new array. Must match the resulting shape of combining the list.
     * @return the instance
     */
    public static INDArray create(List<INDArray> list, int... shape) {
        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(list, shape);
        return ret;
    }

    /**
     * @see #create(List, int[])
     */
    public static INDArray create(List<INDArray> list, long... shape) {
        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(list, shape);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the created ndarray.
     */
    public static INDArray create(int rows, int columns, int[] stride, long offset) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        return  INSTANCE.create(rows, columns, stride, offset);
    }

    /**
     * @see #create(int , int , int[] , long )
     */
    public static INDArray zeros(int rows, int columns, int[] stride, long offset) {
        return create(rows, columns, stride, offset);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape, int[] stride, long offset) {
        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, stride, offset);
        return ret;

    }

    /**
     * @see #create(int[] , int[] , long )
     */
    public static INDArray zeros(int[] shape, int[] stride, long offset) {
        return create(shape, stride, offset);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @return the instance
     */
    public static INDArray create(int rows, int columns, int[] stride) {
        return create(rows, columns, stride, order());
    }

    /**
     * @see #create(int, int, int[], char)
     */
    public static INDArray zeros(int rows, int columns, int[] stride) {
        return create(rows, columns, stride, order());
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape, int[] stride) {
        return create(shape, stride, order());
    }

    /**
     * @see #create(int[], int[])
     */
    public static INDArray create(long[] shape, long[] stride) {
        return create(shape, stride, order());
    }


    /**
     * @see #create(int[], int[])
     */
    public static INDArray zeros(int[] shape, int[] stride) {
        return create(shape, stride);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    public static INDArray create(int rows, int columns) {
        return create(rows, columns, order());
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the array
     * @return the instance
     */
    public static INDArray create(int... shape) {
        return create(shape, order());
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the array
     * @return the instance
     */
    public static INDArray create(long... shape) {
        return create(shape, order());
    }

    /**
     * Create an array with specified shape and datatype.
     *
     * @param type Datatype of the new array.
     * @param shape  desired shape of new array.
     * @return the created ndarray.
     */
    public static INDArray create(DataType type, long... shape) {
        return create(type, shape, order());
    }

    /**
     * TODO: No usages found, candidate for deletion/depreciation.
     * @param data
     * @param shape
     * @param stride
     * @param ordering
     * @param offset
     * @return
     */
    public static INDArray create(float[] data, int[] shape, int[] stride, char ordering, long offset) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset, ordering);
        return ret;
    }

    /**
     * TODO: No usages found, candidate for deletion/depreciation.
     * @param data
     * @param shape
     * @param ordering
     * @param offset
     * @return
     */
    public static INDArray create(float[] data, int[] shape, char ordering, long offset) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape) + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, getStrides(shape, ordering), offset, ordering);
        return ret;
    }


    // TODO: No usages found, candidate for deletion/depreciation.
    public static INDArray create(double[] data, int[] shape, char ordering, long offset) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape) + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, getStrides(shape, ordering), offset, ordering);
        return ret;
    }

    /**
     * Create an array based on the data buffer with given shape, stride and offset.
     *
     * @param data data buffer used for initialisation. . Must have at least the number of elements required.
     * @param shape desired shape of new array.
     * @param strides stride, separation of elements in each dimension.
     * @param offset the offset of data array used for initialisation.
     * @return the created ndarray.
     */
    public static INDArray create(DataBuffer data, int[] shape, int[] strides, long offset) {
        checkShapeValues(shape);
        return  INSTANCE.create(data, shape, strides, offset);
    }

    /**
     * @see #create(DataBuffer, int[], int[], long)
     */
    public static INDArray create(DataBuffer data, long[] shape, long[] strides, long offset) {
        checkShapeValues(shape);
        return  INSTANCE.create(data, shape, strides, offset);
    }

    /**
     * @see #create(DataBuffer, int[], int[], long)
     *
     * Uses default strides based on shape.
     */
    public static INDArray create(DataBuffer data, int[] shape, long offset) {
        checkShapeValues(shape);
        return  INSTANCE.create(data, shape, getStrides(shape), offset);
    }

    /**
     * @see #create(DataBuffer data, long[], long[], long, long, char )
     */
    public static INDArray create(DataBuffer data, int[] newShape, int[] newStride, long offset, char ordering) {
        checkShapeValues(newShape);
        return INSTANCE.create(data, newShape, newStride, offset, ordering);
    }

    /**
     * @see #create(DataBuffer data, long[], long[], long, long, char )
     */
    public static INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, char ordering) {
        checkShapeValues(newShape);
        return INSTANCE.create(data, newShape, newStride, offset, ordering);
    }

    /**
     * Create an array based on the data buffer with given shape, stride and offset.
     *
     * @param data data buffer used for initialisation. . Must have at least the number of elements required.
     * @param newShape desired shape of new array.
     * @param newStride stride, separation of elements in each dimension.
     * @param offset the offset of data array used for initialisation.
     * @param ews element wise stride.
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the created ndarray.
     */
    public static INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, long ews, char ordering) {
        checkShapeValues(newShape);
        return INSTANCE.create(data, newShape, newStride, offset, ews, ordering);
    }

    /**
     * Create an array based on the data buffer with given shape, stride, offset and data type.
     *
     * @param data data buffer used for initialisation. . Must have at least the number of elements required.
     * @param newShape desired shape of new array.
     * @param newStride stride, separation of elements in each dimension.
     * @param offset the offset of data array used for initialisation.
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @param dataType data type.
     * @return the created ndarray.
     */
    public static INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, char ordering, DataType dataType) {
        checkShapeValues(newShape);
        return INSTANCE.create(data, newShape, newStride, offset, ordering, dataType);
    }

    // This method gets it own javadoc and not a @see because it is used  often.
    /**
     * Create an array based on the data buffer with given shape.
     *
     * @param data data data buffer used for initialisation. . Must have at least the number of elements required.
     * @param shape desired shape of new array.
     * @return the created ndarray.
     */
    public static INDArray create(DataBuffer data, int... shape) {
        checkShapeValues(shape);
        return INSTANCE.create(data, shape);
    }

    /**
     * @see #create(DataBuffer, int[])
     */
    public static INDArray create(DataBuffer data, long... shape) {
        checkShapeValues(shape);
        return INSTANCE.create(data, shape);
    }

    // This method gets it own javadoc and not a @see because it is used  often.
    /**
     * Create an array based on the data buffer.
     *
     * @param buffer data data buffer used for initialisation.
     * @return the created ndarray.
     */
     public static INDArray create(DataBuffer buffer) {
        return INSTANCE.create(buffer);
    }

    /**
     * Creates an ndarray with the specified shape
     * TODO: unused method.
     *
     * @param data    the data to use with the ndarray
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(float[] data, int rows, int columns, int[] stride, long offset, char ordering) {
        int[] shape = new int[] {rows, columns};
        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset, ordering);
        return ret;
    }

    /**
     * Create an array of given shape and data type.
     * @param shape desired shape of new array.
     * @param dataType data type.
     * @return  the created ndarray.
     */
    public static INDArray create(int[] shape, DataType dataType) {
        checkShapeValues(shape);
        return INSTANCE.create(shape, dataType, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(int[], DataType)
     */
    public static INDArray zeros(int[] shape, DataType dataType) {
        return create(shape, dataType);
    }

    /**
     * Creates an ndarray with the specified shape
     * TODO: unused method.
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data, int[] shape, int[] stride, long offset, char ordering) {
        if (data.length == 1 && shape.length == 0)
            return scalar(data[0]);

        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset, ordering);
        return ret;
    }

    // This method gets it own javadoc and not a @see because it is used  often.
    /**
     * Create an array withgiven shape and ordering based on a java double array.
     * @param data java array used for initialisation. Must have at least the number of elements required.
     * @@param shape desired shape of new array.
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the created ndarray.
     */
    public static INDArray create(double[] data, int[] shape, char ordering) {
        //TODO: duplicate code and issue #8013
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);
        val lshape = ArrayUtil.toLongArray(shape);
       return INSTANCE.create(data, lshape, Nd4j.getStrides(lshape, ordering), ordering, DataType.DOUBLE, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * @see #create(double[], int[], char)
     */
    public static INDArray create(float[] data, int[] shape, char ordering) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);
        return INSTANCE.create(data, shape, ordering);
    }

    /**
     * @see #create(double[], int[], char)
     */
    public static INDArray create(float[] data, long[] shape, char ordering) {
        checkShapeValues(data.length, shape);
        return INSTANCE.create(data, shape, Nd4j.getStrides(shape, ordering), ordering, DataType.FLOAT);
    }

    /**
     * @see #create(double[], int[], char)
     */
    public static INDArray create(double[] data, long[] shape, char ordering) {
        checkShapeValues(data.length, shape);
        return INSTANCE.create(data, shape, Nd4j.getStrides(shape, ordering), ordering, DataType.DOUBLE, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Creates an ndarray with the specified shape
     * TODO: unused method.
     *
     * @param data    the data to use with tne ndarray
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data, int rows, int columns, int[] stride, long offset, char ordering) {
        int[] shape = new int[]{rows,columns};
        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(Nd4j.createBuffer(data), shape, stride, offset, ordering);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     * TODO: unused method.
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(float[] data, int[] shape, int[] stride, long offset, char ordering) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset, ordering);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     * TODO: unused method.
     *
     * @param shape the shape of the array
     * @return the instance
     */
    public static INDArray create(List<INDArray> list, int[] shape, char ordering) {
        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(list, shape, ordering);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     * TODO: unused method. (only used by the zeros method in this class)
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(int rows, int columns, int[] stride, long offset, char ordering) {
        int[] shape = new int[]{rows, columns};
        checkShapeValues(shape);
        return INSTANCE.create(shape, stride, offset, ordering);
    }

    /**
     * TODO: unused method.
     */
    public static INDArray zeros(int rows, int columns, int[] stride, long offset, char ordering) {
        return create(rows, columns, stride, offset, ordering);
    }

    /**
     * Creates an ndarray with the specified shape
     * TODO: unused method. (only used by the zeros method in this class)
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape, int[] stride, long offset, char ordering) {
        if(shape.length == 0)
            return Nd4j.scalar(0.0);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, stride, offset, ordering);
        return ret;

    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(long[] shape, long[] stride, long offset, char ordering) {
        if(shape.length == 0)
            return Nd4j.scalar(0.0);

        checkShapeValues(shape);
        return INSTANCE.create(shape, stride, offset, ordering);
    }

    // TODO: unused method.
    public static INDArray zeros(int[] shape, int[] stride, long offset, char ordering) {
        return create(shape, stride, offset, ordering);
    }

    /**
     * Create a 2D array with given rows, columns, stride and ordering.
     * @param rows number of rows.
     * @param columns number of columns
     * @param stride stride, separation of elements in each dimension.
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the created array.
     */
    public static INDArray create(int rows, int columns, int[] stride, char ordering) {
        int[] shape = new int[]{rows, columns};
        checkShapeValues(shape);
        return INSTANCE.create(shape, stride, 0, ordering);
    }

    // TODO: unused method.
    public static INDArray zeros(int rows, int columns, int[] stride, char ordering) {
        return create(rows, columns, stride, ordering);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the instance
     */
    public static INDArray create(int[] shape, int[] stride, char ordering) {
        if(shape.length == 0)
            return Nd4j.scalar(Nd4j.dataType(), 0.0);

        checkShapeValues(shape);
        return INSTANCE.create(shape, stride, 0, ordering);
    }

    /**
     * @see #create(int[], int[], char)
     */
    public static INDArray create(long[] shape, long[] stride, char ordering) {
        if(shape.length == 0)
            return Nd4j.scalar(Nd4j.dataType(), 0.0);

        checkShapeValues(shape);
        return INSTANCE.create(shape, stride, 0, ordering);
    }

    /**
     * TODO: unused method.
     * @param shape
     * @param stride
     * @param ordering
     * @return
     */
    public static INDArray zeros(int[] shape, int[] stride, char ordering) {
        return create(shape, stride, ordering);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the instance
     */
    public static INDArray create(long rows, long columns, char ordering) {
        return create(new long[] {rows, columns}, ordering);
    }

    /**
     * Create a 2D array initialized with zeros.
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the instance
     */
    public static INDArray zeros(int rows, int columns, char ordering) {
        return create(new int[] {rows, columns}, ordering);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the array
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the instance
     */
    public static INDArray create(@NonNull int[] shape, char ordering) {
        if(shape.length == 0)
            return Nd4j.scalar(dataType(), 0.0);

       return INSTANCE.create(shape, ordering);
    }

    // used  often.
    /**
     * Create an array with given shape and ordering.
     *
     * @param shape the shape of the array
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the created array.
     */
    public static INDArray create(@NonNull long[] shape, char ordering) {
        if(shape.length == 0)
            return Nd4j.scalar(dataType(), 0.0);
        //ensure shapes that wind up being scalar end up with the write shape

        checkShapeValues(shape);
        return INSTANCE.create(shape, ordering);
    }

    /**
     * Create an array with given shape, stride  and ordering.
     *
     * @param dataType data type.
     * @param shape the shape of the array
     * @param strides stride, separation of elements in each dimension.
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the created array.
     */
    public static INDArray create(DataType dataType, @NonNull long[] shape, long[] strides, char ordering) {
        if(shape.length == 0)
            return Nd4j.scalar(dataType, 0.0);

        checkShapeValues(shape);
        return INSTANCE.create(dataType, shape, strides, ordering, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    // used often.
    /**
     * Create an array with given data type shape and ordering.
     *
     * @param dataType data type.
     * @param shape the shape of the array
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the created array.
     */
    public static INDArray create(@NonNull DataType dataType, @NonNull long[] shape, char ordering) {
        if(shape.length == 0)
            return Nd4j.scalar(dataType, 0.0);
        //ensure shapes that wind up being scalar end up with the write shape
        checkShapeValues(shape);
        return INSTANCE.create(dataType, shape, ordering, Nd4j.getMemoryManager().getCurrentWorkspace());
    }


    // TODO: Leaving these until #8028 is fixed.
    /**
     *
     * @param shape
     */
    public static void checkShapeValues(long... shape) {
        for (long e: shape) {
            if (e < 0)
                throw new ND4JIllegalStateException("Invalid shape: Requested INDArray shape " + Arrays.toString(shape)
                        + " contains dimension size values < 0 (all dimensions must be 0 or more)");
        }
    }

    // TODO: Leaving these until #8028 is fixed.
    /**
     *
     * @param shape
     */
    public static void checkShapeValues(int... shape) {
        for (int e: shape) {
            if (e < 1)
                throw new ND4JIllegalStateException("Invalid shape: Requested INDArray shape " + Arrays.toString(shape)
                        + " contains dimension size values < 0 (all dimensions must be 0 or more)");
        }
    }

    protected static void checkShapeValues(int length, int... shape) {
        checkShapeValues(shape);

        if (ArrayUtil.prodLong(shape) > length)
            throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                    + " doesn't match data length: " + length);
    }

    protected static void checkShapeValues(int length, long... shape) {
        checkShapeValues(shape);

        if (ArrayUtil.prodLong(shape) > length)
            throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                    + " doesn't match data length: " + length);
    }


    /**
     * Creates an *uninitialized* array with the specified shape and ordering.<br>
     * <b>NOTE</b>: The underlying memory (DataBuffer) will not be initialized. Don't use this unless you know what you are doing.
     *
     * @param shape the shape of the array
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the instance
     */
    public static INDArray createUninitialized(int[] shape, char ordering) {
        if (shape.length == 0)
            return scalar(dataType(), 0.0);

        checkShapeValues(shape);
        return INSTANCE.createUninitialized(shape, ordering);
    }

    public static INDArray createUninitialized(DataType type, long... shape) {
        return createUninitialized(type, shape, Nd4j.order());
    }

    /**
     * Creates an *uninitialized* array with the specified data type, shape and ordering.
     *
     * @param type data type
     * @param shape the shape of the array
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the created array.
     */
    public static INDArray createUninitialized(DataType type, long[] shape, char ordering) {
        if (shape.length == 0) {
            if(type == DataType.UTF8){
                return scalar("");
            }
            return scalar(type, 0);
        }

        checkShapeValues(shape);
        return INSTANCE.createUninitialized(type, shape, ordering, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Creates an *uninitialized* array with the specified shape and ordering.
     *
     * @param shape the shape of the array
     * @param ordering Fortran 'f' or C/C++ 'c' ordering.
     * @return the created array.
     */
    public static INDArray createUninitialized(long[] shape, char ordering) {
        if (shape.length == 0)
            return scalar(dataType(), 0.0);

        checkShapeValues(shape);
        return INSTANCE.createUninitialized(shape, ordering);
    }

    /**
     * @see #createUninitialized(long[], char)
     */
    public static INDArray createUninitializedDetached(int[] shape, char ordering) {
        if (shape.length == 0)
            return scalar(dataType(), 0.0);

        checkShapeValues(shape);
        return INSTANCE.createUninitializedDetached(shape, ordering);
    }

    // TODO: unused method. Only used by another unused method).
    /**
     *
     * @param shape
     * @param ordering
     * @return
     */
    public static INDArray createUninitializedDetached(long[] shape, char ordering) {
        if (shape.length == 0)
            return scalar(dataType(), 0.0);

        checkShapeValues(shape);
        return  INSTANCE.createUninitializedDetached(shape, ordering);
    }

    /**
     * @see #createUninitialized(long[])
     */
    public static INDArray createUninitialized(int... shape) {
        if(shape.length == 0)
            return Nd4j.scalar(dataType(), 0.0);
        checkShapeValues(shape);
        //ensure shapes that wind up being scalar end up with the write shape
        return createUninitialized(shape, Nd4j.order());
    }

    /**
     * Creates an *uninitialized* ndarray with the specified shape and default ordering.<br>
     * <b>NOTE</b>: The underlying memory (DataBuffer) will not be initialized. Don't use this unless you know what you are doing.
     *
     * @param shape the shape of the array
     * @return the instance
     */
    public static INDArray createUninitialized(long... shape) {
        checkShapeValues(shape);
        //ensure shapes that wind up being scalar end up with the write shape
        return createUninitialized(shape, Nd4j.order());
    }

    // TODO: unused method.
    /**
     * Cretes uninitialized INDArray detached from any (if any) workspace
     *
     * @param shape
     * @return
     */
    public static INDArray createUninitializedDetached(int... shape) {
        return createUninitializedDetached(shape, Nd4j.order());
    }

    // TODO: unused method.
    /**
     * Cretes uninitialized INDArray detached from any (if any) workspace
     *
     * @param shape
     * @return
     */
    public static INDArray createUninitializedDetached(long... shape) {
        return createUninitializedDetached(shape, Nd4j.order());
    }

    /**
     * @see #createUninitialized(long)
     */
    public static INDArray createUninitialized(int length) {
        return createUninitialized((long)length);
    }

    /**
     * This method creates an *uninitialized* ndarray of specified length and default ordering.
     *
     * PLEASE NOTE: Do not use this method unless you're 100% sure why you use it.
     *
     * @param length
     * @return
     */
    public static INDArray createUninitialized(long length) {
        long[] shape = new long[] {length};
        return INSTANCE.createUninitialized(shape, order());
    }

    /**
     * @see #createUninitialized(long)
     */
    public static INDArray createUninitializedDetached(int length) {
        long[] shape = new long[] {length};
        return INSTANCE.createUninitializedDetached(shape, order());
    }

    // TODO: unused method.
    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    public static INDArray create(double[] data, int[] shape, long offset) {
        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, offset);
        return ret;
    }

    // Todo: This method has zero documentation. Only used from tests.
    /**
     * @param data
     * @param columns
     * @param pointerB
     * @param pointerE
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCSR(double[] data, int[] columns, int[] pointerB, int[] pointerE, long[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCSR(data, columns, pointerB, pointerE, shape);

        return matrix;
    }

    // TODO: unused method.
    /**
     * @param data
     * @param columns
     * @param pointerB
     * @param pointerE
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCSR(float[] data, int[] columns, int[] pointerB, int[] pointerE, long[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCSR(data, columns, pointerB, pointerE, shape);

        return matrix;
    }

    // TODO: unused method.
    /**
     * @param data
     * @param columns
     * @param pointerB
     * @param pointerE
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCSR(DataBuffer data, int[] columns, int[] pointerB, int[] pointerE,
                                           long[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCSR(data, columns, pointerB, pointerE, shape);

        return matrix;
    }


    // Todo: Only used from tests.
    /**
     * @param data
     * @param indices
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(double[] data, int[][] indices, long[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(data, indices, shape);

        return matrix;
    }

    // TODO: unused method.
    /**
     * @param data
     * @param indices
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(float[] data, int[][] indices, long[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(data, indices, shape);

        return matrix;
    }

    // Todo: Only used from tests.
    /**
     * @param data
     * @param indices
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(double[] data, long[][] indices, long[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(data, indices, shape);

        return matrix;
    }

    // TODO: unused method.
    /**
     * @param data
     * @param indices
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(float[] data, long[][] indices, long[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(data, indices, shape);

        return matrix;
    }

    // TODO: unused method.
    /**
     * @param data
     * @param indices
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(DataBuffer data, DataBuffer indices, long[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(data, indices, shape);

        return matrix;
    }

    //TODO: Only used once, from an unused method.
    /**
     * @param values a DataBuffer with the sparse non-null values
     * @param indices a DataBuffer with the indexes of the values
     * @param sparseInformation a DataBuffer containing the sparse information (flags, offsets and hidden dimensions)
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(DataBuffer values, DataBuffer indices, DataBuffer sparseInformation,
                                           long[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(values, indices, sparseInformation, shape);
        return matrix;
    }

    // TODO: unused method.
    /**
     * @param values a DataBuffer with the sparse non-null values
     * @param indices a DataBuffer with the indexes of the values
     * @param sparseOffsets the sparse
     * @param flags an array that define the inactive dimension
     * @param shape
     * @param hiddenDimensions an array containing the position of the hidden dimensions
     * @param underlyingRank the rank of the original ndarray
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(DataBuffer values, DataBuffer indices, long[] sparseOffsets, int[] flags,
                                           long[] shape, int[] hiddenDimensions, int underlyingRank) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(values, indices, sparseOffsets, flags, hiddenDimensions,
                underlyingRank, shape);
        return matrix;
    }

    ////////////////////// OTHER ///////////////////////////////

    /**
     * Creates a 2D array with specified number of rows, columns initialized with zero.
     *
     * @param rows    number of rows.
     * @param columns number of columns.
     * @return the created array.
     */
    public static INDArray zeros(long rows, long columns) {
        return  INSTANCE.zeros(rows, columns);
    }

    /**
     * Creates a 1D array with the specified number of columns initialized with zero.
     *
     * @param columns number of columns.
     * @return the created array
     */
    public static INDArray zeros(int columns) {
        return INSTANCE.zeros(columns);
    }

    /**
     * Creates a 1D array with the specified data tyoe and number of columns initialized with zero.
     *
     * @param dataType data type.
     * @param columns number of columns.
     * @return the created array.
     */
    public static INDArray zeros(DataType dataType, int columns) {
        return INSTANCE.create(dataType, new long[]{columns}, 'c', Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Creates an array with the specified data tyoe and shape initialized with zero.
     *
     * @param dataType data type.
     * @param shape the shape of the array
     * @return the created array.
     */
    public static INDArray zeros(DataType dataType, @NonNull long... shape) {
        return INSTANCE.create(dataType, shape, 'c', Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray.
     * Some people may know this as np.full
     *
     * @param shape the shape of the array
     * @param value the value to assign
     * @return the created ndarray
     */
    public static INDArray valueArrayOf(int[] shape, double value) {
        if (shape.length == 0)
            return scalar(value);

        checkShapeValues(shape);
        return INSTANCE.valueArrayOf(shape, value);
    }

    /**
     * Creates an ndarray with the specified value as the only value in the FLOAT32 datatype NDArray.
     * Equivalent to Numpy's np.full
     *
     * @param shape the shape of the array
     * @param value the value to assign
     * @return the created ndarray
     */
    public static INDArray valueArrayOf(long[] shape, float value) {
        return valueArrayOf(shape, (double)value, DataType.FLOAT);
    }

    /**
     * Creates an ndarray with the specified value as the only value in the INTEGER datatype NDArray.
     * Equivalent to Numpy's np.full
     *
     * @param shape the shape of the array
     * @param value the value to assign
     * @return the created ndarray
     */
    public static INDArray valueArrayOf(long[] shape, int value) {
        return valueArrayOf(shape, (double)value, DataType.INT);
    }

    /**
     * @see #valueArrayOf(long[], double, DataType)
     */
    public static INDArray valueArrayOf(long[] shape, double value) {
        if (shape.length == 0)
            return scalar(value);

        checkShapeValues(shape);
        return INSTANCE.valueArrayOf(shape, value);
    }

    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray.
     * Some people may know this as np.full
     *
     * @param shape the shape of the array
     * @param value the value to assign
     * @param type data type
     * @return the created ndarray
     */
    public static INDArray valueArrayOf(long[] shape, double value, DataType type) {
        if (shape.length == 0)
            return scalar(type, value);

        checkShapeValues(shape);

        INDArray ret = createUninitialized(type, shape);
        ret.assign(value);
        return ret;
    }

    /**
     * @see #valueArrayOf(long[], double, DataType)
     */
    public static INDArray valueArrayOf(long[] shape, long value, DataType type) {
        if (shape.length == 0)
            return scalar(type, value);

        checkShapeValues(shape);

        INDArray ret = createUninitialized(type, shape);
        ret.assign(value);
        return ret;
    }

    /**
     * Creates a row vector ndarray with the specified value
     * as the  only value in the ndarray
     *
     * Some people may know this as np.full
     *
     * @param num   number of columns
     * @param value the value to assign
     * @return the created ndarray
     */
    public static INDArray valueArrayOf(long num, double value) {
        INDArray ret = INSTANCE.valueArrayOf(new long[] {num}, value);
        return ret;
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * Some people may know this as np.full
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param value   the value to assign
     * @return the created ndarray
     */
    public static INDArray valueArrayOf(long rows, long columns, double value) {
        INDArray ret = INSTANCE.valueArrayOf(rows, columns, value);
        return ret;
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray ones(int rows, int columns) {
        INDArray ret = INSTANCE.ones(rows, columns);
        return ret;
    }

    /**
     * Create a 2D array with the given rows, columns and data type initialised with ones.
     *
     * @param dataType data type
     * @param rows rows of the new array.
     * @param columns columns of the new arrau.
     * @return the created array
     */
    public static INDArray ones(DataType dataType, int rows, int columns) {
        INDArray ret = INSTANCE.createUninitialized(dataType, new long[]{rows, columns}, Nd4j.order(), Nd4j.getMemoryManager().getCurrentWorkspace());
        ret.assign(1);
        return ret;
    }

    /**
     * Empty like
     *
     * @param arr the array to create the ones like
     * @return ones in the shape of the given array
     */
    public static INDArray zerosLike(INDArray arr) {
        return zeros(arr.dataType(), arr.shape());
    }

    /**
     * Ones like
     *
     * @param arr the array to create the ones like
     * @return ones in the shape of the given array
     */
    public static INDArray onesLike(INDArray arr) {
        return ones(arr.dataType(), arr.shape());
    }

    /**
     * Creates an array with the specified datatype and shape, with values all set to 1
     *
     * @param shape Shape fo the array
     * @return the created ndarray
     */
    public static INDArray ones(DataType dataType, @NonNull long... shape) {
        if(shape.length == 0)
            return Nd4j.scalar(dataType, 1.0);
        INDArray ret = INSTANCE.createUninitialized(dataType, shape, Nd4j.order(), Nd4j.getMemoryManager().getCurrentWorkspace());
        ret.assign(1);
        return ret;
    }

    /**
     * Concatenates two matrices horizontally. Matrices must have identical
     * numbers of rows.
     *
     * @param arrs the first matrix to concat
     */
    public static INDArray hstack(@NonNull INDArray... arrs) {
        INDArray ret = INSTANCE.hstack(arrs);
        return ret;
    }

    /**
     * Concatenates two matrices horizontally. Matrices must have identical
     * numbers of rows.
     *
     * @param arrs the first matrix to concat
     */
    public static INDArray hstack(Collection<INDArray> arrs) {
        INDArray[] arrays = arrs.toArray(new INDArray[0]);
        return  INSTANCE.hstack(arrays);
    }

    /**
     * Concatenates two matrices vertically. Matrices must have identical numbers of columns.<br>
     * Note that for vstack on rank 1 arrays, this is equivalent to {@link Nd4j#pile(INDArray...)}. Example: vstack([3],[3]) -> [2,3]
     *
     * @param arrs Arrays to vstack
     */
    public static INDArray vstack(@NonNull INDArray... arrs) {
        Preconditions.checkState(arrs != null && arrs.length > 0, "No input specified to vstack (null or length 0)");
        if(arrs[0].rank() == 1){
            //Edge case: vstack rank 1 arrays - gives rank 2... vstack([3],[3]) -> [2,3]
            return pile(arrs);
        }
        return  INSTANCE.vstack(arrs);
    }

    /**
     * Concatenates two matrices vertically. Matrices must have identical numbers of columns.<br>
     * Note that for vstack on rank 1 arrays, this is equivalent to {@link Nd4j#pile(INDArray...)}. Example: vstack([3],[3]) -> [2,3]
     *
     * @param arrs Arrays to vstack
     */
    public static INDArray vstack(Collection<INDArray> arrs) {
        INDArray[] arrays = arrs.toArray(new INDArray[0]);
        return vstack(arrays);
    }

    // TODO: unused method
    /**
     * This method averages input arrays, and returns averaged array.
     * On top of that, averaged array is propagated to all input arrays
     *
     * @param arrays
     * @return
     */
    public static INDArray averageAndPropagate(INDArray target, INDArray[] arrays) {
        return  INSTANCE.average(target, arrays);
    }

    /**
     * This method averages input arrays, and returns averaged array.
     * On top of that, averaged array is propagated to all input arrays
     *
     * @param arrays
     * @return
     */
    public static INDArray averageAndPropagate(INDArray[] arrays) {
        return INSTANCE.average(arrays);
    }


    /**
     * This method averages input arrays, and returns averaged array.
     * On top of that, averaged array is propagated to all input arrays
     *
     * @param arrays
     * @return
     */
    public static INDArray averageAndPropagate(Collection<INDArray> arrays) {
        return  INSTANCE.average(arrays);
    }

    /**
     * This method averages input arrays, and returns averaged array.
     * On top of that, averaged array is propagated to all input arrays
     *
     * @param arrays
     * @return
     */
    public static INDArray averageAndPropagate(INDArray target, Collection<INDArray> arrays) {
        return INSTANCE.average(target, arrays);
    }



    /**
     * Reshapes an ndarray to remove leading 1s
     * @param toStrip the ndarray to newShapeNoCopy
     * @return the reshaped ndarray
     */
    public static INDArray stripOnes(INDArray toStrip) {
        if (toStrip.isVector())
            return toStrip;
        else {
            long[] shape = Shape.squeeze(toStrip.shape());
            return toStrip.reshape(shape);
        }
    }

    // TODO: unused method.
    /**
     * This method sums given arrays and stores them to a new target array
     *
     * @param arrays
     * @return
     */
    public static INDArray accumulate(Collection<INDArray> arrays) {
        if (arrays == null|| arrays.size() == 0)
            throw new ND4JIllegalStateException("Input for accumulation is null or empty");

        return accumulate(arrays.toArray(new INDArray[0]));
    }

    /**
     * This method sums given arrays and stores them to a new array
     *
     * @param arrays
     * @return
     */
    public static INDArray accumulate(@NonNull INDArray... arrays) {
        if (arrays == null|| arrays.length == 0)
            throw new ND4JIllegalStateException("Input for accumulation is null or empty");

        return accumulate(Nd4j.create(arrays[0].shape(), arrays[0].ordering()), arrays);
    }

    /**
     * This method sums given arrays and stores them to a given target array
     *
     * @param target
     * @param arrays
     * @return
     */
    public static INDArray accumulate(INDArray target, Collection<INDArray> arrays) {
        return accumulate(target, arrays.toArray(new INDArray[0]));
    }

    /**
     * This method sums given arrays and stores them to a given target array
     *
     * @param target
     * @param arrays
     * @return
     */
    public static INDArray accumulate(INDArray target, INDArray[] arrays) {
        if (arrays == null|| arrays.length == 0)
            return target;

        return factory().accumulate(target, arrays);
    }

    /**
     * This method produces concatenated array, that consist from tensors, fetched from source array, against some dimension and specified indexes
     *
     * @param source source tensor
     * @param sourceDimension dimension of source tensor
     * @param indexes indexes from source array
     * @return
     */
    public static INDArray pullRows(INDArray source, int sourceDimension, @NonNull int... indexes) {
        return pullRows(source, sourceDimension, indexes, Nd4j.order());
    }

    /**
     * This method produces concatenated array,
     * that consist from tensors,
     * fetched from source array,
     * against some dimension and specified indexes
     *
     * @param source source tensor
     * @param sourceDimension dimension of source tensor
     * @param indexes indexes from source array
     * @return
     */
    public static INDArray pullRows(INDArray source, int sourceDimension, int[] indexes, char order) {
        if (sourceDimension >= source.rank())
            throw new IllegalStateException("Source dimension can't be higher the rank of source tensor");

        if (indexes == null || indexes.length == 0)
            throw new IllegalStateException("Indexes shouldn't be empty");

        if (order != 'c' && order != 'f' && order != 'a')
            throw new IllegalStateException("Unknown order being passed in [" + order + "]");

        for (int idx : indexes) {
            if (idx < 0 || idx >= source.shape()[source.rank() - sourceDimension - 1]) {
                throw new IllegalStateException(
                        "Index can't be < 0 and >= " + source.shape()[source.rank() - sourceDimension - 1]);
            }
        }

        Preconditions.checkArgument(source.rank() > 1, "pullRows() can't operate on 0D/1D arrays");

        INDArray ret = INSTANCE.pullRows(source, sourceDimension, indexes, order);
        return ret;
    }

    /**
     * This method produces concatenated array, that consist from tensors, fetched from source array, against some
     * dimension and specified indexes.
     * The concatenated arrays are placed in the specified array.
     *
     * @param source source tensor
     * @param destination Destination tensor (result will be placed here)
     * @param sourceDimension dimension of source tensor
     * @param indexes indexes from source array
     * @return Destination array with specified tensors
     */
    public static INDArray pullRows(INDArray source, INDArray destination, int sourceDimension, @NonNull int... indexes){
        if (sourceDimension >= source.rank())
            throw new IllegalStateException("Source dimension can't be higher the rank of source tensor");

        if (indexes == null || indexes.length == 0)
            throw new IllegalStateException("Indexes shouldn't be empty");

        for (int idx : indexes) {
            if (idx < 0 || idx >= source.shape()[source.rank() - sourceDimension - 1]) {
                throw new IllegalStateException(
                        "Index can't be < 0 and >= " + source.shape()[source.rank() - sourceDimension - 1]);
            }
        }

        Preconditions.checkArgument(source.rank() > 1, "pullRows() can't operate on 0D/1D arrays");

        INDArray ret = INSTANCE.pullRows(source, destination, sourceDimension, indexes);
        return ret;
    }

    /**
     * Stack a set of N SDVariables of rank X into one rank X+1 variable.
     * If inputs have shape [a,b,c] then output has shape:<br>
     * axis = 0: [N,a,b,c]<br>
     * axis = 1: [a,N,b,c]<br>
     * axis = 2: [a,b,N,c]<br>
     * axis = 3: [a,b,c,N]<br>
     *
     * @param axis   Axis to stack on
     * @param values Input variables to stack. Must have the same shape for all inputs
     * @return Output array
     * @see #concat(int, INDArray...)
     */
    public static INDArray stack(int axis, @NonNull INDArray... values){
        Preconditions.checkArgument(values != null && values.length > 0, "No inputs: %s", values);
        Preconditions.checkState(axis >= -(values[0].rank()+1) && axis < values[0].rank()+1, "Invalid axis: must be between " +
                "%s (inclusive) and %s (exclusive) for rank %s input, got %s", -(values[0].rank()+1), values[0].rank()+1,
                values[0].rank(), axis);

        Stack stack = new Stack(values, null, axis);
        INDArray[] outputArrays = Nd4j.getExecutioner().allocateOutputArrays(stack);
        stack.addOutputArgument(outputArrays);
        Nd4j.getExecutioner().execAndReturn(stack);
        return outputArrays[0];
    }

    /**
     * Concatneate ndarrays along a dimension
     *
     * @param dimension the dimension to concatneate along
     * @param toConcat  the ndarrays to concat
     * @return the merged ndarrays with an output shape of
     * the ndarray shapes save the dimension shape specified
     * which is then the sum of the sizes along that dimension
     */
    public static INDArray concat(int dimension, @NonNull INDArray... toConcat) {
        if(dimension < 0) {
            dimension += toConcat[0].rank();
        }

        return INSTANCE.concat(dimension, toConcat);
    }

    /**
     * Concatneate ndarrays along a dimension
     *
     * PLEASE NOTE: This method is special for GPU backend, it works on HOST side only.
     *
     * @param dimension
     * @param toConcat
     * @return
     */
    public static INDArray specialConcat(int dimension, @NonNull INDArray... toConcat) {
        INDArray ret = INSTANCE.specialConcat(dimension, toConcat);
        return ret;
    }

    /**
     * Create an ndarray of zeros
     *
     * @param shape the shape of the array
     * @return an ndarray with ones filled in
     */
    public static INDArray zeros(int[] shape, char order) {
        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, order);
        return ret;
    }

    /**
     * @see #zeros(int[] , char)
     */
    public static INDArray zeros(long[] shape, char order) {
        checkShapeValues(shape);
        return  INSTANCE.create(shape, order);
    }

    /**
     * Create an ndarray of zeros
     *
     * @param shape the shape of the array
     * @return an ndarray with ones filled in
     */
    public static INDArray zeros(@NonNull int... shape) {
        return Nd4j.create(shape);
    }


    /**
     * Create an ndarray of zeros
     *
     * @param shape the shape of the array
     * @return an ndarray with ones filled in
     */
    public static INDArray zeros(@NonNull long... shape) {
        return Nd4j.create(shape);
    }

    /**
     * Create an ndarray of ones
     *
     * @param shape the shape of the array
     * @return an ndarray with ones filled in
     */
    public static INDArray ones(@NonNull int... shape) {
        if(shape.length == 0)
            return Nd4j.scalar(dataType(), 1.0);
        return INSTANCE.ones(shape);
    }


    /**
     * @see #ones(int... shape)
     */
    public static INDArray ones(@NonNull long... shape) {
        if(shape.length == 0)
            return Nd4j.scalar(dataType(), 1.0);
        checkShapeValues(shape);
        return INSTANCE.ones(shape);
    }

    /**
     * Create a scalar ndarray with the specified value
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    public static INDArray scalar(Number value) {
        return INSTANCE.scalar(value);
    }

    /**
     * Create a scalar ndarray with the specified value and datatype
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    public static INDArray scalar(DataType dataType, Number value) {
        return INSTANCE.trueScalar(dataType, value);
    }

    /**
     * Create a scalar nd array with the specified value
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    public static INDArray scalar(double value) {
        return scalar(DataType.DOUBLE, value);
    }

    /**
     * Create a scalar NDArray with the specified value and FLOAT datatype
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    public static INDArray scalar(float value) {
        return scalar(DataType.FLOAT, value);
    }

    /**
     * Create a scalar NDArray with the specified value and BOOLEAN datatype
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    public static INDArray scalar(boolean value) {
        return INSTANCE.trueScalar(DataType.BOOL, value ? 1 : 0);
    }

    /**
     * Create a scalar NDArray with the specified value and INT datatype
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    public static INDArray scalar(int value) {
        return scalar(DataType.INT, value);
    }

    /**
     * Create a scalar NDArray with the specified value and LONG datatype
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    public static INDArray scalar(long value) {
        return scalar(DataType.LONG, value);
    }

    /**
     * Get the strides for the given order and shape
     *
     * @param shape the shape of the array
     * @param order the order to getScalar the strides for
     * @return the strides for the given shape and order
     */
    public static int[] getStrides(int[] shape, char order) {
        if (order == NDArrayFactory.FORTRAN)
            return ArrayUtil.calcStridesFortran(shape);
        return ArrayUtil.calcStrides(shape);
    }

    public static long[] getStrides(long[] shape, char order) {
        if (order == NDArrayFactory.FORTRAN)
            return ArrayUtil.calcStridesFortran(shape);
        return ArrayUtil.calcStrides(shape);
    }

    /**
     * Get the strides based on the shape
     * and NDArrays.order()
     *
     * @param shape the shape of the array
     * @return the strides for the given shape
     * and order specified by NDArrays.order()
     */
    public static int[] getStrides(@NonNull int... shape) {
        return getStrides(shape, Nd4j.order());
    }

    /**
     * Get the strides based on the shape
     * and NDArrays.order()
     *
     * @param shape the shape of the array
     * @return the strides for the given shape
     * and order specified by NDArrays.order()
     */
    public static long[] getStrides(@NonNull long... shape) {
        return getStrides(shape, Nd4j.order());
    }

    /**
     * An alias for repmat
     *
     * @param tile   the ndarray to tile
     * @param repeat the shape to repeat
     * @return the tiled ndarray
     */
    public static INDArray tile(INDArray tile, @NonNull int... repeat) {
        int d = repeat.length;
        long[] shape = ArrayUtil.copy(tile.shape());
        long n = Math.max(tile.length(), 1);
        if (d < tile.rank()) {
            repeat = Ints.concat(ArrayUtil.nTimes(tile.rank() - d, 1), repeat);
        }
        for (int i = 0; i < shape.length; i++) {
            if (repeat[i] != 1) {
                tile = tile.reshape(-1, n).repeat(0, repeat[i]);
            }

            long in = shape[i];
            long nOut = in * repeat[i];
            shape[i] = nOut;
            n /= Math.max(in, 1);

        }

        return tile.reshape(shape);
    }

    /**
     * Get the strides for the given order and shape
     *
     * @param shape the shape of the array
     * @param order the order to getScalar the strides for
     * @return the strides for the given shape and order
     */
    public static int[] getComplexStrides(int[] shape, char order) {
        if (order == NDArrayFactory.FORTRAN)
            return ArrayUtil.calcStridesFortran(shape, 2);
        return ArrayUtil.calcStrides(shape, 2);
    }

    public static long[] getComplexStrides(long[] shape, char order) {
        if (order == NDArrayFactory.FORTRAN)
            return ArrayUtil.calcStridesFortran(shape, 2);
        return ArrayUtil.calcStrides(shape, 2);
    }

    /**
     * Get the strides based on the shape
     * and NDArrays.order()
     *
     * @param shape the shape of the array
     * @return the strides for the given shape
     * and order specified by NDArrays.order()
     */
    public static int[] getComplexStrides(@NonNull int... shape) {
        return getComplexStrides(shape, Nd4j.order());
    }

    public static long[] getComplexStrides(@NonNull long... shape) {
        return getComplexStrides(shape, Nd4j.order());
    }

    /**
     * Initializes nd4j
     */
    public synchronized void initContext() {
        try {
            defaultFloatingPointDataType = new AtomicReference<>();
            defaultFloatingPointDataType.set(DataType.FLOAT);
            Nd4jBackend backend = Nd4jBackend.load();
            initWithBackend(backend);
        } catch (NoAvailableBackendException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Initialize with the specific backend
     * @param backend the backend to initialize with
     */
    public void initWithBackend(Nd4jBackend backend) {
        VersionCheck.checkVersions();

        try {
            if (System.getProperties().getProperty("backends") != null
                    && !System.getProperties().getProperty("backends").contains(backend.getClass().getName())) {
                return;
            }

            if (!isSupportedPlatform()) {
                showAttractiveMessage(getMessageForUnsupportedPlatform());
                return;
            }

            Nd4j.backend = backend;
            updateNd4jContext();
            props = Nd4jContext.getInstance().getConf();
            PropertyParser pp = new PropertyParser(props);

            String otherDtype = pp.toString(ND4JSystemProperties.DTYPE);
            dtype = otherDtype.equalsIgnoreCase("float") ? DataType.FLOAT
                    : otherDtype.equalsIgnoreCase("half") ? DataType.HALF : DataType.DOUBLE;

            if (dtype == DataType.HALF && backend.getClass().getName().equals("CpuBackend")) {
                showAttractiveMessage(getMessageForNativeHalfPrecision());
            }

            if (Nd4j.dataType() != dtype){
                DataTypeUtil.setDTypeForContext(dtype);
            }

            compressDebug = pp.toBoolean(COMPRESSION_DEBUG);
            ORDER = pp.toChar(ORDER_KEY, NDArrayFactory.C);

            affinityManagerClazz = (Class<? extends BasicAffinityManager>) Class
                    .forName(pp.toString(AFFINITY_MANAGER));
            affinityManager = affinityManagerClazz.newInstance();
            ndArrayFactoryClazz = (Class<? extends NDArrayFactory>) Class.forName(
                    pp.toString(NDARRAY_FACTORY_CLASS));
            sparseNDArrayClazz = (Class<? extends NDArrayFactory>) Class.forName(
                    pp.toString(SPARSE_NDARRAY_FACTORY_CLASS));
            convolutionInstanceClazz = (Class<? extends ConvolutionInstance>) Class
                    .forName(pp.toString(CONVOLUTION_OPS, DefaultConvolutionInstance.class.getName()));
            String defaultName = pp.toString(DATA_BUFFER_OPS, DefaultDataBufferFactory.class.getName());
            dataBufferFactoryClazz = (Class<? extends DataBufferFactory>) Class
                    .forName(pp.toString(DATA_BUFFER_OPS, defaultName));
            shapeInfoProviderClazz = (Class<? extends BaseShapeInfoProvider>) Class
                    .forName(pp.toString(SHAPEINFO_PROVIDER));
            sparseInfoProviderClazz = (Class<? extends BaseSparseInfoProvider>) Class.forName(
                    pp.toString(SPARSEINFO_PROVIDER));

            constantProviderClazz = (Class<? extends BasicConstantHandler>) Class
                    .forName(pp.toString(CONSTANT_PROVIDER));

            memoryManagerClazz = (Class<? extends BasicMemoryManager>) Class
                    .forName(pp.toString(MEMORY_MANAGER));

            allowsOrder = backend.allowsOrder();
            String rand = pp.toString(RANDOM_PROVIDER, DefaultRandom.class.getName());
            randomClazz = (Class<? extends org.nd4j.linalg.api.rng.Random>) Class.forName(rand);
            randomFactory = new RandomFactory(randomClazz);

            workspaceManagerClazz = (Class<? extends MemoryWorkspaceManager>) Class
                    .forName(pp.toString(WORKSPACE_MANAGER));

            blasWrapperClazz = (Class<? extends BlasWrapper>) Class
                    .forName(pp.toString(BLAS_OPS));
            sparseBlasWrapperClazz = (Class<? extends BlasWrapper>) Class
                    .forName(pp.toString(SPARSE_BLAS_OPS));
            String clazzName = pp.toString(DISTRIBUTION, DefaultDistributionFactory.class.getName());
            distributionFactoryClazz = (Class<? extends DistributionFactory>) Class.forName(clazzName);


            memoryManager = memoryManagerClazz.newInstance();
            constantHandler = constantProviderClazz.newInstance();
            shapeInfoProvider = shapeInfoProviderClazz.newInstance();
            sparseInfoProvider = sparseInfoProviderClazz.newInstance();
            workspaceManager = workspaceManagerClazz.newInstance();

            opExecutionerClazz = (Class<? extends OpExecutioner>) Class
                    .forName(pp.toString(OP_EXECUTIONER, DefaultOpExecutioner.class.getName()));

            OP_EXECUTIONER_INSTANCE = opExecutionerClazz.newInstance();
            Constructor c2 = ndArrayFactoryClazz.getConstructor(DataType.class, char.class);
            INSTANCE = (NDArrayFactory) c2.newInstance(dtype, ORDER);
            SPARSE_INSTANCE = sparseNDArrayClazz.newInstance();
            CONVOLUTION_INSTANCE = convolutionInstanceClazz.newInstance();
            BLAS_WRAPPER_INSTANCE = blasWrapperClazz.newInstance();
            SPARSE_BLAS_WRAPPER_INSTANCE = sparseBlasWrapperClazz.newInstance();
            DATA_BUFFER_FACTORY_INSTANCE = dataBufferFactoryClazz.newInstance();

            DISTRIBUTION_FACTORY = distributionFactoryClazz.newInstance();

            if (isFallback()) {
                fallbackMode.set(true);
                showAttractiveMessage(getMessageForFallback());
            } else {
                fallbackMode.set(false);
            }

            String logInitProperty = System.getProperty(ND4JSystemProperties.LOG_INITIALIZATION, "true");
            if(Boolean.parseBoolean(logInitProperty)) {
                OP_EXECUTIONER_INSTANCE.printEnvironmentInformation();
            }

            val actions = ServiceLoader.load(EnvironmentalAction.class);
            val mappedActions = new HashMap<String, EnvironmentalAction>();
            for (val a: actions) {
                if (!mappedActions.containsKey(a.targetVariable()))
                    mappedActions.put(a.targetVariable(), a);
            }

            for (val e: mappedActions.keySet()) {
                val action = mappedActions.get(e);
                val value = System.getenv(e);
                if (value != null) {
                    try {
                        action.process(value);
                    } catch (Exception e2) {
                        logger.info("Failed to process env variable [" + e + "], got exception: " + e2.toString());
                    }
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    private static boolean isSupportedPlatform() {
        return (System.getProperty("java.vm.name").equalsIgnoreCase("Dalvik")
                || System.getProperty("os.arch").toLowerCase().startsWith("arm")
                || System.getProperty("sun.arch.data.model").equals("64"));
    }

    private static void showAttractiveMessage(String... strings) {
        System.out.println(attract(strings));
    }

    private static String attract(String... strings) {
        String delimiter = "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
        String shift = "                 ";
        StringBuilder sb = new StringBuilder().append(delimiter).append("\n").append("\n");
        for (String s : strings) {
            sb.append(shift).append(s).append("\n");
        }
        sb.append("\n").append(delimiter).append("\n");
        return sb.toString();
    }

    private static String[] getMessageForUnsupportedPlatform() {
        return new String[] {"Unfortunately you can't use DL4j/ND4j on 32-bit x86 JVM",
                "Please, consider running this on 64-bit JVM instead"};
    }

    private static String[] getMessageForFallback() {
        return new String[] {"ND4J_FALLBACK environment variable is detected!", "Performance will be slightly reduced"};
    }

    private String[] getMessageForNativeHalfPrecision() {
        return new String[] {"Half-precision data opType isn't support for nd4j-native",
                "Please, consider using FLOAT or DOUBLE data opType instead"};
    }

    private void updateNd4jContext() throws IOException {
        try (InputStream is = backend.getConfigurationResource().getInputStream()) {
            Nd4jContext.getInstance().updateProperties(is);
        }
    }

    private boolean isFallback() {
        String fallback = System.getenv(ND4JEnvironmentVars.ND4J_FALLBACK);
        if (fallback == null) {
            return false;
        }
        return (fallback.equalsIgnoreCase("true") || fallback.equalsIgnoreCase("1"));
    }

    /**
     *
     * @return
     */
    public static ShapeInfoProvider getShapeInfoProvider() {
        return shapeInfoProvider;
    }

    /**
     *
     * @return
     */
    public static SparseInfoProvider getSparseInfoProvider() {
        return sparseInfoProvider;
    }

    /**
     *
     * @return
     */
    public static ConstantHandler getConstantHandler() {
        return constantHandler;
    }

    /**
     *
     * @return
     */
    public static AffinityManager getAffinityManager() {
        return affinityManager;
    }

    /**
     *
     * @return
     */
    public static NDArrayFactory getNDArrayFactory() {
        return INSTANCE;
    }

    /**
     * This method returns BasicNDArrayCompressor instance,
     * suitable for NDArray compression/decompression
     * at runtime
     *
     * @return
     */
    public static BasicNDArrayCompressor getCompressor() {
        return BasicNDArrayCompressor.getInstance();
    }

    /**
     * This method returns backend-specific MemoryManager implementation, for low-level memory management
     * @return
     */
    public static MemoryManager getMemoryManager() {
        return memoryManager;
    }

    public static INDArray typeConversion(INDArray array, DataTypeEx targetType) {
        return null;
    }

    /**
     * This method returns sizeOf(currentDataType), in bytes
     *
     * @return number of bytes per element
     * @deprecated Use DataType.width()
     */
    @Deprecated
    public static int sizeOfDataType() {
        return sizeOfDataType(Nd4j.dataType());
    }

    /**
     * This method returns size of element for specified dataType, in bytes
     *
     * @param dtype number of bytes per element
     * @return
     */
    public static int sizeOfDataType(DataType dtype) {
        switch (dtype) {
            case BYTE:
            case BOOL:
            case UBYTE:
                return 1;
            case UINT16:
            case SHORT:
            case BFLOAT16:
            case HALF:
                return 2;
            case UINT32:
            case FLOAT:
            case INT:
                return 4;
            case UINT64:
            case LONG:
            case DOUBLE:
                return 8;
            default:
                throw new ND4JIllegalStateException("Unsupported data type: [" + dtype +"]" );
        }
    }

    /**
     * This method enables fallback to safe-mode for specific operations. Use of this method will reduce performance.
     * Currently supported operations are:
     *  1) CPU GEMM
     *
     * PLEASE NOTE: Do not use this method, unless you have too.
     *
     * @param reallyEnable
     */
    public static void enableFallbackMode(boolean reallyEnable) {
        fallbackMode.set(reallyEnable);
    }

    /**
     * This method checks, if fallback mode was enabled.
     *
     * @return
     */
    public static boolean isFallbackModeEnabled() {
        return fallbackMode.get();
    }

    /**
     * This method returns WorkspaceManager implementation to be used within this JVM process
     *
     * @return
     */
    public static MemoryWorkspaceManager getWorkspaceManager() {
        return workspaceManager;
    }

    /**
     * This method stacks vertically examples with the same shape, increasing result dimensionality. I.e. if you provide bunch of 3D tensors, output will be 4D tensor. Alignment is always applied to axis 0.
     *
     * @return
     */
    public static INDArray pile(@NonNull INDArray... arrays) {
        // if we have vectors as input, it's just vstack use case

        long[] shape = arrays[0].shape();
        long[] newShape = ArrayUtils.add(shape, 0, 1);

        boolean shouldReshape = true;

        List<INDArray> reshaped = new ArrayList<>();
        for(INDArray array: arrays) {
            if (!shouldReshape)
                reshaped.add(array);
            else
                reshaped.add(array.reshape(array.ordering(), newShape));
        }

        return Nd4j.vstack(reshaped);
    }

    /**
     * This method stacks vertically examples with the same shape, increasing result dimensionality. I.e. if you provide bunch of 3D tensors, output will be 4D tensor. Alignment is always applied to axis 0.
     *
     * @return
     */
    public static INDArray pile(@NonNull Collection<INDArray> arrays) {
        return pile(arrays.toArray(new INDArray[0]));
    }

    /**
     * This method does the opposite to pile/vstack/hstack - it returns independent TAD copies along given dimensions
     *
     * @param tensor
     * @param dimensions
     * @return
     */
    public static INDArray[] tear(INDArray tensor, @NonNull int... dimensions) {
        if (dimensions.length >= tensor.rank())
            throw new ND4JIllegalStateException("Target dimensions number should be less tensor rank");

        for (int e = 0; e < dimensions.length; e++)
            if (dimensions[e] < 0)
                throw new ND4JIllegalStateException("Target dimensions can't have negative values");

        return factory().tear(tensor, dimensions);
    }


    /**
     *   Upper triangle of an array.

     Return a copy of a matrix with the elements below the `k`-th diagonal
     zeroed.

     Please refer to the documentation for `tril` for further details.

     * @param m
     * @param k
     * @return
     */
    public static INDArray triu(INDArray m,int k) {
        /**
         *     """
         Upper triangle of an array.

         Return a copy of a matrix with the elements below the `k`-th diagonal
         zeroed.

         Please refer to the documentation for `tril` for further details.

         See Also
         --------
         tril : lower triangle of an array

         Examples
         --------
         >>> np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
         array([[ 1,  2,  3],
         [ 4,  5,  6],
         [ 0,  8,  9],
         [ 0,  0, 12]])

         """
         m = asanyarray(m)
         mask = tri(*m.shape[-2:], k=k-1, dtype=bool)

         return where(mask, zeros(1, m.dtype), m)
         */

        //INDArray mask = tri(m.size(-2),1);
        /**
         * Find a way to apply choose with an existing condition array.
         * (This appears to be the select op in libnd4j)
         */
        /*
        Select select = new Select(new INDArray[]{mask,Nd4j.zeros(1),m},new INDArray[]{Nd4j.zerosLike(m)});
        Nd4j.getExecutioner().exec(select);
        return select.getOutputArgument(0);
        */

        INDArray result = Nd4j.createUninitialized(m.shape());

        val op = DynamicCustomOp.builder("triu")
                .addInputs(m)
                .addOutputs(result)
                .addIntegerArguments(k)
                .build();

        Nd4j.getExecutioner().execAndReturn(op);

        return result;
    }


    /**
     *
     * @param n
     * @return
     */
    public static INDArray tri(int n) {
        return tri(n,n,0);
    }

    /**
     *
     * @param n
     * @param k
     * @return
     */
    public static INDArray tri(int n,int k) {
        return tri(n,n,k);
    }

    /**
     * Like the scipy function tri.
     * From the scipy documentation:
     *  An array with ones at and below the given diagonal and zeros elsewhere.
     * @param n number of rows in the array
     * @param m number of columns in the array ( can be just equal to n)
     * @param k    The sub-diagonal at and below which the array is filled.
    `k` = 0 is the main diagonal, while `k` < 0 is below it,
    and `k` > 0 is above.  The default is 0.
     * @return
     */
    public static INDArray tri(int n,int m,int k) {
        /*
        INDArray mRet = Transforms.greaterThanOrEqual(arange(n),arange(-k,m - k));

        return mRet;
        */

        INDArray ret = Nd4j.createUninitialized(n, m);

        val op = DynamicCustomOp.builder("tri")
                .addIntegerArguments(n, m, k)
                .addOutputs(ret)
                .build();

        Nd4j.getExecutioner().execAndReturn(op);

        return ret;
    }

    /**
     * Similar to numpy.where operation.
     * Supports two modes of operation:<br>
     * (a) condition array only is provided: returns N 1d arrays of the indices where "condition" values are non-zero.
     * Specifically, each output out has shape [numNonZero(condition)], such that in[out[0], ..., out[n-1]] is non-zero<br>
     * (b) all 3 arrays are provided: returns {@code out[i] = (condition[i] != 0 ? x[i] : y[i])}<br>
     * @param condition Condition array
     * @param x         X array. If null, y must be null also.
     * @param y         Y array. If null, x must be null also
     * @return Either the indices where condition is non-zero (if x and y are null), or values from x/y depending on
     * value of condition
     */
    public static INDArray[] where(INDArray condition, INDArray x, INDArray y){
        Preconditions.checkState((x == null && y == null) || (x != null && y != null), "Both X and Y must be" +
                "null, or neither must be null");
        INDArray out;
        DynamicCustomOp.DynamicCustomOpsBuilder op = DynamicCustomOp.builder("where_np");
        List<LongShapeDescriptor> outShapes;
        if(x == null){
            //First case: condition only...
            op.addInputs(condition);
        } else {
            if(!x.equalShapes(y) || !x.equalShapes(condition)){
                Preconditions.throwStateEx("Shapes must be equal: condition=%s, x=%s, y=%s", condition.shape(), x.shape(), y.shape());
            }
            op.addInputs(condition, x, y);
        }
        DynamicCustomOp o = op.build();
        outShapes = Nd4j.getExecutioner().calculateOutputShape(o);
        INDArray[] outputs = new INDArray[outShapes.size()];

        if(x == null && (outShapes.get(0) == null || outShapes.get(0).getShape().length == 0 || outShapes.get(0).getShape()[0] == 0)){
            //Empty: no conditions match
            for( int i=0; i<outputs.length; i++ ){
                outputs[i]  = Nd4j.empty();
            }
            return outputs;
        }

        for(int i=0; i<outputs.length; i++){
            outputs[i] = Nd4j.create(outShapes.get(i), false);
        }
        op.addOutputs(outputs);

        Nd4j.getExecutioner().execAndReturn(op.build());
        return outputs;
    }


    /**
     * Write an {@link INDArray} to a {@link File} in Numpy .npy format, which can then be loaded with numpy.load
     * @param arr the array to write in Numpy .npy format
     * @param file the file to write to
     * @throws IOException if an error occurs when writing the file
     */
    public static void writeAsNumpy(INDArray arr, File file) throws IOException {
        writeAsNumpy(arr, new FileOutputStream(file));
    }


    /**
     * Converts an {@link INDArray} to a numpy struct.
     * @param arr the array to convert
     * @return a pointer to the numpy struct
     */
    public static Pointer convertToNumpy(INDArray arr)  {
        return INSTANCE.convertToNumpy(arr);
    }


    /**
     * Writes an array to an output stream
     * @param arr the array to write
     * @param writeTo the output stream to write to
     * @throws IOException
     */
    public static void writeAsNumpy(INDArray arr, OutputStream writeTo) throws IOException {
        try(BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(writeTo)) {
            Pointer asNumpy = convertToNumpy(arr);
            WritableByteChannel channel = Channels.newChannel(bufferedOutputStream);

            int written = channel.write(asNumpy.asByteBuffer());
            if(written != asNumpy.capacity()) {
                throw new IllegalStateException("Not all bytes were written! Original capacity " + asNumpy.capacity() + " but wrote " + written);
            }

            bufferedOutputStream.flush();
        }

    }


    /**
     * Create from an in memory numpy pointer
     *
     * @param pointer the pointer to the
     *                numpy array
     * @return an ndarray created from the in memory
     * numpy pointer
     */

    public static INDArray createFromNpyPointer(Pointer pointer) {
        return INSTANCE.createFromNpyPointer(pointer);
    }

    /**
     * Create from a given Numpy .npy file.
     *
     * @param file the file to create the ndarray from
     * @return the created ndarray
     */
    public static INDArray createFromNpyFile(File file) {
        return INSTANCE.createFromNpyFile(file);
    }

    public static Map<String, INDArray> createFromNpzFile(File file) throws Exception{
        return INSTANCE.createFromNpzFile(file);
    }

    /**
     * Create a numpy array based on the passed in input stream
     * @param is the input stream to read
     * @return the loaded ndarray
     * @throws IOException
     */
    public static INDArray createNpyFromInputStream(InputStream is) throws IOException {
        byte[] content = IOUtils.toByteArray(is);
        return createNpyFromByteArray(content);
    }


    /**
     * Create an {@link INDArray} from the given numpy input.<br>
     * The numpy input follows the format:
     * https://docs.scipy.org/doc/numpy-1.14.0/neps/npy-format.html
     *
     * @param input the input byte array with the npy format
     * @return the equivalent {@link INDArray}
     */
    public static INDArray createNpyFromByteArray(byte[] input) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(input.length);
        byteBuffer.put(input);
        byteBuffer.rewind();
        Pointer pointer = new Pointer(byteBuffer);
        return createFromNpyPointer(pointer);
    }

    /**
     * Converts an {@link INDArray} to a byte array
     * @param input the input array
     * @return the {@link INDArray} as a byte array
     * with the numpy format.
     * For more on the format, see: https://docs.scipy.org/doc/numpy-1.14.0/neps/npy-format.html
     * @throws IOException
     */
    public static byte[] toNpyByteArray(INDArray input) {
        try {
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            writeAsNumpy(input, byteArrayOutputStream);
            return byteArrayOutputStream.toByteArray();
        } catch (IOException e){
            //Should never happen
            throw new RuntimeException(e);
        }
    }


    /**
     * Create an {@link INDArray} from a flatbuffers {@link FlatArray}
     * @param array the array to create the {@link INDArray} from
     * @return the created {@link INDArray}
     */
    public static INDArray createFromFlatArray(FlatArray array) {
        val dtype = array.dtype();
        val order = array.byteOrder();
        val rank = (int) array.shape(0);
        val shapeInfo = new long[Shape.shapeInfoLength(rank)];
        for (int e = 0; e < shapeInfo.length; e++)
            shapeInfo[e] = array.shape(e);

        if (Shape.isEmpty(shapeInfo))
            return Nd4j.empty();

        char ordering = shapeInfo[shapeInfo.length - 1] == 99 ? 'c' : 'f';

        val shapeOf = Shape.shapeOf(shapeInfo);
        val stridesOf = Shape.stridesOf(shapeInfo);

        val _dtype = FlatBuffersMapper.getDataTypeFromByte(dtype);
        val _order = FlatBuffersMapper.getOrderFromByte(order);
        val prod = rank > 0 ? ArrayUtil.prod(shapeOf) : 1;

        val bb = array.bufferAsByteBuffer();
        switch (_dtype) {
            case DOUBLE: {
                val doubles = new double[prod];
                val db = bb.order(_order).asDoubleBuffer();
                for (int e = 0; e < prod; e++)
                    doubles[e] = db.get(e);

                return Nd4j.create(doubles, shapeOf, stridesOf, ordering, DataType.DOUBLE);
            }
            case FLOAT: {
                val doubles = new float[prod];
                val fb = bb.order(_order).asFloatBuffer();
                for (int e = 0; e < prod; e++)
                    doubles[e] = fb.get(e);

                return Nd4j.create(doubles, shapeOf, stridesOf, ordering, DataType.FLOAT);
            }
            case HALF: {
                val doubles = new float[prod];
                val sb = bb.order(_order).asShortBuffer();
                for (int e = 0; e < prod; e++)
                    doubles[e] = HalfIndexer.toFloat((int) sb.get(e));

                return Nd4j.create(doubles, shapeOf, stridesOf, ordering, DataType.HALF);
            }
            case INT: {
                val doubles = new int[prod];
                val sb = bb.order(_order).asIntBuffer();
                for (int e = 0; e < prod; e++)
                    doubles[e] = sb.get(e);

                return Nd4j.create(doubles, shapeOf, stridesOf, ordering, DataType.INT);
            }
            case LONG: {
                val doubles = new long[prod];
                val sb = bb.order(_order).asLongBuffer();
                for (int e = 0; e < prod; e++)
                    doubles[e] = sb.get(e);

                return Nd4j.create(doubles, shapeOf, stridesOf, ordering, DataType.LONG);
            }
            case SHORT: {
                val doubles = new short[prod];
                val sb = bb.order(_order).asShortBuffer();
                for (int e = 0; e < prod; e++)
                    doubles[e] = sb.get(e);

                return Nd4j.create(doubles, shapeOf, stridesOf, ordering, DataType.SHORT);
            }
            case BYTE: {
                val bytes = new byte[prod];
                val sb = bb.order(_order).asReadOnlyBuffer();
                for (int e = 0; e < prod; e++)
                    bytes[e] = (byte) sb.get(e + sb.position());

                return Nd4j.create(bytes, shapeOf, stridesOf, ordering, DataType.BYTE);
            }
            case BOOL: {
                val doubles = new boolean[prod];
                val sb = bb.order(_order).asReadOnlyBuffer();
                for (int e = 0; e < prod; e++)
                    doubles[e] = sb.get(e + sb.position()) == 1;

                return Nd4j.create(doubles, shapeOf, stridesOf, ordering, DataType.BOOL);
            }
            case UTF8: {
                try {
                    val sb = bb.order(_order);
                    val pos = sb.position();
                    val arr = new byte[sb.limit() - pos];

                    for (int e = 0; e < arr.length; e++) {
                        arr[e] = sb.get(e + pos);
                    }

                    val buffer = new Utf8Buffer(arr, prod);
                    return Nd4j.create(buffer, shapeOf);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            case UBYTE:
                UInt8Buffer b = new UInt8Buffer(ArrayUtil.prod(shapeOf));
                val sb = bb.order(_order).asReadOnlyBuffer();
                for (int e = 0; e < prod; e++)
                    b.put(e, sb.get(e));

                return Nd4j.create(b, shapeOf);
            default:
                throw new UnsupportedOperationException("Unknown datatype: [" + _dtype + "]");
        }

    }

    /**
     * This method returns maximal allowed number of threads for Nd4j.
     * If value wasn't set in advance, max(1, availableProcessor) will be returned
     * @return
     */
    public static int numThreads() {
        val v = numThreads.get();
        if (v <= 0)
            return Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        else
            return v;
    }

    /**
     * This method sets maximal allowed number of threads for Nd4j
     * @param numthreads
     */
    public static void setNumThreads(int numthreads) {
        numThreads.set(numthreads);
    }

    public static DataType defaultFloatingPointType() {
        return defaultFloatingPointDataType.get();
    }

    public static boolean isPrecisionBoostAllowed() {
        return false;
    }


    public static INDArray scalar(@NonNull String string) {
        return create(Collections.singletonList(string), new long[0]);
    }

    public static INDArray create(@NonNull String... strings) {
        return create(Arrays.asList(strings), new long[]{strings.length}, Nd4j.order());
    }

    public static INDArray create(@NonNull Collection<String> strings, long... shape) {
        return create(strings, shape, Nd4j.order());
    }

    public static INDArray create(@NonNull Collection<String> strings, long[] shape, char order) {
        return INSTANCE.create(strings, shape, order);
    }

///////////////////

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with DOUBLE data type
     */
    public static INDArray createFromArray(double... array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        if(array.length == 0)
            return Nd4j.empty(DataType.DOUBLE);
        long[] shape = new long[]{array.length};
        return create(array, shape, ArrayUtil.calcStrides(shape), 'c', DataType.DOUBLE);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with FLOAT data type
     */
    public static INDArray createFromArray(float... array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        if(array.length == 0)
            return Nd4j.empty(DataType.FLOAT);
        long[] shape = new long[]{array.length};
        return create(array, shape, ArrayUtil.calcStrides(shape), 'c', DataType.FLOAT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with INT32 data type
     */
    public static INDArray createFromArray(int... array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        if(array.length == 0)
            return Nd4j.empty(DataType.INT);
        long[] shape = new long[]{array.length};
        return create(array, shape, ArrayUtil.calcStrides(shape), 'c', DataType.INT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with INT16 data type
     */
    public static INDArray createFromArray(short... array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        if(array.length == 0)
            return Nd4j.empty(DataType.SHORT);
        long[] shape = new long[]{array.length};
        return create(array, shape, ArrayUtil.calcStrides(shape), 'c', DataType.SHORT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with INT8 data type
     */
    public static INDArray createFromArray(byte... array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        if(array.length == 0)
            return Nd4j.empty(DataType.BYTE);
        long[] shape = new long[]{array.length};
        return create(array, shape, ArrayUtil.calcStrides(shape), 'c', DataType.BYTE);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with INT64 data type
     */
    public static INDArray createFromArray(long... array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        if(array.length == 0)
            return Nd4j.empty(DataType.LONG);
        long[] shape = new long[]{array.length};
        return create(array, shape, ArrayUtil.calcStrides(shape), 'c', DataType.LONG);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with BOOL data type
     */
    public static INDArray createFromArray(boolean... array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        if(array.length == 0)
            return Nd4j.empty(DataType.BOOL);
        long[] shape = new long[]{array.length};
        return create(array, shape, ArrayUtil.calcStrides(shape), 'c', DataType.BOOL);
    }

///////////////////

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with DOUBLE data type
     */
    public static INDArray createFromArray(double[][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0)
            return Nd4j.empty(DataType.DOUBLE);
        long[] shape = new long[]{array.length, array[0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.DOUBLE);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with FLOAT data type
     */
    public static INDArray createFromArray(float[][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0)
            return Nd4j.empty(DataType.FLOAT);
        long[] shape = new long[]{array.length, array[0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.FLOAT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with INT64 data type
     */
    public static INDArray createFromArray(long[][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0)
            return Nd4j.empty(DataType.LONG);
        long[] shape = new long[]{array.length, array[0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.LONG);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with INT32 data type
     */
    public static INDArray createFromArray(int[][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0)
            return Nd4j.empty(DataType.INT);
        long[] shape = new long[]{array.length, array[0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.INT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with INT16 data type
     */
    public static INDArray createFromArray(short[][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0)
            return Nd4j.empty(DataType.SHORT);
        long[] shape = new long[]{array.length, array[0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.SHORT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with INT8 data type
     */
    public static INDArray createFromArray(byte[][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0)
            return Nd4j.empty(DataType.BYTE);
        long[] shape = new long[]{array.length, array[0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.BYTE);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with BOOL data type
     */
    public static INDArray createFromArray(boolean[][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0)
            return Nd4j.empty(DataType.BOOL);

        long[] shape = new long[]{array.length, array[0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.BOOL);
    }

///////////////////

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with DOUBLE data type
     */
    public static INDArray createFromArray(double[][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0)
            return Nd4j.empty(DataType.DOUBLE);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.DOUBLE);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with FLOAT data type
     */
    public static INDArray createFromArray(float[][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0)
            return Nd4j.empty(DataType.FLOAT);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.FLOAT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with INT64 data type
     */
    public static INDArray createFromArray(long[][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0)
            return Nd4j.empty(DataType.LONG);

        long[] shape = new long[]{array.length, array[0].length, array[0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.LONG);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with INT32 data type
     */
    public static INDArray createFromArray(int[][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0)
            return Nd4j.empty(DataType.INT);

        long[] shape = new long[]{array.length, array[0].length, array[0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.INT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with INT16 data type
     */
    public static INDArray createFromArray(short[][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0)
            return Nd4j.empty(DataType.SHORT);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.SHORT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with INT8 data type
     */
    public static INDArray createFromArray(byte[][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0)
            return Nd4j.empty(DataType.BYTE);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.BYTE);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with BOOL data type
     */
    public static INDArray createFromArray(boolean[][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0)
            return Nd4j.empty(DataType.BOOL);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.BOOL);
    }

///////////////////

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with DOUBLE data type
     */
    public static INDArray createFromArray(double[][][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0 || array[0][0][0].length == 0)
            return Nd4j.empty(DataType.DOUBLE);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length, array[0][0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.DOUBLE);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with FLOAT data type
     */
    public static INDArray createFromArray(float[][][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0 || array[0][0][0].length == 0)
            return Nd4j.empty(DataType.FLOAT);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length, array[0][0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.FLOAT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with INT64 data type
     */
    public static INDArray createFromArray(long[][][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0 || array[0][0][0].length == 0)
            return Nd4j.empty(DataType.LONG);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length, array[0][0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.LONG);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with INT32 data type
     */
    public static INDArray createFromArray(int[][][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0 || array[0][0][0].length == 0)
            return Nd4j.empty(DataType.INT);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length, array[0][0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.INT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with INT16 data type
     */
    public static INDArray createFromArray(short[][][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0 || array[0][0][0].length == 0)
            return Nd4j.empty(DataType.SHORT);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length, array[0][0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.SHORT);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with INT8 data type
     */
    public static INDArray createFromArray(byte[][][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0 || array[0][0][0].length == 0)
            return Nd4j.empty(DataType.BYTE);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length, array[0][0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.BYTE);
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with BOOL data type
     */
    public static INDArray createFromArray(boolean[][][][] array) {
        Preconditions.checkNotNull(array, "Cannot create INDArray from null Java array");
        ArrayUtil.assertNotRagged(array);
        if(array.length == 0 || array[0].length == 0 || array[0][0].length == 0 || array[0][0][0].length == 0)
            return Nd4j.empty(DataType.BOOL);
        long[] shape = new long[]{array.length, array[0].length, array[0][0].length, array[0][0][0].length};
        return create(ArrayUtil.flatten(array), shape, ArrayUtil.calcStrides(shape), 'c', DataType.BOOL);
    }


    public static synchronized DeallocatorService getDeallocatorService() {
        if (deallocatorService == null)
            deallocatorService = new DeallocatorService();

        return deallocatorService;
    }

///////////////////

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with DOUBLE data type
     */
    public static INDArray createFromArray(Double[] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with FLOAT data type
     */
    public static INDArray createFromArray(Float[] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with INT32 data type
     */
    public static INDArray createFromArray(Integer[] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with INT16 data type
     */
    public static INDArray createFromArray(Short[] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with INT8 data type
     */
    public static INDArray createFromArray(Byte[] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with INT64 data type
     */
    public static INDArray createFromArray(Long[] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 1D INDArray with BOOL data type
     */
    public static INDArray createFromArray(Boolean[] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

///////////////////

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with DOUBLE data type
     */
    public static INDArray createFromArray(Double[][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with FLOAT data type
     */
    public static INDArray createFromArray(Float[][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with INT32 data type
     */
    public static INDArray createFromArray(Integer[][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with INT16 data type
     */
    public static INDArray createFromArray(Short[][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with INT8 data type
     */
    public static INDArray createFromArray(Byte[][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with INT64 data type
     */
    public static INDArray createFromArray(Long[][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 2D INDArray with BOOL data type
     */
    public static INDArray createFromArray(Boolean[][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

///////////////////

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with DOUBLE data type
     */
    public static INDArray createFromArray(Double[][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with FLOAT data type
     */
    public static INDArray createFromArray(Float[][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with INT32 data type
     */
    public static INDArray createFromArray(Integer[][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with INT16 data type
     */
    public static INDArray createFromArray(Short[][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with INT8 data type
     */
    public static INDArray createFromArray(Byte[][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with INT64 data type
     */
    public static INDArray createFromArray(Long[][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 3D INDArray with BOOL data type
     */
    public static INDArray createFromArray(Boolean[][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

///////////////////

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with DOUBLE data type
     */
    public static INDArray createFromArray(Double[][][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with FLOAT data type
     */
    public static INDArray createFromArray(Float[][][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with INT32 data type
     */
    public static INDArray createFromArray(Integer[][][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with INT16 data type
     */
    public static INDArray createFromArray(Short[][][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with INT8 data type
     */
    public static INDArray createFromArray(Byte[][][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with INT64 data type
     */
    public static INDArray createFromArray(Long[][][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    /**
     * This method creates INDArray from provided jvm array
     * @param array
     * @return 4D INDArray with BOOL data type
     */
    public static INDArray createFromArray(Boolean[][][][] array) {
        return createFromArray(ArrayUtil.toPrimitives(array));
    }

    public static boolean isExperimentalMode() {
        return getExecutioner().isExperimentalMode();
    }

    /**
     * Execute the operation and return the result
     *
     * @param op the operation to execute
     */
    public static INDArray exec(Op op){
        return getExecutioner().exec(op);
    }

    /**
     * Execute the operation and return the result
     *
     * @param op the operation to execute
     */
    public static INDArray[] exec(CustomOp op){
        return getExecutioner().exec(op);
    }

    /**
     * Execute the operation and return the result
     *
     * @param op the operation to execute
     */
    public static INDArray[] exec(CustomOp op, OpContext context){
        return getExecutioner().exec(op, context);
    }


    /**
     * This method applies ScatterUpdate op
     *
     * @param op
     * @param array
     * @param indices
     * @param updates
     * @param axis
     */
    @Deprecated
    public static void scatterUpdate(ScatterUpdate.UpdateOp op, @NonNull INDArray array, @NonNull INDArray indices, @NonNull INDArray updates, int... axis) {
        Preconditions.checkArgument(indices.dataType() == DataType.INT, "Indices should have INT data type");
        Preconditions.checkArgument(array.dataType() == updates.dataType(), "Array and updates should have the same data type");
        getExecutioner().scatterUpdate(op, array, indices, updates, axis);
    }
}
