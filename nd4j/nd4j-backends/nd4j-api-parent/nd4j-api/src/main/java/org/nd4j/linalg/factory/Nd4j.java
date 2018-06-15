/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.factory;

import com.google.common.base.Function;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import lombok.NonNull;
import lombok.val;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.HalfIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.context.Nd4jContext;
import org.nd4j.graph.FlatArray;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.api.buffer.factory.DefaultDataBufferFactory;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.concurrency.BasicAffinityManager;
import org.nd4j.linalg.api.instrumentation.InMemoryInstrumentation;
import org.nd4j.linalg.api.instrumentation.Instrumentation;
import org.nd4j.linalg.api.memory.MemoryWorkspaceManager;
import org.nd4j.linalg.api.ndarray.*;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.factory.DefaultOpFactory;
import org.nd4j.linalg.api.ops.factory.OpFactory;
import org.nd4j.linalg.api.ops.impl.controlflow.Select;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.shape.Diag;
import org.nd4j.linalg.api.ops.impl.transforms.OldReverse;
import org.nd4j.linalg.api.ops.impl.transforms.ReplaceNans;
import org.nd4j.linalg.api.ops.random.impl.Choice;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.api.ops.random.impl.Linspace;
import org.nd4j.linalg.api.ops.random.impl.UniformDistribution;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.factory.DefaultDistributionFactory;
import org.nd4j.linalg.api.rng.distribution.factory.DistributionFactory;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.cache.BasicConstantHandler;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.compression.BasicNDArrayCompressor;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.convolution.ConvolutionInstance;
import org.nd4j.linalg.convolution.DefaultConvolutionInstance;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.exception.ND4JComplexNumbersNotSupportedException;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4jBackend.NoAvailableBackendException;
import org.nd4j.linalg.memory.BasicMemoryManager;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.linalg.ops.transforms.Transforms;
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
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * Creation of ndarrays via classpath discovery.
 *
 * @author Adam Gibson
 */
public class Nd4j {

    public final static String NUMERICAL_STABILITY = "force.stability";
    public final static String FFT_OPS = "fft";
    public final static String DATA_BUFFER_OPS = "databufferfactory";
    public final static String CONVOLUTION_OPS = "convops";
    public final static String DTYPE = "dtype";
    public final static String BLAS_OPS = "blas.ops";
    public final static String SPARSE_BLAS_OPS = "sparseblas.ops";
    public final static String NATIVE_OPS = "native.ops";
    public final static String ORDER_KEY = "ndarray.order";
    public final static String NDARRAY_FACTORY_CLASS = "ndarrayfactory.class";
    public final static String SPARSE_NDARRAY_FACTORY_CLASS = "sparsendarrayfactory.class";
    public final static String COPY_OPS = "ndarray.copyops";
    public final static String OP_EXECUTIONER = "opexec";
    public final static String OP_FACTORY = "opfactory";
    public final static String DISTRIBUTION = "dist";
    public final static String INSTRUMENTATION = "instrumentation";
    public final static String INSTRUMENTATION_CLASS = "instrumentation.class";
    public final static String RESOURCE_MANGER_ON = "resourcemanager_state";
    public final static String EXECUTION_MODE = "opexec.mode";
    public final static String SHAPEINFO_PROVIDER = "shapeinfoprovider";
    public final static String SPARSEINFO_PROVIDER = "sparseinfoprovider";
    public final static String CONSTANT_PROVIDER = "constantsprovider";
    public final static String AFFINITY_MANAGER = "affinitymanager";
    //disable toString() on compressed arrays for debugging. Should be off by default.
    public final static String COMPRESSION_DEBUG = "compressiondebug";
    public final static String MEMORY_MANAGER = "memorymanager";
    public final static String WORKSPACE_MANAGER = "workspacemanager";
    public final static String RANDOM_PROVIDER = "random";
    //execution mode for element wise operations
    public static OpExecutioner.ExecutionMode executionMode = OpExecutioner.ExecutionMode.JAVA;

    //the datatype used for allocating buffers
    protected static DataBuffer.Type dtype = DataBuffer.Type.FLOAT;
    //the allocation mode for the heap
    public static DataBuffer.AllocationMode alloc = DataBuffer.AllocationMode.HEAP;
    public static char ORDER = 'c';
    public static IComplexNumber UNIT;
    public static IComplexNumber ZERO;
    public static IComplexNumber NEG_UNIT;
    public static double EPS_THRESHOLD = 1e-5;
    //number of elements to print in begin and end
    public static int MAX_ELEMENTS_PER_SLICE = 3;
    public static int MAX_SLICES_TO_PRINT = 3;
    public static boolean ENFORCE_NUMERICAL_STABILITY = false;
    public static boolean copyOnOps = true;
    public static boolean shouldInstrument = false;
    public static boolean resourceManagerOn = false;
    private static boolean allowsOrder = false;
    public static boolean compressDebug = false;
    public static boolean preventUnpack = System.getenv("ND4J_PREVENT_UNPACK") != null;
    public static Nd4jBackend backend;
    public static RandomFactory randomFactory;
    private static MemoryWorkspaceManager workspaceManager;

    protected static Class<? extends MemoryWorkspaceManager> workspaceManagerClazz;
    protected static Class<? extends BlasWrapper> blasWrapperClazz;
    protected static Class<? extends BlasWrapper> sparseBlasWrapperClazz;
    protected static Class<? extends NDArrayFactory> ndArrayFactoryClazz;
    protected static Class<? extends NDArrayFactory> sparseNDArrayClazz;
    protected static Class<? extends ConvolutionInstance> convolutionInstanceClazz;
    protected static Class<? extends DataBufferFactory> dataBufferFactoryClazz;
    protected static Class<? extends OpExecutioner> opExecutionerClazz;
    protected static Class<? extends OpFactory> opFactoryClazz;
    protected static Class<? extends org.nd4j.linalg.api.rng.Random> randomClazz;
    protected static Class<? extends DistributionFactory> distributionFactoryClazz;
    protected static Class<? extends Instrumentation> instrumentationClazz;
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
    protected static OpFactory OP_FACTORY_INSTANCE;
    protected static Instrumentation instrumentation;
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

    static {
        fallbackMode = new AtomicBoolean(false);
        Nd4j nd4j = new Nd4j();
        nd4j.initContext();
    }


    public enum PadMode {
        CONSTANT, EDGE, LINEAR_RAMP, MAXIMUM, MEAN, MEDIAN, MINIMUM, REFLECT, SYMMETRIC, WRAP

    }

    /**
     * Pad the given ndarray to the size along each dimension
     * @param toPad the ndarray to pad
     * @param padWidth the width to pad along each dimension
     * @param padMode the mode to pad in
     * @return the padded ndarray
     * based on the specified mode
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
     * Pad the given ndarray to the size along each dimension
     * @param toPad the ndarray to pad
     * @param padWidth the width to pad along each dimension
     * @param constantValues the values to append for each dimension
     * @param padMode the mode to pad in
     * @return the padded ndarray
     * based on the specified mode
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
     * Pad the given ndarray to the size along each dimension
     * @param toPad the ndarray to pad
     * @param padWidth the width to pad along each dimension
     * @param padMode the mode to pad in
     * @return the padded ndarray
     * based on the specified mode
     */
    public static INDArray pad(INDArray toPad, int[] padWidth, PadMode padMode) {
        return pad(toPad, padWidth, ArrayUtil.zerosMatrix(padWidth), padMode);
    }


    /**
     * Append the given
     * array with the specified value size
     * along a particular axis
     * @param arr the array to append to
     * @param padAmount the pad amount of the array to be returned
     * @param val the value to append
     * @param axis the axis to append to
     * @return the newly created array
     */
    public static INDArray append(INDArray arr, int padAmount, double val, int axis) {
        if (padAmount == 0)
            return arr;
        long[] paShape = ArrayUtil.copy(arr.shape());
        if (axis < 0)
            axis = axis + arr.shape().length;
        paShape[axis] = padAmount;
        INDArray concatArray = Nd4j.valueArrayOf(paShape, val);
        return Nd4j.concat(axis, arr, concatArray);
    }

    /**
     * Append the given
     * array with the specified value size
     * along a particular axis
     * @param arr the array to append to
     * @param padAmount the pad amount of the array to be returned
     * @param val the value to append
     * @param axis the axis to append to
     * @return the newly created array
     */
    public static INDArray prepend(INDArray arr, int padAmount, double val, int axis) {
        if (padAmount == 0)
            return arr;

        long[] paShape = ArrayUtil.copy(arr.shape());
        if (axis < 0)
            axis = axis + arr.shape().length;
        paShape[axis] = padAmount;
        INDArray concatArr = Nd4j.valueArrayOf(paShape, val);
        return Nd4j.concat(axis, concatArr, arr);
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
    public static void shuffle(INDArray toShuffle, Random random, int... dimension) {
        INSTANCE.shuffle(toShuffle, random, dimension);
    }

    /**
     * In place shuffle of an ndarray
     * along a specified set of dimensions
     * @param toShuffle the ndarray to shuffle
     * @param dimension the dimension to do the shuffle
     * @return
     */
    public static void shuffle(INDArray toShuffle, int... dimension) {
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
    public static void shuffle(Collection<INDArray> toShuffle, int... dimension) {
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
    public static void shuffle(Collection<INDArray> toShuffle, Random rnd, int... dimension) {
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
     * Gets the instrumentation instance
     *
     * @return the instrumentation instance
     */
    public static Instrumentation getInstrumentation() {
        return instrumentation;
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

    public static void setNdArrayFactoryClazz(Class<? extends NDArrayFactory> clazz) {
        ndArrayFactoryClazz = clazz;
    }

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
        int slices = arrays.length;
        INDArray ret = Nd4j.create(ArrayUtil.combine(new int[] {slices}, sliceShape));
        for (int i = 0; i < ret.slices(); i++)
            ret.putSlice(i, Nd4j.create(arrays[i]).reshape(ArrayUtil.toLongArray(sliceShape)));
        return ret;
    }

    /**
     * Create an ndarray based on the given data
     * @param sliceShape the shape of each slice
     * @param arrays the arrays of data to create
     * @return the ndarray of the specified shape where
     * number of slices is equal to array length and each
     * slice is the specified shape
     */
    public static INDArray create(int[] sliceShape, double[]... arrays) {
        int slices = arrays.length;
        INDArray ret = Nd4j.create(ArrayUtil.combine(new int[] {slices}, sliceShape));
        for (int i = 0; i < ret.slices(); i++)
            ret.putSlice(i, Nd4j.create(arrays[i]).reshape(ArrayUtil.toLongArray(sliceShape)));
        return ret;
    }


    /**
     * Get the operation executioner instance
     *
     * @return the operation executioner instance
     */
    public static OpExecutioner getExecutioner() {
        return OP_EXECUTIONER_INSTANCE;
    }

    /**
     * Get the operation factory
     *
     * @return the operation factory
     */
    public static OpFactory getOpFactory() {
        return OP_FACTORY_INSTANCE;
    }

    /**
     *
     * @return
     */
    public static DataBufferFactory getDataBufferFactory() {
        return DATA_BUFFER_FACTORY_INSTANCE;
    }


    /**
     * Given a sequence of Iterators over a transform of matrices, fill in all of
     * the matrices with the entries in the theta vector.  Errors are
     * thrown if the theta vector does not exactly fill the matrices.
     */
    public static void setParams(INDArray theta, Collection<INDArray>... matrices) {
        int index = 0;
        for (Collection<INDArray> matrixCollection : matrices) {
            for (INDArray matrix : matrixCollection) {
                INDArray linear = matrix.linearView();
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
     *
     * @param arr
     * @param dimension
     * @return
     */
    public static INDArray argMax(INDArray arr, int... dimension) {
        IMax imax = new IMax(arr);
        if(dimension == null || dimension.length == 0){
            return Nd4j.getExecutioner().exec(imax, Integer.MAX_VALUE);
        } else {
            return Nd4j.getExecutioner().exec(imax, dimension);
        }
    }

    /**
     *
     * @param arr
     * @param dimension
     * @return
     */
    public static INDArray argMin(INDArray arr, int... dimension) {
        IMin imin = new IMin(arr);
        if(dimension == null || dimension.length == 0){
            return Nd4j.getExecutioner().exec(imin, Integer.MAX_VALUE);
        } else {
            return Nd4j.getExecutioner().exec(imin, dimension);
        }
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
        INDArray ret = at.mmul(bt);

        long[] aPlusB = Longs.concat(oldShapeA, oldShapeB);
        return ret.reshape(aPlusB);
    }

    /**
     *
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
        INDArray c = Nd4j.createUninitialized(new long[] {cRows, cCols}, 'f');
        return gemm(a, b, c, transposeA, transposeB, 1.0, 0.0);
    }

    /** Matrix multiply: Implements c = alpha*op(a)*op(b) + beta*c where op(X) means transpose X (or not)
     * depending on setting of arguments transposeA and transposeB.<br>
     * Note that matrix c MUST be fortran order, have zero offset and have c.data().length == c.length().
     * An exception will be thrown otherwise.<br>
     * Don't use this unless you know about level 3 blas and NDArray storage orders.
     * @param a First matrix
     * @param b Second matrix
     * @param c result matrix. Used in calculation (assuming beta != 0) and result is stored in this. f order,
     *          zero offset and length == data.length only
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
        getBlasWrapper().level3().gemm(a, b, c, transposeA, transposeB, alpha, beta);
        return c;
    }

    /**
     * Given a sequence of Iterators over a transform of matrices, fill in all of
     * the matrices with the entries in the theta vector.  Errors are
     * thrown if the theta vector does not exactly fill the matrices.
     */
    public static void setParams(INDArray theta, Iterator<? extends INDArray>... matrices) {
        int index = 0;
        for (Iterator<? extends INDArray> matrixIterator : matrices) {
            while (matrixIterator.hasNext()) {
                INDArray matrix = matrixIterator.next().linearView();
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


    private static void logCreationIfNecessary(DataBuffer log) {
        if (shouldInstrument)
            Nd4j.getInstrumentation().log(log);

    }


    private static void logCreationIfNecessary(INDArray log) {
        if (shouldInstrument)
            Nd4j.getInstrumentation().log(log);
    }

    /**
     * The factory used for creating ndarrays
     *
     * @return the factory instance used for creating ndarrays
     */
    public static NDArrayFactory factory() {
        return INSTANCE;
    }

    public static NDArrayFactory sparseFactory() {
        return SPARSE_INSTANCE;
    }

    public static INDArray cumsum(INDArray compute) {
        return compute.cumsum(Integer.MAX_VALUE);
    }

    public static INDArray max(INDArray compute) {
        return compute.max(Integer.MAX_VALUE);
    }

    public static INDArray min(INDArray compute) {
        return compute.min(Integer.MAX_VALUE);
    }

    public static INDArray prod(INDArray compute) {
        return compute.prod(Integer.MAX_VALUE);
    }

    public static INDArray normmax(INDArray compute) {
        return compute.normmax(Integer.MAX_VALUE);
    }

    public static INDArray norm2(INDArray compute) {
        return compute.norm2(Integer.MAX_VALUE);
    }

    public static INDArray norm1(INDArray compute) {
        return compute.norm1(Integer.MAX_VALUE);
    }

    public static INDArray std(INDArray compute) {
        return compute.std(Integer.MAX_VALUE);
    }

    public static INDArray var(INDArray compute) {
        return compute.var(Integer.MAX_VALUE);
    }

    public static INDArray sum(INDArray compute) {
        return compute.sum(Integer.MAX_VALUE);
    }

    public static INDArray mean(INDArray compute) {
        return compute.mean(Integer.MAX_VALUE);
    }

    public static IComplexNDArray cumsum(IComplexNDArray compute) {
        return compute.cumsum(Integer.MAX_VALUE);
    }

    public static IComplexNDArray max(IComplexNDArray compute) {
        return compute.max(Integer.MAX_VALUE);
    }

    public static IComplexNDArray min(IComplexNDArray compute) {
        return compute.min(Integer.MAX_VALUE);
    }

    public static IComplexNDArray prod(IComplexNDArray compute) {
        return compute.prod(Integer.MAX_VALUE);
    }

    public static IComplexNDArray normmax(IComplexNDArray compute) {
        return compute.normmax(Integer.MAX_VALUE);
    }

    public static IComplexNDArray norm2(IComplexNDArray compute) {
        return compute.norm2(Integer.MAX_VALUE);
    }

    public static IComplexNDArray norm1(IComplexNDArray compute) {
        return compute.norm1(Integer.MAX_VALUE);
    }

    public static IComplexNDArray std(IComplexNDArray compute) {
        return compute.std(Integer.MAX_VALUE);
    }

    public static IComplexNDArray var(IComplexNDArray compute) {
        return compute.var(Integer.MAX_VALUE);
    }

    public static IComplexNDArray sum(IComplexNDArray compute) {
        return compute.sum(Integer.MAX_VALUE);
    }

    public static IComplexNDArray mean(IComplexNDArray compute) {
        return compute.mean(Integer.MAX_VALUE);
    }

    public static INDArray cumsum(INDArray compute, int dimension) {
        return compute.cumsum(dimension);
    }

    public static INDArray max(INDArray compute, int dimension) {
        return compute.max(dimension);
    }

    public static INDArray min(INDArray compute, int dimension) {
        return compute.min(dimension);
    }

    public static INDArray prod(INDArray compute, int dimension) {
        return compute.prod(dimension);
    }

    public static INDArray normmax(INDArray compute, int dimension) {
        return compute.normmax(dimension);
    }

    public static INDArray norm2(INDArray compute, int dimension) {
        return compute.norm2(dimension);
    }

    public static INDArray norm1(INDArray compute, int dimension) {
        return compute.norm1(dimension);
    }

    public static INDArray std(INDArray compute, int dimension) {
        return compute.std(dimension);
    }

    public static INDArray var(INDArray compute, int dimension) {
        return compute.var(dimension);
    }

    public static INDArray sum(INDArray compute, int dimension) {
        return compute.sum(dimension);
    }

    public static INDArray mean(INDArray compute, int dimension) {
        return compute.mean(dimension);
    }

    public static IComplexNDArray cumsum(IComplexNDArray compute, int dimension) {
        return compute.cumsum(dimension);
    }

    public static IComplexNDArray max(IComplexNDArray compute, int dimension) {
        return compute.max(dimension);
    }

    public static IComplexNDArray min(IComplexNDArray compute, int dimension) {
        return compute.min(dimension);
    }

    public static IComplexNDArray prod(IComplexNDArray compute, int dimension) {
        return compute.prod(dimension);
    }

    public static IComplexNDArray normmax(IComplexNDArray compute, int dimension) {
        return compute.normmax(dimension);
    }

    public static IComplexNDArray norm2(IComplexNDArray compute, int dimension) {
        return compute.norm2(dimension);
    }

    public static IComplexNDArray norm1(IComplexNDArray compute, int dimension) {
        return compute.norm1(dimension);
    }

    public static IComplexNDArray std(IComplexNDArray compute, int dimension) {
        return compute.std(dimension);
    }

    public static IComplexNDArray var(IComplexNDArray compute, int dimension) {
        return compute.var(dimension);
    }

    public static IComplexNDArray sum(IComplexNDArray compute, int dimension) {
        return compute.sum(dimension);
    }

    public static IComplexNDArray mean(IComplexNDArray compute, int dimension) {
        return compute.mean(dimension);
    }


    /**
     * Create a view of a data buffer
     * that leverages the underlying storage of the buffer
     * with a new view
     * @param underlyingBuffer the underlying buffer
     * @param offset the offset for the view
     * @return the new view of the data buffer
     */
    public static DataBuffer createBuffer(DataBuffer underlyingBuffer, long offset, long length) {
        return DATA_BUFFER_FACTORY_INSTANCE.create(underlyingBuffer, offset, length);
    }


    /**
     *
     * Create a buffer equal of length prod(shape)
     *
     * @param shape the shape of the buffer to create
     * @param type  the opType to create
     * @return the created buffer
     */
    public static DataBuffer createBuffer(int[] shape, DataBuffer.Type type, long offset) {
        int length = ArrayUtil.prod(shape);
        return type == DataBuffer.Type.DOUBLE ? createBuffer(new double[length], offset)
                : createBuffer(new float[length], offset);
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
    public static DataBuffer createBuffer(ByteBuffer buffer, DataBuffer.Type type, int length, long offset) {
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
        if (dataType() == DataBuffer.Type.DOUBLE)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createDouble(offset, data, length);
        else
            ret = DATA_BUFFER_FACTORY_INSTANCE.createFloat(offset, data, length);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a buffer equal of length prod(shape)
     *
     * @param data the shape of the buffer to create
     * @return the created buffer
     */
    public static DataBuffer createBuffer(int[] data, long offset) {
        DataBuffer ret;
        ret = DATA_BUFFER_FACTORY_INSTANCE.createInt(offset, data);
        logCreationIfNecessary(ret);
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
        if (dataType() == DataBuffer.Type.FLOAT)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createFloat(offset, length);
        else if (dataType() == DataBuffer.Type.INT)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createInt(offset, length);
        else if (dataType() == DataBuffer.Type.DOUBLE)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createDouble(offset, length);
        else if (dataType() == DataBuffer.Type.HALF)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createHalf(offset, length);
        else
            ret = null;


        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a double data opType buffer
     * of the specified length
     * @param doublePointer
     * @param length
     * @return
     */
    public static DataBuffer createBuffer(DoublePointer doublePointer, long length) {
        return DATA_BUFFER_FACTORY_INSTANCE.create(doublePointer, length);
    }


    /**
     * Create a float opType
     * data buffer of the given length
     * @param floatPointer
     * @param length
     * @return
     */
    public static DataBuffer createBuffer(FloatPointer floatPointer, long length) {
        return DATA_BUFFER_FACTORY_INSTANCE.create(floatPointer, length);
    }


    /**
     * Create an int data buffer
     * from the given pointer
     * and the given length
     * @param intPointer
     * @param length
     * @return
     */
    public static DataBuffer createBuffer(IntPointer intPointer, long length) {
        return DATA_BUFFER_FACTORY_INSTANCE.create(intPointer, length);
    }

    /**
     * Create a buffer based on the data opType
     *
     * @param data the data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBuffer(float[] data, long offset) {
        DataBuffer ret;
        if (dataType() == DataBuffer.Type.FLOAT)
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createFloat(offset, data) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(offset, data, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (dataType() == DataBuffer.Type.HALF)
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createHalf(offset, data) : DATA_BUFFER_FACTORY_INSTANCE.createHalf(offset, data, Nd4j.getMemoryManager().getCurrentWorkspace());
        else
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(offset, ArrayUtil.toDoubles(data)) : DATA_BUFFER_FACTORY_INSTANCE.createDouble(offset, ArrayUtil.toDoubles(data), Nd4j.getMemoryManager().getCurrentWorkspace());
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a buffer based on the data opType
     *
     * @param data the data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBuffer(double[] data, long offset) {
        DataBuffer ret;
        if (dataType() == DataBuffer.Type.DOUBLE)
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(offset, data) : DATA_BUFFER_FACTORY_INSTANCE.createDouble(offset, data, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (dataType() == DataBuffer.Type.HALF)
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createHalf(offset, data) : DATA_BUFFER_FACTORY_INSTANCE.createHalf(offset, ArrayUtil.toFloats(data), Nd4j.getMemoryManager().getCurrentWorkspace());
        else
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createFloat(offset, ArrayUtil.toFloats(data)) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(offset, ArrayUtil.toFloats(data), Nd4j.getMemoryManager().getCurrentWorkspace());
        logCreationIfNecessary(ret);
        return ret;
    }



    /**
     * Create a buffer equal of length prod(shape)
     *
     * @param shape the shape of the buffer to create
     * @param type  the opType to create
     * @return the created buffer
     */
    public static DataBuffer createBuffer(int[] shape, DataBuffer.Type type) {
        long length = ArrayUtil.prodLong(shape);

        if (type == DataBuffer.Type.INT)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createInt(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createInt(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (type == DataBuffer.Type.LONG)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createLong(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createLong(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (type == DataBuffer.Type.HALF)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createHalf(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createHalf(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (type == DataBuffer.Type.DOUBLE)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createDouble(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createFloat(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
    }


    public static DataBuffer createBufferDetached(int[] shape, DataBuffer.Type type) {
        long length = ArrayUtil.prodLong(shape);
        if (type == DataBuffer.Type.INT)
            return DATA_BUFFER_FACTORY_INSTANCE.createInt(length);
        if (type == DataBuffer.Type.LONG)
            return DATA_BUFFER_FACTORY_INSTANCE.createLong(new long[]{length});
        else if (type == DataBuffer.Type.HALF)
            return DATA_BUFFER_FACTORY_INSTANCE.createHalf(length);

        return type == DataBuffer.Type.DOUBLE ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(length) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(length);
    }

    public static DataBuffer createBuffer(long[] shape, DataBuffer.Type type) {
        long length = ArrayUtil.prodLong(shape);

        if (type == DataBuffer.Type.INT)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createInt(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createInt(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (type == DataBuffer.Type.LONG)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createLong(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createLong(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (type == DataBuffer.Type.HALF)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createHalf(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createHalf(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (type == DataBuffer.Type.DOUBLE)
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createDouble(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
        else
            return Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createFloat(length, true) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(length, true, Nd4j.getMemoryManager().getCurrentWorkspace());
    }


    public static DataBuffer createBufferDetached(long[] shape, DataBuffer.Type type) {
        long length = ArrayUtil.prodLong(shape);
        if (type == DataBuffer.Type.INT)
            return DATA_BUFFER_FACTORY_INSTANCE.createInt(length);
        else if (type == DataBuffer.Type.LONG)
            return DATA_BUFFER_FACTORY_INSTANCE.createLong(length);
        else if (type == DataBuffer.Type.HALF)
            return DATA_BUFFER_FACTORY_INSTANCE.createHalf(length);

        return type == DataBuffer.Type.DOUBLE ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(length) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(length);
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
    public static DataBuffer createBuffer(ByteBuffer buffer, DataBuffer.Type type, int length) {
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
        if (dataType() == DataBuffer.Type.DOUBLE)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createDouble(data, length);
        else if (dataType() == DataBuffer.Type.HALF)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createHalf(data, length);
        else
            ret = DATA_BUFFER_FACTORY_INSTANCE.createFloat(data, length);
        logCreationIfNecessary(ret);
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
        logCreationIfNecessary(ret);
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
        logCreationIfNecessary(ret);
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
        logCreationIfNecessary(ret);
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
    public static DataBuffer createBuffer(Pointer pointer, DataBuffer.Type type, long length, Indexer indexer) {
        return DATA_BUFFER_FACTORY_INSTANCE.create(pointer, type, length, indexer);
    }

    /**
     *
     * @param length
     * @param initialize
     * @return
     */
    public static DataBuffer createBuffer(long length, boolean initialize) {
        DataBuffer ret;
        if (dataType() == DataBuffer.Type.FLOAT)
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createFloat(length, initialize) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(length, initialize, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (dataType() == DataBuffer.Type.INT)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createInt(length, initialize);
        else if (dataType() == DataBuffer.Type.HALF)
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createHalf(length, initialize) : DATA_BUFFER_FACTORY_INSTANCE.createHalf(length, initialize, Nd4j.getMemoryManager().getCurrentWorkspace());
        else
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(length, initialize) : DATA_BUFFER_FACTORY_INSTANCE.createDouble(length, initialize, Nd4j.getMemoryManager().getCurrentWorkspace());

        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a buffer based on the data opType
     *
     * @param data the data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBuffer(float[] data) {
        DataBuffer ret;
        if (dataType() == DataBuffer.Type.FLOAT)
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createFloat(data) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(data, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (dataType() == DataBuffer.Type.HALF)
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createHalf(data): DATA_BUFFER_FACTORY_INSTANCE.createHalf(data, Nd4j.getMemoryManager().getCurrentWorkspace());
        else
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(ArrayUtil.toDoubles(data)) : DATA_BUFFER_FACTORY_INSTANCE.createDouble(ArrayUtil.toDoubles(data), Nd4j.getMemoryManager().getCurrentWorkspace()) ;
        logCreationIfNecessary(ret);
        return ret;
    }

    public static DataBuffer createBufferDetached(float[] data) {
        DataBuffer ret;
        if (dataType() == DataBuffer.Type.FLOAT)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createFloat(data);
        else if (dataType() == DataBuffer.Type.HALF)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createHalf(data);
        else
            ret = DATA_BUFFER_FACTORY_INSTANCE.createDouble(ArrayUtil.toDoubles(data));
        logCreationIfNecessary(ret);
        return ret;
    }

    public static DataBuffer createBufferDetached(double[] data) {
        DataBuffer ret;
        if (dataType() == DataBuffer.Type.DOUBLE)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createDouble(data);
        else if (dataType() == DataBuffer.Type.HALF)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createHalf(ArrayUtil.toFloats(data));
        else
            ret = DATA_BUFFER_FACTORY_INSTANCE.createFloat(ArrayUtil.toFloats(data));
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a buffer based on the data opType
     *
     * @param data the data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBuffer(double[] data) {
        DataBuffer ret;
        if (dataType() == DataBuffer.Type.DOUBLE)
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createDouble(data) : DATA_BUFFER_FACTORY_INSTANCE.createDouble(data, Nd4j.getMemoryManager().getCurrentWorkspace());
        else if (dataType() == DataBuffer.Type.HALF)
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createHalf(data) : DATA_BUFFER_FACTORY_INSTANCE.createHalf(ArrayUtil.toFloats(data), Nd4j.getMemoryManager().getCurrentWorkspace());
        else
            ret = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? DATA_BUFFER_FACTORY_INSTANCE.createFloat(ArrayUtil.toFloats(data)) : DATA_BUFFER_FACTORY_INSTANCE.createFloat(ArrayUtil.toFloats(data), Nd4j.getMemoryManager().getCurrentWorkspace());
        logCreationIfNecessary(ret);
        return ret;
    }


    public static void setFactory(NDArrayFactory factory) {
        INSTANCE = factory;
    }

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
    public static DataBuffer.Type dataType() {
        return DataTypeUtil.getDtypeFromContext();
    }

    /**
     * This method sets dataType for the current JVM runtime
     * @param dType
     */
    public static void setDataType(@NonNull DataBuffer.Type dType) {
        DataTypeUtil.setDTypeForContext(dType);
    }

    /**
     *
     * @return
     */
    public static Nd4jBackend getBackend() {
        return backend;
    }

    /**
     *
     * @return
     */
    public static BlasWrapper getBlasWrapper() {
        return BLAS_WRAPPER_INSTANCE;
    }

    /**
     *
     * @return
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
     * Create an n x (shape)
     * ndarray where the ndarray is repeated num times
     *
     * @param n   the ndarray to replicate
     * @param num the number of copies to repeat
     * @return the repeated ndarray
     */
    public static IComplexNDArray repeat(IComplexNDArray n, int num) {
        /*
        List<IComplexNDArray> list = new ArrayList<>();
        for (int i = 0; i < num; i++)
            list.add(n.dup());
        IComplexNDArray ret = Nd4j.createComplex(list, Ints.concat(new long[] {num}, n.shape()));
        logCreationIfNecessary(ret);
        return ret;
        */
        throw new UnsupportedOperationException();
    }

    /**
     * Sort an ndarray along a particular dimension
     *
     * @param ndarray   the ndarray to sort
     * @param dimension the dimension to sort
     * @return an array with indices and the sorted ndarray
     */
    public static INDArray[] sortWithIndices(IComplexNDArray ndarray, int dimension, boolean ascending) {
        INDArray indices = Nd4j.create(ndarray.shape());
        INDArray[] ret = new INDArray[2];

        for (int i = 0; i < ndarray.vectorsAlongDimension(dimension); i++) {
            IComplexNDArray vec = ndarray.vectorAlongDimension(i, dimension);
            INDArray indexVector = indices.vectorAlongDimension(i, dimension);

            final IComplexNumber[] data = new IComplexNumber[(int) vec.length()];
            final Double[] index = new Double[(int) vec.length()];

            for (int j = 0; j < vec.length(); j++) {
                data[j] = vec.getComplex(j);
                index[j] = (double) j;

            }


            if (ascending)
                Arrays.sort(index, new Comparator<Double>() {
                    @Override
                    public int compare(Double o1, Double o2) {
                        int idx1 = (int) o1.doubleValue();
                        int idx2 = (int) o2.doubleValue();

                        return Double.compare(data[idx1].absoluteValue().doubleValue(),
                                data[idx2].absoluteValue().doubleValue());
                    }
                });

            else
                Arrays.sort(index, new Comparator<Double>() {
                    @Override
                    public int compare(Double o1, Double o2) {
                        int idx1 = (int) o1.doubleValue();
                        int idx2 = (int) o2.doubleValue();

                        return -Double.compare(data[idx1].absoluteValue().doubleValue(),
                                data[idx2].absoluteValue().doubleValue());
                    }
                });


            for (int j = 0; j < vec.length(); j++) {
                vec.putScalar(j, data[(int) index[j].doubleValue()]);
                indexVector.putScalar(j, index[j]);
            }


        }

        ret[0] = indices;
        ret[1] = ndarray;

        return ret;
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

        for (int i = 0; i < ndarray.vectorsAlongDimension(dimension); i++) {
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
     * Sort an ndarray along a particular dimension
     *
     * @param ndarray   the ndarray to sort
     * @param dimension the dimension to sort
     * @return the sorted ndarray
     */
    public static IComplexNDArray sort(IComplexNDArray ndarray, int dimension, boolean ascending) {
        /*
        for (int i = 0; i < ndarray.vectorsAlongDimension(dimension); i++) {
            IComplexNDArray vec = ndarray.vectorAlongDimension(i, dimension);
            IComplexNumber[] data = new IComplexNumber[vec.length()];
            for (int j = 0; j < vec.length(); j++) {
                data[j] = vec.getComplex(j);
            }
            if (ascending)
                Arrays.sort(data, new Comparator<IComplexNumber>() {
                    @Override
                    public int compare(IComplexNumber o1, IComplexNumber o2) {
                        return Double.compare(o1.asDouble().absoluteValue().doubleValue(),
                                o2.asDouble().absoluteValue().doubleValue());
                    }
                });

            else
                Arrays.sort(data, new Comparator<IComplexNumber>() {
                    @Override
                    public int compare(IComplexNumber o1, IComplexNumber o2) {
                        return -Double.compare(o1.asDouble().absoluteValue().doubleValue(),
                                o2.asDouble().absoluteValue().doubleValue());
                    }
                });

            for (int j = 0; j < vec.length(); j++)
                vec.putScalar(j, data[j]);


        }

        return ndarray;
        */
        throw new UnsupportedOperationException();
    }


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

        INDArray out = Nd4j.create(in.shape());
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
     * @param upper lower bound
     * @param num   the step size
     * @return the linearly spaced vector
     */
    public static INDArray linspace(long lower, long upper, long num) {
        // for now we'll temporarily keep original impl
        if(lower == upper && num == 1) {
            return Nd4j.scalar(lower);
        }


        double approx = (double) num / ((double) (upper - lower) + 1);
        if (approx % 1 <= EPS_THRESHOLD) {
            // FIXME: int cast
            return INSTANCE.linspace((int) lower, (int) upper, (int) num);
        } else {
            return linspace((double) lower, (double) upper, (int) num);
        }
    }


    /**
     * Generate a linearly spaced vector
     *
     * @param lower upper bound
     * @param upper lower bound
     * @param num   the step size
     * @return the linearly spaced vector
     */
    public static INDArray linspace(double lower, double upper, long num) {
        // FIXME: int cast
        return Nd4j.getExecutioner().exec(new Linspace(lower, upper, (int) num));
    }

    /**
     * Generate a linearly spaced vector
     *
     * @param lower upper bound
     * @param upper lower bound
     * @param num   the step size
     * @return the linearly spaced vector
     */
    public static INDArray linspace(float lower, float upper, long num) {
        return linspace((double) lower, (double) upper, num);
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

        INDArray xOut = Nd4j.createUninitialized(y.length(), x.length());
        INDArray yOut = Nd4j.createUninitialized(y.length(), x.length());

        CustomOp op = DynamicCustomOp.builder("meshgrid")
                .addInputs(x, y)
                .addOutputs(xOut, yOut)
                .build();
        Nd4j.getExecutioner().exec(op);

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
        logCreationIfNecessary(ret);
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
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a long row vector of all of the given ndarrays
     * @param flatten the matrices to create the flattened ndarray for
     * @return the flattened representation of
     * these ndarrays
     */
    public static IComplexNDArray complexFlatten(List<IComplexNDArray> flatten) {
        IComplexNDArray ret = INSTANCE.complexFlatten(flatten);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a long row vector of all of the given ndarrays
     * @param flatten the matrices to create the flattened ndarray for
     * @return the flattened representation of
     * these ndarrays
     */
    public static IComplexNDArray complexFlatten(IComplexNDArray... flatten) {
        IComplexNDArray ret = INSTANCE.complexFlatten(flatten);
        logCreationIfNecessary(ret);
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
        logCreationIfNecessary(ret);
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
    public static INDArray toFlattened(INDArray... matrices) {
        return INSTANCE.toFlattened(matrices);
    }

    /**
     * Create a long row vector of all of the given ndarrays/
     * @param order order in which to flatten ndarrays
     * @param matrices the matrices to create the flattened ndarray for

     * @return the flattened representation of
     * these ndarrays
     */
    public static INDArray toFlattened(char order, INDArray... matrices) {
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
        logCreationIfNecessary(ret);
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
     * @param filePath
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
     * Array written to outputstream
     *
     * @param os the outputstream stream ndarray
     * @param split
     * @deprecated custom col separators are no longer supported; uses ","
     * @param precision
     * @deprecated precision can no longer be specified. The array is written in scientific notation.
     * Use {@link #writeTxtString(INDArray, OutputStream)}
     */
    public static void writeTxtString(INDArray write, OutputStream os, String split, int precision) {
        writeTxtString(write,os);
    }

    /**
     *
     * @param write
     * @param os
     * @param precision
     * @deprecated precision can no longer be specified. The array is written in scientific notation.
     * Use {@link #writeTxtString(INDArray, OutputStream)}
     */
    @Deprecated
    public static void writeTxtString(INDArray write, OutputStream os, int precision) {
        writeTxtString(write,os);
    }

    /**
     * @param write
     * @param os
     * @param split
     * @deprecated column separator can longer be specified; Uses ","
     * Use {@link #writeTxtString(INDArray, OutputStream)} instead
     */
    @Deprecated
    public static void writeTxtString(INDArray write, OutputStream os, String split) {
        writeTxtString(write, os);
    }

    /**
     * Write ndarray as text to output stream
     * @param write
     * @param os
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
    public static INDArray readNumpy(InputStream filePath, String split) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(filePath));
        String line;
        List<float[]> data2 = new ArrayList<>();
        int numColumns = -1;
        INDArray ret;
        while ((line = reader.readLine()) != null) {
            String[] data = line.trim().split(split);
            if (numColumns < 0) {
                numColumns = data.length;

            } else
                assert data.length == numColumns : "Data has inconsistent number of columns";
            data2.add(readSplit(data));


        }

        ret = Nd4j.create(data2.size(), numColumns);
        for (int i = 0; i < data2.size(); i++)
            ret.putRow(i, Nd4j.create(Nd4j.createBuffer(data2.get(i))));
        return ret;
    }

    private static float[] readSplit(String[] split) {
        float[] ret = new float[split.length];
        for (int i = 0; i < split.length; i++) {
            ret[i] = Float.parseFloat(split[i]);
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
    @Deprecated
    public static void writeNumpy(INDArray write, String filePath, String split) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filePath));
        for (int i = 0; i < write.rows(); i++) {
            StringBuilder sb = new StringBuilder();
            INDArray row = write.getRow(i);
            for (int j = 0; j < row.columns(); j++) {
                sb.append(row.getDouble(j));
                if (j < row.columns() - 1) // not the last element
                    sb.append(split);
            }
            sb.append("\n");
            writer.write(sb.toString());
        }

        writer.flush();
        writer.close();


    }

    /**
     * Read line via input streams
     *
     * @param filePath the input stream ndarray
     * @param split    the split separator
     * @return the read txt method
     */
    public static INDArray readNumpy(String filePath, String split) throws IOException {
        return readNumpy(new FileInputStream(filePath), split);
    }

    /**
     * Read line via input streams
     *
     * @param filePath the input stream ndarray
     * @return the read txt method
     */
    public static INDArray readNumpy(String filePath) throws IOException {
        return readNumpy(filePath, "\t");
    }



    /**
     * Raad an ndarray from an input stream
     * @param reader the input stream to use
     * @return the given ndarray
     * @throws IOException
     */
    public static INDArray read(InputStream reader) throws IOException {
        return read(new DataInputStream(reader));

    }

    /**
     * Parse a complex number
     * @param val the string to parse
     * @return the parsed complex number
     */
    public static IComplexNumber parseComplexNumber(String val) {
        // real + " - " + (-imag) + "i"
        String[] split = val.split(" ");
        double real = Double.valueOf(split[0]);
        char op = split[1].charAt(0);
        //grab all but the i
        double imag = Double.valueOf(split[2].substring(0, split[2].length() - 1));
        return Nd4j.createComplexNumber(real, op == '-' ? -imag : imag);
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
            int[] theShape = null;
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
                        newArr = Nd4j.trueScalar(0);
                    } else {
                        String[] shapeArr = shapeString.split(",");
                        rank = shapeArr.length;
                        theShape = new int[rank];
                        for (int i = 0; i < rank; i++) {
                            theShape[i] = Integer.parseInt(shapeArr[i]);
                        }
                        if (theOrder == 'f' && theShape[rank-1] == 1) {
                            //Hack fix for tad issue with 'f' order and rank-1 dim shape == 1
                            newArr = Nd4j.zeros(theShape, 'c');
                        }
                        else {
                            newArr = Nd4j.zeros(theShape, theOrder);
                        }
                        subsetArr = new double[theShape[rank - 1]];
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
                        for (int i = 0; i < theShape[rank - 1]; i++) {
                            try {
                                BigDecimal number = (BigDecimal) format.parse(entries[i]);
                                subsetArr[i] = number.doubleValue();
                            } catch (ParseException e) {
                                e.printStackTrace();
                            }
                        }
                        INDArray subTensor = Nd4j.create(subsetArr);
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
        return newArr;
    }




    /**
     * Read line via input streams
     *
     * @param ndarray the input stream ndarray
     * @param  sep character, defaults to ","
     * @return NDArray
     * Custom separators no longer supported; Use {@link #writeTxt(INDArray, String)} along with {@link #readTxt(String)}
     */
    @Deprecated
    public static INDArray readTxtString(InputStream ndarray, String sep) {
        return readTxtString(ndarray);
    }


    /**
     * Read line via input streams
     *
     * @param filePath the input stream ndarray
     * @param  sep character, defaults to ","
     * @return NDArray
     * Custom separators no longer supported; Use {@link #writeTxt(INDArray, String)} along with {@link #readTxt(String)}
     */
    @Deprecated
    public static INDArray readTxt(String filePath, String sep) {
        return readTxt(filePath);
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
     *
     * @param data
     * @param shapeInfo
     * @return
     */
    public static INDArray createArrayFromShapeBuffer(DataBuffer data, DataBuffer shapeInfo) {
        int rank = Shape.rank(shapeInfo);
        long offset = 0;
        INDArray result = Nd4j.create(data, toIntArray(rank, Shape.shapeOf(shapeInfo)),
                toIntArray(rank, Shape.stride(shapeInfo)), offset, Shape.order(shapeInfo));
        if (data instanceof CompressedDataBuffer)
            result.markAsCompressed(true);

        return result;
    }

    /**
     *
     * @param data
     * @param shapeInfo
     * @return
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
        DataBuffer shapeInformation = Nd4j.createBufferDetached(new long[1], DataBuffer.Type.LONG);
        shapeInformation.read(dis);
        val length = Shape.length(shapeInformation);
        DataBuffer data = CompressedDataBuffer.readUnknown(dis, length);
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
     * Read in an ndarray from a data input stream
     *
     * @param dis the data input stream to read from
     * @return the ndarray
     * @throws IOException
     */
    public static IComplexNDArray readComplex(DataInputStream dis) throws IOException {
        int dimensions = dis.readInt();
        int[] shape = new int[dimensions];
        int[] stride = new int[dimensions];

        for (int i = 0; i < dimensions; i++)
            shape[i] = dis.readInt();
        for (int i = 0; i < dimensions; i++)
            stride[i] = dis.readInt();
        String dataType = dis.readUTF();

        String type = dis.readUTF();

        if (!type.equals("complex"))
            throw new IllegalArgumentException("Trying to read in a real ndarray");

        if (dataType.equals("double")) {
            double[] data = ArrayUtil.readDouble(ArrayUtil.prod(shape), dis);
            return createComplex(data, shape, stride, 0);
        }

        double[] data = ArrayUtil.read(ArrayUtil.prod(shape), dis);
        return createComplex(data, shape, stride, 0);
    }

    /**
     * Write an ndarray to the specified outputs tream
     *
     * @param arr              the array to write
     * @param dataOutputStream the data output stream to write to
     * @throws IOException
     */
    public static void writeComplex(IComplexNDArray arr, DataOutputStream dataOutputStream) throws IOException {
        /*
        dataOutputStream.writeInt(arr.shape().length);
        for (int i = 0; i < arr.shape().length; i++)
            dataOutputStream.writeInt(arr.size(i));
        for (int i = 0; i < arr.stride().length; i++)
            dataOutputStream.writeInt(arr.stride()[i]);
        dataOutputStream.writeUTF(dataType() == DataBuffer.Type.FLOAT ? "float" : "double");

        dataOutputStream.writeUTF("complex");

        if (dataType() == DataBuffer.Type.DOUBLE)
            ArrayUtil.write(arr.data().asDouble(), dataOutputStream);
        else
            ArrayUtil.write(arr.data().asFloat(), dataOutputStream);

        */
        throw new UnsupportedEncodingException();
    }

    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     *
     * @param reverse the matrix to reverse
     * @return the reversed matrix
     */
    public static INDArray rot(INDArray reverse) {
        INDArray ret = INSTANCE.rot(reverse);
        logCreationIfNecessary(ret);
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
        return Nd4j.getExecutioner().exec(new OldReverse(reverse)).z();
    }

    /**
     * Array of evenly spaced values.
     *
     * @param begin the begin of the range
     * @param end   the end of the range
     * @return the range vector
     */
    public static INDArray arange(double begin, double end) {
        INDArray ret = INSTANCE.arange(begin, end);
        logCreationIfNecessary(ret);
        return ret;
    }


    /**
     * Array of evenly spaced values.
     *
     * @param end   the end of the range
     * @return the range vector
     */
    public static INDArray arange(double end) {
        return arange(0, end);
    }

    /**
     * Create double
     *
     * @param real real component
     * @param imag imag component
     * @return
     */
    public static IComplexFloat createFloat(float real, float imag) {
        return INSTANCE.createFloat(real, imag);
    }

    /**
     * Create an instance of a complex double
     *
     * @param real the real component
     * @param imag the imaginary component
     * @return a new imaginary double with the specified real and imaginary components
     */
    public static IComplexDouble createDouble(double real, double imag) {
        return INSTANCE.createDouble(real, imag);
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
     * @param k the kth diagonal to getDouble
     * @return new matrix
     */
    public static IComplexNDArray diag(IComplexNDArray x, int k) {
        /*
        if (x.isScalar())
            return x.dup();

        if (x.isVector()) {
            IComplexNDArray m = Nd4j.createComplex(x.length(), x.length());
            IComplexNDArray xLinear = x.linearView();

            for (int i = 0; i < x.length(); i++)
                m.putScalar(i, i, xLinear.getComplex(i));

            return m;
        } else if (x.isMatrix()) {
            int vectorLength = x.rows() - k;
            IComplexNDArray ret = Nd4j.createComplex(new int[] {vectorLength, 1});
            for (int i = 0; i < vectorLength; i++) {
                ret.putScalar(i, x.getComplex(i, i));
            }

            return ret;
        }


        throw new IllegalArgumentException("Illegal input for diagonal of shape " + x.shape().length);
        */
        throw new UnsupportedOperationException();
    }

    /**
     * Creates a new matrix where the values of the given vector are the diagonal values of
     * the matrix if a vector is passed in, if a matrix is returns the kth diagonal
     * in the matrix
     *
     * @param x the diagonal values
     * @param k the kth diagonal to getDouble
     * @return new matrix
     */
    public static INDArray diag(INDArray x, int k) {
        INDArray ret = Nd4j.create(new long[] {x.length(),x.length()});
        Nd4j.getExecutioner().exec(new Diag(new INDArray[]{x},new INDArray[]{ret}));
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
    public static IComplexNDArray diag(IComplexNDArray x) {
        return diag(x, 0);

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
     * @param source
     * @param probs
     * @param target
     * @return
     */
    public static INDArray choice(@NonNull INDArray source, @NonNull INDArray probs, @NonNull INDArray target,
                                  @NonNull org.nd4j.linalg.api.rng.Random rng) {
        if (source.length() != probs.length())
            throw new ND4JIllegalStateException("Nd4j.choice() requires lengths of Source and Probs to be equal");

        return Nd4j.getExecutioner().exec(new Choice(source, probs, target), rng);
    }

    /**
     * This method samples value from Source array to Target, with probabilites provided in Probs argument
     *
     * @param source
     * @param probs
     * @param target
     * @return
     */
    public static INDArray choice(INDArray source, INDArray probs, INDArray target) {
        return choice(source, probs, target, Nd4j.getRandom());
    }

    /**
     * This method returns new INDArray instance, sampled from Source array with probabilities given in Probs
     *
     * @param source
     * @param probs
     * @param numSamples
     * @return
     */
    public static INDArray choice(INDArray source, INDArray probs, int numSamples,
                                  @NonNull org.nd4j.linalg.api.rng.Random rng) {
        if (numSamples < 1)
            throw new ND4JIllegalStateException("Nd4j.choice() numSamples must be positive value");

        return choice(source, probs, createUninitialized(numSamples), rng);
    }

    /**
     * This method returns new INDArray instance, sampled from Source array with probabilities given in Probs
     *
     * @param source
     * @param probs
     * @param numSamples
     * @return
     */
    public static INDArray choice(INDArray source, INDArray probs, int numSamples) {
        return choice(source, probs, numSamples, Nd4j.getRandom());
    }

    public static INDArray appendBias(INDArray... vectors) {
        INDArray ret = INSTANCE.appendBias(vectors);
        logCreationIfNecessary(ret);
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

    /**
     * Perform an operation along a diagonal
     *
     * @param x    the ndarray to perform the operation on
     * @param func the operation to perform
     */
    public static void doAlongDiagonal(IComplexNDArray x, Function<IComplexNumber, IComplexNumber> func) {
        if (x.isMatrix())
            for (int i = 0; i < x.rows(); i++)
                x.putScalar(i, i, func.apply(x.getComplex(i, i)));
    }


    ////////////////////// RANDOM ///////////////////////////////
    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     *
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    public static IComplexNDArray complexRand(int... shape) {
        INDArray based = Nd4j.rand(new int[] {1, ArrayUtil.prod(shape) * 2});
        IComplexNDArray ret = Nd4j.createComplex(based.data(), shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     *
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape) {
        INDArray ret = createUninitialized(shape, order()); //INSTANCE.rand(shape, Nd4j.getRandom());
        logCreationIfNecessary(ret);
        return rand(ret);
    }

    public static INDArray rand(long[] shape) {
        INDArray ret = createUninitialized(shape, order()); //INSTANCE.rand(shape, Nd4j.getRandom());
        logCreationIfNecessary(ret);
        return rand(ret);
    }

    /**
     * Create a random ndarray with the given shape and array order
     *
     * @param order the order of the ndarray to return
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(char order, int[] shape) {
        INDArray ret = Nd4j.createUninitialized(shape, order); //INSTANCE.rand(order, shape);
        logCreationIfNecessary(ret);
        return rand(ret);
    }

    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows, int columns) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        INDArray ret = createUninitialized(new int[] {rows, columns}, Nd4j.order());//INSTANCE.rand(rows, columns, Nd4j.getRandom());
        logCreationIfNecessary(ret);
        return rand(ret);
    }

    /**
     * Create a random ndarray with the given shape and output order
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(char order, int rows, int columns) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        INDArray ret = createUninitialized(new int[] {rows, columns}, order);//INSTANCE.rand(order, rows, columns);
        logCreationIfNecessary(ret);
        return rand(ret);
    }

    /**
     * Create a random ndarray with the given shape using given seed
     *
     * @param shape the shape of the ndarray
     * @param seed  the  seed to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape, long seed) {
        INDArray ret = createUninitialized(shape, Nd4j.order());//;INSTANCE.rand(shape, seed);
        logCreationIfNecessary(ret);
        return rand(ret, seed);
    }

    /**
     * Create a random ndarray with the given shape using the given seed
     *
     * @param seed the seed
     * @param shape  the shape
     * @return the random ndarray with the specified shape
     */
    @Deprecated
    public static INDArray rand(long seed, int... shape) {
        INDArray ret = createUninitialized(shape, Nd4j.order());//INSTANCE.rand(shape, seed);
        logCreationIfNecessary(ret);
        return rand(ret, seed);
    }

    /**
     * Create a random ndarray with the given shape using the given seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param seed    the  seed to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows, int columns, long seed) {
        INDArray ret = createUninitialized(new int[] {rows, columns}, Nd4j.order());
        logCreationIfNecessary(ret);
        return rand(ret, seed);
    }

    /**
     * Create a random ndarray with the given shape using the given RandomGenerator
     *
     * @param shape the shape of the ndarray
     * @param rng     the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape, org.nd4j.linalg.api.rng.Random rng) {
        INDArray ret = createUninitialized(shape, Nd4j.order()); //INSTANCE.rand(shape, rng);
        logCreationIfNecessary(ret);
        return rand(ret, rng);
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param dist  distribution to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape, Distribution dist) {
        //INDArray ret = INSTANCE.rand(shape, dist);
        //logCreationIfNecessary(ret);
        return dist.sample(shape);
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param dist  distribution to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(long[] shape, Distribution dist) {
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
    public static INDArray rand(int rows, int columns, org.nd4j.linalg.api.rng.Random rng) {
        INDArray ret = createUninitialized(new int[] {rows, columns}, order());//INSTANCE.rand(rows, columns, rng);
        logCreationIfNecessary(ret);
        return rand(ret, rng);
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
    public static INDArray rand(int[] shape, double min, double max, org.nd4j.linalg.api.rng.Random rng) {
        INDArray ret = createUninitialized(shape, order()); //INSTANCE.rand(shape, min, max, rng);
        logCreationIfNecessary(ret);
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
    public static INDArray rand(long[] shape, double min, double max, org.nd4j.linalg.api.rng.Random rng) {
        INDArray ret = createUninitialized(shape, order()); //INSTANCE.rand(shape, min, max, rng);
        logCreationIfNecessary(ret);
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
    public static INDArray rand(int rows, int columns, double min, double max, org.nd4j.linalg.api.rng.Random rng) {
        INDArray ret = createUninitialized(rows, columns);//INSTANCE.rand(rows, columns, min, max, rng);
        logCreationIfNecessary(ret);
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
     * Random normal using the current time stamp
     * as the seed
     *
     * @param shape the shape of the ndarray
     * @return
     */
    public static INDArray randn(int[] shape) {
        INDArray ret = Nd4j.createUninitialized(shape, order());
        logCreationIfNecessary(ret);
        return randn(ret);
    }

    public static INDArray randn(long[] shape) {
        INDArray ret = Nd4j.createUninitialized(shape, order());
        logCreationIfNecessary(ret);
        return randn(ret);
    }

    /**
     * Random normal N(0,1) with the specified shape and array order
     *
     * @param order order of the output ndarray
     * @param shape the shape of the ndarray
     */
    public static INDArray randn(char order, int[] shape) {
        INDArray ret = Nd4j.createUninitialized(shape, order);
        logCreationIfNecessary(ret);
        return randn(ret);
    }

    /**
     * Random normal N(0,1) with the specified shape and array order
     *
     * @param order order of the output ndarray
     * @param shape the shape of the ndarray
     */
    public static INDArray randn(char order, long[] shape) {
        INDArray ret = Nd4j.createUninitialized(shape, order);
        logCreationIfNecessary(ret);
        return randn(ret);
    }

    /**
     * Random normal using the specified seed
     *
     * @param shape the shape of the ndarray
     * @return
     */
    public static INDArray randn(int[] shape, long seed) {
        INDArray ret = Nd4j.createUninitialized(shape, order());
        logCreationIfNecessary(ret);
        return randn(ret, seed);
    }

    /**
     * Random normal using the current time stamp
     * as the seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    public static INDArray randn(long rows, long columns) {
        INDArray ret = Nd4j.createUninitialized(new long[]{rows, columns}, order());
        logCreationIfNecessary(ret);
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
        logCreationIfNecessary(ret);
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
        logCreationIfNecessary(ret);
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
    public static INDArray randn(long rows, long columns, org.nd4j.linalg.api.rng.Random r) {
        INDArray ret = Nd4j.createUninitialized(new long[]{rows, columns}, order());
        logCreationIfNecessary(ret);
        return randn(ret, r);
    }

    /**
     * Random normal using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return
     */
    public static INDArray randn(int[] shape, org.nd4j.linalg.api.rng.Random r) {
        final INDArray ret = Nd4j.createUninitialized(shape, order());
        logCreationIfNecessary(ret);
        return randn(ret, r);
    }

    /**
     * Random normal using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return
     */
    public static INDArray randn(long[] shape, org.nd4j.linalg.api.rng.Random r) {
        final INDArray ret = Nd4j.createUninitialized(shape, order());
        logCreationIfNecessary(ret);
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
    public static INDArray rand(INDArray target, org.nd4j.linalg.api.rng.Random rng) {
        return getExecutioner().exec(new UniformDistribution(target), rng);
    }

    /**
     * Fill the given ndarray with random numbers drawn from the given distribution
     *
     * @param target  target array
     * @param dist  distribution to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(INDArray target, Distribution dist) {
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
    public static INDArray rand(INDArray target,  double min, double max, org.nd4j.linalg.api.rng.Random rng) {
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
    public static INDArray randn(INDArray target, org.nd4j.linalg.api.rng.Random rng) {
        return getExecutioner().exec(new GaussianDistribution(target), rng);
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
     * Creates a row vector with the data
     *
     * @param list the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(List<? extends Number> list) {
        INDArray array = create(list.size());
        int cnt = 0;
        if (dataType() == DataBuffer.Type.DOUBLE) {
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
     * Creates an ndarray with the specified data
     *
     * @param data the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray createComplex(float[] data) {
        return createComplex(data, Nd4j.order());

    }

    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(double[] data) {
        return create(data, order());
    }

    /**
     *
     * @param data
     * @return
     */
    public static INDArray create(float[][] data) {
        return INSTANCE.create(data);
    }

    /**
     *
     * @param data
     * @param ordering
     * @return
     */
    public static INDArray create(float[][] data, char ordering) {
        return INSTANCE.create(data, ordering);
    }


    /**
     * Create an ndarray based on the given data layout
     *
     * @param data the data to use
     * @return an ndarray with the given data layout
     */
    public static INDArray create(double[][] data) {
        return INSTANCE.create(data);
    }


    public static INDArray create(double[][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length});
    }

    public static INDArray create(float[][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length});
    }

    public static INDArray create(int[][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length});
    }

    public static INDArray create(double[][][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length, data[0][0][0].length});
    }

    public static INDArray create(float[][][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length, data[0][0][0].length});
    }

    public static INDArray create(int[][][][] data) {
        return create(ArrayUtil.flatten(data), new int[] {data.length, data[0].length, data[0][0].length, data[0][0][0].length});
    }


    /**
     *
     * @param data
     * @param ordering
     * @return
     */
    public static INDArray create(double[][] data, char ordering) {
        return INSTANCE.create(data, ordering);
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arr the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(INDArray arr) {
        if (arr instanceof IComplexNDArray)
            return (IComplexNDArray) arr;
        IComplexNDArray ret = INSTANCE.createComplex(arr);
        logCreationIfNecessary(ret);
        return ret;
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
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray createComplex(int columns) {
        return createComplex(columns, order());
    }


    /**
     * Create double based on real and imaginary
     *
     * @param real real component
     * @param imag imag component
     * @return
     */
    public static IComplexNumber createComplexNumber(Number real, Number imag) {
        if (dataType() == DataBuffer.Type.FLOAT)
            return INSTANCE.createFloat(real.floatValue(), imag.floatValue());
        return INSTANCE.createDouble(real.doubleValue(), imag.doubleValue());

    }

    /**
     * Create a complex ndarray based on the
     * real and imaginary
     *
     * @param real the real numbers
     * @param imag the imaginary components
     * @return the complex
     */
    public static IComplexNDArray createComplex(INDArray real, INDArray imag) {
        throw new UnsupportedOperationException();
        /*
        assert Shape.shapeEquals(real.shape(), imag.shape());
        IComplexNDArray ret = Nd4j.createComplex(real.shape());
        INDArray realLinear = real.linearView();
        INDArray imagLinear = imag.linearView();
        IComplexNDArray retLinear = ret.linearView();
        for (int i = 0; i < ret.length(); i++) {
            retLinear.putScalar(i, Nd4j.createComplexNumber(realLinear.getDouble(i), imagLinear.getDouble(i)));
        }
        logCreationIfNecessary(ret);
        return ret;
        */
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param data the data to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(IComplexNumber[] data, int[] shape) {
        shape = getEnsuredShape(shape);
        IComplexNDArray ret = INSTANCE.createComplex(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param arrs the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(List<IComplexNDArray> arrs, int[] shape) {
        shape = getEnsuredShape(shape);
        IComplexNDArray ret = INSTANCE.createComplex(arrs, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(float[] data, char order) {
        INDArray ret = INSTANCE.create(data, order);
        logCreationIfNecessary(ret);
        return ret;
    }

    private static IComplexNDArray createComplex(float[] data, char order) {
        IComplexNDArray ret = INSTANCE.createComplex(data, order);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a row vector with the data
     *
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(double[] data, char order) {
        INDArray ret = INSTANCE.create(data, order);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified data
     *
     * @param data the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray createComplex(double[] data, char order) {
        IComplexNDArray ret = INSTANCE.createComplex(data, Nd4j.getComplexStrides(new int[] {1, data.length}, order), 0,
                order);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(int columns, char order) {
        if (columns < 1)
            throw new ND4JIllegalStateException("Number of columns should be positive for new INDArray");

        INDArray ret = INSTANCE.create(new int[] {1, columns}, Nd4j.getStrides(new int[] {1, columns}, order), 0,
                order);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray zeros(int columns, char order) {
        return Nd4j.create(columns, order);
    }

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray createComplex(int columns, char order) {
        int[] shape = new int[] {1, columns};
        IComplexNDArray ret =
                INSTANCE.createComplex(new int[] {1, columns}, Nd4j.getComplexStrides(shape, order), 0, order);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a complex ndarray from the passed in indarray
     *
     * @param data the data to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, long offset, char order) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, offset, order);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * This method creates new 0D INDArray, aka scalar.
     *
     * PLEASE NOTE: Temporary method, added to ensure backward compatibility
     * @param scalar
     * @return
     */
    public static INDArray trueScalar(Number scalar) {
        val ret = INSTANCE.trueScalar(scalar);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray trueVector(float[] data) {
        val ret = INSTANCE.trueVector(data);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray trueVector(double[] data) {
        val ret = INSTANCE.trueVector(data);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(float[] data, int[] shape) {
        if (shape.length == 0 && data.length == 1) {
            return trueScalar(data[0]);
        }

        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array doesn't match data length");
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(float[] data, long[] shape) {
        if (shape.length == 0 && data.length == 1) {
            return trueScalar(data[0]);
        }

        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array doesn't match data length");
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(double[] data, long[] shape) {
        if (shape.length == 0 && data.length == 1) {
            return trueScalar(data[0]);
        }

        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array doesn't match data length");
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(float[] data, int[] shape) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new int[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array doesn't match data length");
        }

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(double[] data, int[] shape) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape) + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(double[] data, int[] shape) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new int[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data   the data to use with tne ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(float[] data, int[] shape, int[] stride) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data   the data to use with tne ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(double[] data, int[] shape, int[] stride) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, long offset) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride, offset);
        logCreationIfNecessary(ret);
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
    public static INDArray create(double[] data, int[] shape, int[] stride, long offset) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new int[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset);
        logCreationIfNecessary(ret);
        return ret;
    }


    public static INDArray create(double[] data, long[] shape, long[] stride, long offset, char order) {
        shape = getEnsuredShape(shape);

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset, order);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data   the data to use with the ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, long offset) {
        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
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
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data    the data to use with the ndarray
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(float[] data, int rows, int columns, int[] stride, long offset) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        IComplexNDArray ret = INSTANCE.createComplex(data, rows, columns, stride, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
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
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(double[] data, int rows, int columns, int[] stride, long offset) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        IComplexNDArray ret = INSTANCE.createComplex(data, rows, columns, stride, offset);
        logCreationIfNecessary(ret);
        return ret;
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(float[] data, int[] shape, long offset) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new int[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, offset, Nd4j.order());
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(float[] data, long[] shape, long offset) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new long[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, offset, Nd4j.order());
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data, int[] shape, long offset, char ordering) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new int[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, Nd4j.getStrides(shape, ordering), offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(double[] data, long[] shape, long offset, char ordering) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new long[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, Nd4j.getStrides(shape, ordering), offset, ordering);
        logCreationIfNecessary(ret);
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
    public static INDArray create(float[] data, int[] shape, int[] stride, long offset) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new int[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray create(List<INDArray> list, int[] shape) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(list, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(List<INDArray> list, long[] shape) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(list, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows, int columns, int[] stride, long offset) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        IComplexNDArray ret = INSTANCE.createComplex(rows, columns, stride, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(int rows, int columns, int[] stride, long offset) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        INDArray ret = INSTANCE.create(rows, columns, stride, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray zeros(int rows, int columns, int[] stride, long offset) {
        return create(rows, columns, stride, offset);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape, int[] stride, long offset) {
        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(shape, stride, offset);
        logCreationIfNecessary(ret);
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
    public static INDArray create(int[] shape, int[] stride, long offset) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, stride, offset);
        logCreationIfNecessary(ret);
        return ret;

    }

    public static INDArray zeros(int[] shape, int[] stride, long offset) {
        return create(shape, stride, offset);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows, int columns, int[] stride) {
        return createComplex(rows, columns, stride, order());
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

    public static INDArray zeros(int rows, int columns, int[] stride) {
        return create(rows, columns, stride, order());
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape, int[] stride) {
        return createComplex(shape, stride, order());
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

    public static INDArray create(long[] shape, long[] stride) {
        return create(shape, stride, order());
    }


    /**
     *
     * @param shape
     * @param stride
     * @return
     */
    public static INDArray zeros(int[] shape, int[] stride) {
        return create(shape, stride);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows, int columns) {
        return createComplex(rows, columns, order());
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
     * Creates a complex ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int... shape) {
        return createComplex(shape, order());
    }

    public static IComplexNDArray createComplex(long... shape) {
        //return createComplex(shape, order());
        throw new ND4JComplexNumbersNotSupportedException();
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray create(int... shape) {
        return create(shape, order());
    }


    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray create(long... shape) {
        return create(shape, order());
    }

    /**
     *
     * @param data
     * @param shape
     * @param stride
     * @param ordering
     * @param offset
     * @return
     */
    public static INDArray create(float[] data, int[] shape, int[] stride, char ordering, long offset) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new int[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @param ordering
     * @param offset
     * @return
     */
    public static INDArray create(float[] data, int[] shape, char ordering, long offset) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape) + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, getStrides(shape, ordering), offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }


    public static INDArray create(double[] data, int[] shape, char ordering, long offset) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape) + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, getStrides(shape, ordering), offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @param strides
     * @param offset
     * @return
     */
    public static INDArray create(DataBuffer data, int[] shape, int[] strides, long offset) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(data, shape, strides, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(DataBuffer data, long[] shape, long[] strides, long offset) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(data, shape, strides, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    public static INDArray create(DataBuffer data, int[] shape, long offset) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(data, shape, getStrides(shape), offset);
        logCreationIfNecessary(ret);
        return ret;

    }

    /**
     *
     * @param data
     * @param newShape
     * @param newStride
     * @param offset
     * @param ordering
     * @return
     */
    public static INDArray create(DataBuffer data, int[] newShape, int[] newStride, long offset, char ordering) {
        checkShapeValues(newShape);

        INDArray ret = INSTANCE.create(data, newShape, newStride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, char ordering) {
        checkShapeValues(newShape);

        INDArray ret = INSTANCE.create(data, newShape, newStride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @param newStrides
     * @param offset
     * @return
     */
    public static IComplexNDArray createComplex(DataBuffer data, int[] shape, int[] newStrides, long offset) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, newStrides, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    public static IComplexNDArray createComplex(DataBuffer data, int[] shape, long offset) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @param newStrides
     * @param offset
     * @param ordering
     * @return
     */
    public static IComplexNDArray createComplex(DataBuffer data, int[] shape, int[] newStrides, long offset,
                                                char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, newStrides, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     * @return
     */
    public static IComplexNDArray createComplex(DataBuffer data, int[] shape, long offset, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @return
     */
    public static INDArray create(DataBuffer data, int[] shape) {
        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(DataBuffer data, long[] shape) {
        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @return
     */
    public static IComplexNDArray createComplex(DataBuffer data, int[] shape) {
        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param buffer
     * @return
     */
    public static INDArray create(DataBuffer buffer) {
        INDArray ret = INSTANCE.create(buffer);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a complex array from the given numbers
     * @param iComplexNumbers the numbers to use
     * @return the complex numbers
     */
    public static IComplexNDArray createComplex(IComplexNumber[] iComplexNumbers) {
        if (iComplexNumbers == null || iComplexNumbers.length < 1)
            throw new ND4JIllegalStateException("Number of complex numbers can't be < 1 for new INDArray");

        return createComplex(iComplexNumbers, new int[] {1, iComplexNumbers.length});
    }

    /**
     * Create a complex ndarray based on the specified matrices
     * @param iComplexNumbers the complex numbers
     * @return the complex numbers
     */
    public static IComplexNDArray createComplex(IComplexNumber[][] iComplexNumbers) {
        IComplexNDArray shape = Nd4j.createComplex(iComplexNumbers.length, iComplexNumbers[0].length);
        for (int i = 0; i < iComplexNumbers.length; i++) {
            for (int j = 0; j < iComplexNumbers[i].length; j++) {
                shape.putScalar(i, j, iComplexNumbers[i][j]);
            }
        }
        return shape;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data    the data to use with the ndarray
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(float[] data, int rows, int columns, int[] stride, long offset,
                                                char ordering) {
        int[] shape = new int[] {rows, columns};
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
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
        shape = getEnsuredShape(shape);

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param shape
     * @param dataType
     * @return
     */
    public static INDArray create(int[] shape, DataBuffer.Type dataType) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, dataType);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray zeros(int[] shape, DataBuffer.Type dataType) {
        return create(shape, dataType);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param data   the data to use with the ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, long offset, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
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
    public static INDArray create(double[] data, int[] shape, int[] stride, long offset, char ordering) {
        if (data.length == 1 && shape.length == 0)
            return trueScalar(data[0]);

        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] != data.length)
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(double[] data, int[] shape, char ordering) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new int[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(float[] data, int[] shape, char ordering) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new int[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(float[] data, long[] shape, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, 0, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(double[] data, long[] shape, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, 0, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(float[] data, int[] shape, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data  the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(double[] data, int[] shape, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data   the data to use with tne ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndrray with the specified shape
     *
     * @param data   the data to use with tne ndarray
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(double[] data, int rows, int columns, int[] stride, long offset,
                                                char ordering) {
        int[] shape = getEnsuredShape(rows, columns);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param data    the data to use with tne ndarray
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data, int rows, int columns, int[] stride, long offset, char ordering) {
        int[] shape = getEnsuredShape(rows, columns);

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(Nd4j.createBuffer(data), shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, long offset, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
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
    public static INDArray create(float[] data, int[] shape, int[] stride, long offset, char ordering) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new int[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray create(List<INDArray> list, int[] shape, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(list, shape, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows, int columns, int[] stride, long offset, char ordering) {
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        IComplexNDArray ret = INSTANCE.createComplex(new int[] {rows, columns}, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(int rows, int columns, int[] stride, long offset, char ordering) {
        int[] shape = getEnsuredShape(rows, columns);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray zeros(int rows, int columns, int[] stride, long offset, char ordering) {
        return create(rows, columns, stride, offset, ordering);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape, int[] stride, long offset, char ordering) {
        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
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
    public static INDArray create(int[] shape, int[] stride, long offset, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
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
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;

    }

    public static INDArray zeros(int[] shape, int[] stride, long offset, char ordering) {
        return create(shape, stride, offset, ordering);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows, int columns, int[] stride, char ordering) {
        int[] shape = getEnsuredShape(rows, columns);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(shape, stride, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride  the stride for the ndarray
     * @return the instance
     */
    public static INDArray create(int rows, int columns, int[] stride, char ordering) {
        int[] shape = getEnsuredShape(rows, columns);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, stride, 0, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray zeros(int rows, int columns, int[] stride, char ordering) {
        return create(rows, columns, stride, ordering);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape, int[] stride, char ordering) {
        return createComplex(shape, stride, 0, ordering);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape, int[] stride, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, stride, 0, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(long[] shape, long[] stride, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, stride, 0, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param shape
     * @param stride
     * @param ordering
     * @return
     */
    public static INDArray zeros(int[] shape, int[] stride, char ordering) {
        return create(shape, stride, ordering);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows, int columns, char ordering) {
        int[] shape = getEnsuredShape(rows, columns);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(shape, Nd4j.getComplexStrides(shape, ordering), 0, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    public static INDArray create(long rows, long columns, char ordering) {
        return create(new long[] {rows, columns}, ordering);
    }

    /**
     *
     * @param rows
     * @param columns
     * @param ordering
     * @return
     */
    public static INDArray zeros(int rows, int columns, char ordering) {
        return create(new int[] {rows, columns}, ordering);
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(createBuffer(ArrayUtil.prodLong(shape) * 2), shape, 0, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape, char ordering) {
        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        } else if (shape.length == 1) {
            shape = new int[] {1, shape[0]};
        }

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(long[] shape, char ordering) {
        //ensure shapes that wind up being scalar end up with the write shape

        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static void checkShapeValues(long[] shape) {
        for (long e: shape) {
            if (e < 1)
                throw new ND4JIllegalStateException("Invalid shape: Requested INDArray shape " + Arrays.toString(shape)
                        + " contains dimension size values < 1 (all dimensions must be 1 or more)");
        }
    }

    public static void checkShapeValues(int[] shape) {
        for (int e: shape) {
            if (e < 1)
                throw new ND4JIllegalStateException("Invalid shape: Requested INDArray shape " + Arrays.toString(shape)
                        + " contains dimension size values < 1 (all dimensions must be 1 or more)");
        }
    }

    protected static void checkShapeValues(int length, int[] shape) {
        checkShapeValues(shape);

        if (ArrayUtil.prodLong(shape) > length)
            throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                    + " doesn't match data length: " + length);
    }

    protected static void checkShapeValues(int length, long[] shape) {
        checkShapeValues(shape);

        if (ArrayUtil.prodLong(shape) > length)
            throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                    + " doesn't match data length: " + length);
    }


    /**
     * Creates an *uninitialized* ndarray with the specified shape and ordering.<br>
     * <b>NOTE</b>: The underlying memory (DataBuffer) will not be initialized. Don't use this unless you know what you are doing.
     *
     * @param shape the shape of the ndarray
     * @param ordering the order of the ndarray
     * @return the instance
     */
    public static INDArray createUninitialized(int[] shape, char ordering) {
        if (shape.length == 0)
            return trueScalar(0.0);

        shape = getEnsuredShape(shape);
        // now we allow 1D vectors
        /*else if (shape.length == 1) {
            shape = new int[] {1, shape[0]};
        }
        */

        checkShapeValues(shape);

        INDArray ret = INSTANCE.createUninitialized(shape, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }


    public static INDArray createUninitialized(long[] shape, char ordering) {
        if (shape.length == 0)
            return trueScalar(0.0);

        shape = getEnsuredShape(shape);
        // now we allow 1D vectors
        /*else if (shape.length == 1) {
            shape = new int[] {1, shape[0]};
        }
        */

        checkShapeValues(shape);

        INDArray ret = INSTANCE.createUninitialized(shape, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Cretes uninitialized INDArray detached from any (if any) workspace
     *
     * @param shape
     * @param ordering
     * @return
     */
    public static INDArray createUninitializedDetached(int[] shape, char ordering) {
        if (shape.length == 0)
            return trueScalar(0.0);

        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new int[] {1, 1};
        } else if (shape.length == 1) {
            shape = new int[] {1, shape[0]};
        }

        checkShapeValues(shape);

        INDArray ret = INSTANCE.createUninitializedDetached(shape, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray createUninitializedDetached(long[] shape, char ordering) {
        if (shape.length == 0)
            return trueScalar(0.0);

        //ensure shapes that wind up being scalar end up with the write shape
        if (shape.length == 1 && shape[0] == 0) {
            shape = new long[] {1, 1};
        } else if (shape.length == 1) {
            shape = new long[] {1, shape[0]};
        }

        checkShapeValues(shape);

        INDArray ret = INSTANCE.createUninitializedDetached(shape, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an *uninitialized* ndarray with the specified shape and default ordering.<br>
     * <b>NOTE</b>: The underlying memory (DataBuffer) will not be initialized. Don't use this unless you know what you are doing.
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray createUninitialized(int[] shape) {
        checkShapeValues(shape);
        //ensure shapes that wind up being scalar end up with the write shape
        return createUninitialized(shape, Nd4j.order());
    }

    public static INDArray createUninitialized(long[] shape) {
        checkShapeValues(shape);
        //ensure shapes that wind up being scalar end up with the write shape
        return createUninitialized(shape, Nd4j.order());
    }

    /**
     * Cretes uninitialized INDArray detached from any (if any) workspace
     *
     * @param shape
     * @return
     */
    public static INDArray createUninitializedDetached(int[] shape) {
        return createUninitializedDetached(shape, Nd4j.order());
    }

    /**
     * Cretes uninitialized INDArray detached from any (if any) workspace
     *
     * @param shape
     * @return
     */
    public static INDArray createUninitializedDetached(long[] shape) {
        return createUninitializedDetached(shape, Nd4j.order());
    }

    /**
     * This method creates an *uninitialized* ndarray of specified length and default ordering.
     *
     * PLEASE NOTE: Do not use this method unless you're 100% sure why you use it.
     *
     * @param length
     * @return
     */
    public static INDArray createUninitialized(int length) {
        if (length < 1)
            throw new IllegalStateException("INDArray length should be positive value");

        int[] shape = new int[] {1, length};

        INDArray ret = INSTANCE.createUninitialized(shape, order());
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray createUninitialized(long length) {
        if (length < 1)
            throw new IllegalStateException("INDArray length should be positive value");

        long[] shape = new long[] {1, length};

        INDArray ret = INSTANCE.createUninitialized(shape, order());
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Cretes uninitialized INDArray detached from any (if any) workspace
     *
     * @param length
     * @return
     */
    public static INDArray createUninitializedDetached(int length) {
        if (length < 1)
            throw new IllegalStateException("INDArray length should be positive value");

        int[] shape = new int[] {1, length};

        INDArray ret = INSTANCE.createUninitializedDetached(shape, order());
        logCreationIfNecessary(ret);
        return ret;
    }



    /**
     * Create complex ndarray
     *
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     * @return
     */
    public static IComplexNDArray createComplex(float[] data, int[] shape, long offset, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret =
                INSTANCE.createComplex(data, shape, Nd4j.getComplexStrides(shape, ordering), offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    public static IComplexNDArray createComplex(double[] data, int[] shape, long offset) {
        return createComplex(data, shape, offset, Nd4j.order());
    }

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    public static INDArray create(double[] data, int[] shape, long offset) {
        shape = getEnsuredShape(shape);

        if (shape.length == 1) {
            if (shape[0] == data.length) {
                shape = new int[] {1, data.length};
            } else
                throw new ND4JIllegalStateException("Shape of the new array " + Arrays.toString(shape)
                        + " doesn't match data length: " + data.length);
        }

        checkShapeValues(data.length, shape);

        INDArray ret = INSTANCE.create(data, shape, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @param ordering
     * @return
     */
    public static IComplexNDArray createComplex(double[] data, int[] shape, long offset, char ordering) {
        shape = getEnsuredShape(shape);

        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param dim
     * @return
     */
    public static IComplexNDArray createComplex(double[] dim) {
        IComplexNDArray ret = INSTANCE.createComplex(dim, new int[] {1, dim.length / 2});
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param data
     * @param shape
     * @param offset
     * @return
     */
    public static IComplexNDArray createComplex(float[] data, int[] shape, long offset) {
        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.createComplex(data, shape, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * @param data
     * @param columns
     * @param pointerB
     * @param pointerE
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCSR(double[] data, int[] columns, int[] pointerB, int[] pointerE, int[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCSR(data, columns, pointerB, pointerE, shape);

        return matrix;
    }

    /**
     * @param data
     * @param columns
     * @param pointerB
     * @param pointerE
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCSR(float[] data, int[] columns, int[] pointerB, int[] pointerE, int[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCSR(data, columns, pointerB, pointerE, shape);

        return matrix;
    }

    /**
     * @param data
     * @param columns
     * @param pointerB
     * @param pointerE
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCSR(DataBuffer data, int[] columns, int[] pointerB, int[] pointerE,
                                           int[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCSR(data, columns, pointerB, pointerE, shape);

        return matrix;
    }

    public static INDArray createSparseCOO(double[] data, int[][] indices, long[] shape) {
        throw new UnsupportedOperationException();
    }

    /**
     * @param data
     * @param indices
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(double[] data, int[][] indices, int[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(data, indices, shape);

        return matrix;
    }

    /**
     * @param data
     * @param indices
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(float[] data, int[][] indices, int[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(data, indices, shape);

        return matrix;
    }

    /**
     * @param data
     * @param indices
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(DataBuffer data, DataBuffer indices, int[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(data, indices, shape);

        return matrix;
    }

    /**
     * @param values a DataBuffer with the sparse non-null values
     * @param indices a DataBuffer with the indexes of the values
     * @param sparseInformation a DataBuffer containing the sparse information (flags, offsets and hidden dimensions)
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(DataBuffer values, DataBuffer indices, DataBuffer sparseInformation,
                                           int[] shape) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(values, indices, sparseInformation, shape);
        return matrix;
    }

    /**
     * @param values a DataBuffer with the sparse non-null values
     * @param indices a DataBuffer with the indexes of the values
     * @param sparseOffsets the sparse
     * @param flags an array that define the inactive dimension
     * @param hiddenDimensions an array containing the position of the hidden dimensions
     * @param underlyingRank the rank of the original ndarray
     * @param shape
     * @return a INDArray
     * */
    public static INDArray createSparseCOO(DataBuffer values, DataBuffer indices, long[] sparseOffsets, int[] flags,
                                           int[] shape, int[] hiddenDimensions, int underlyingRank) {
        INDArray matrix = SPARSE_INSTANCE.createSparseCOO(values, indices, sparseOffsets, flags, hiddenDimensions,
                underlyingRank, shape);
        return matrix;
    }


    ////////////////////// OTHER ///////////////////////////////



    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the rows of the sndarray
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray zeros(long rows, long columns) {
        INDArray ret = INSTANCE.zeros(rows, columns);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a matrix of zeros
     *
     * @param rows    te number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static INDArray complexZeros(int rows, int columns) {
        IComplexNDArray ret = INSTANCE.complexZeros(rows, columns);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray zeros(int columns) {
        return INSTANCE.zeros(columns);
    }

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray complexZeros(int columns) {
        return INSTANCE.complexZeros(columns);
    }

    /**
     *
     * @param num
     * @param value
     * @return
     */
    public static IComplexNDArray complexValueOf(int num, IComplexNumber value) {
        IComplexNDArray ret = INSTANCE.complexValueOf(num, value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param shape
     * @param value
     * @return
     */
    public static IComplexNDArray complexValueOf(int[] shape, IComplexNumber value) {
        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.complexValueOf(shape, value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     *
     * @param num
     * @param value
     * @return
     */
    public static IComplexNDArray complexValueOf(int num, double value) {
        return INSTANCE.complexValueOf(num, value);
    }

    /**
     *
     * @param shape
     * @param value
     * @return
     */
    public static IComplexNDArray complexValueOf(int[] shape, double value) {
        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.complexValueOf(shape, value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray.
     * Some people may know this as np.full
     *
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return the created ndarray
     */
    public static INDArray valueArrayOf(int[] shape, double value) {
        if (shape.length == 0)
            return trueScalar(value);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.valueArrayOf(shape, value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray.
     * Some people may know this as np.full
     *
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return the created ndarray
     */
    public static INDArray valueArrayOf(long[] shape, double value) {
        if (shape.length == 0)
            return trueScalar(value);

        checkShapeValues(shape);

        INDArray ret = INSTANCE.valueArrayOf(shape, value);
        logCreationIfNecessary(ret);
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
        INDArray ret = INSTANCE.valueArrayOf(new long[] {1, num}, value);
        logCreationIfNecessary(ret);
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
        if (rows < 1 || columns < 1)
            throw new ND4JIllegalStateException("Number of rows and columns should be positive for new INDArray");

        INDArray ret = INSTANCE.valueArrayOf(rows, columns, value);
        logCreationIfNecessary(ret);
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
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Empty like
     *
     * @param arr the array to create the ones like
     * @return ones in the shape of the given array
     */
    public static INDArray zerosLike(INDArray arr) {
        return create(arr.shape());
    }

    /**
     * Empty like
     *
     * @param arr the array to create the ones like
     * @return ones in the shape of the given array
     */
    public static INDArray emptyLike(INDArray arr) {
        return create(arr.shape());
    }

    /**
     * Ones like
     *
     * @param arr the array to create the ones like
     * @return ones in the shape of the given array
     */
    public static INDArray onesLike(INDArray arr) {
        return ones(arr.shape());
    }

    /**
     * Creates an ndarray
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray complexOnes(int rows, int columns) {
        IComplexNDArray ret = INSTANCE.complexOnes(rows, columns);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray ones(int columns) {
        INDArray ret = INSTANCE.ones(columns);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray complexOnes(int columns) {
        IComplexNDArray ret = INSTANCE.complexOnes(columns);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Concatenates two matrices horizontally. Matrices must have identical
     * numbers of rows.
     *
     * @param arrs the first matrix to concat
     */
    public static INDArray hstack(INDArray... arrs) {
        INDArray ret = INSTANCE.hstack(arrs);
        logCreationIfNecessary(ret);
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
        INDArray ret = INSTANCE.hstack(arrays);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Concatenates two matrices vertically. Matrices must have identical
     * numbers of columns.
     *
     * @param arrs
     */
    public static INDArray vstack(INDArray... arrs) {
        INDArray ret = INSTANCE.vstack(arrs);
        logCreationIfNecessary(ret);
        return ret;
    }


    /**
     * Concatenates two matrices vertically. Matrices must have identical
     * numbers of columns.
     *
     * @param arrs
     */
    public static INDArray vstack(Collection<INDArray> arrs) {
        INDArray[] arrays = arrs.toArray(new INDArray[0]);
        INDArray ret = INSTANCE.vstack(arrays);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * This method averages input arrays, and returns averaged array.
     * On top of that, averaged array is propagated to all input arrays
     *
     * @param arrays
     * @return
     */
    public static INDArray averageAndPropagate(INDArray target, INDArray[] arrays) {
        INDArray ret = INSTANCE.average(target, arrays);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * This method averages input arrays, and returns averaged array.
     * On top of that, averaged array is propagated to all input arrays
     *
     * @param arrays
     * @return
     */
    public static INDArray averageAndPropagate(INDArray[] arrays) {
        INDArray ret = INSTANCE.average(arrays);
        logCreationIfNecessary(ret);
        return ret;
    }


    /**
     * This method averages input arrays, and returns averaged array.
     * On top of that, averaged array is propagated to all input arrays
     *
     * @param arrays
     * @return
     */
    public static INDArray averageAndPropagate(Collection<INDArray> arrays) {
        INDArray ret = INSTANCE.average(arrays);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * This method averages input arrays, and returns averaged array.
     * On top of that, averaged array is propagated to all input arrays
     *
     * @param arrays
     * @return
     */
    public static INDArray averageAndPropagate(INDArray target, Collection<INDArray> arrays) {
        INDArray ret = INSTANCE.average(target, arrays);
        logCreationIfNecessary(ret);
        return ret;
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
    public static INDArray accumulate(INDArray... arrays) {
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
    public static INDArray pullRows(INDArray source, int sourceDimension, int[] indexes) {
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

        INDArray ret = INSTANCE.pullRows(source, sourceDimension, indexes, order);
        logCreationIfNecessary(ret);
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
    public static INDArray pullRows(INDArray source, INDArray destination, int sourceDimension, int[] indexes){
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

        INDArray ret = INSTANCE.pullRows(source, destination, sourceDimension, indexes);
        logCreationIfNecessary(ret);
        return ret;
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
    public static INDArray concat(int dimension, INDArray... toConcat) {
        if(dimension < 0) {
            dimension += toConcat[0].rank();
        }

        INDArray ret = INSTANCE.concat(dimension, toConcat);
        logCreationIfNecessary(ret);
        return ret;
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
    public static INDArray specialConcat(int dimension, INDArray... toConcat) {
        INDArray ret = INSTANCE.specialConcat(dimension, toConcat);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Concatneate ndarrays along a dimension
     *
     * @param dimension the dimension to concatneate along
     * @param toConcat  the ndarrays to concat
     * @return the concatted ndarrays with an output shape of
     * the ndarray shapes save the dimension shape specified
     * which is then the sum of the sizes along that dimension
     */
    public static IComplexNDArray concat(int dimension, IComplexNDArray... toConcat) {
        IComplexNDArray ret = INSTANCE.concat(dimension, toConcat);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndarray of zeros
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static INDArray zeros(int[] shape, char order) {
        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, order);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray zeros(long[] shape, char order) {
        checkShapeValues(shape);

        INDArray ret = INSTANCE.create(shape, order);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndarray of zeros
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static INDArray zeros(int... shape) {
        return Nd4j.create(shape);
    }


    /**
     * Create an ndarray of zeros
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static INDArray zeros(long... shape) {
        return Nd4j.create(shape);
    }

    /**
     * Create an ndarray of ones
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static IComplexNDArray complexZeros(int... shape) {
        checkShapeValues(shape);

        IComplexNDArray ret = INSTANCE.complexZeros(shape);
        logCreationIfNecessary(ret);
        return ret;

    }

    /**
     * Create an ndarray of ones
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static INDArray ones(int... shape) {
        checkShapeValues(shape);

        INDArray ret = INSTANCE.ones(shape);
        logCreationIfNecessary(ret);
        return ret;
    }


    public static INDArray ones(long... shape) {
        checkShapeValues(shape);

        INDArray ret = INSTANCE.ones(shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create an ndarray of ones
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static IComplexNDArray complexOnes(int... shape) {
        checkShapeValues(shape);
        IComplexNDArray ret = INSTANCE.complexOnes(shape);
        logCreationIfNecessary(ret);
        return ret;

    }

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    public static INDArray scalar(Number value, long offset) {
        return INSTANCE.scalar(value, offset);
    }

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray complexScalar(Number value, long offset) {
        IComplexNDArray arr = INSTANCE.complexScalar(value, offset);
        logCreationIfNecessary(arr);
        return arr;
    }

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    public static IComplexNDArray complexScalar(Number value) {
        return INSTANCE.complexScalar(value);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static INDArray scalar(double value, long offset) {
        INDArray ret = INSTANCE.scalar(value, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static INDArray scalar(float value, long offset) {
        INDArray ret = INSTANCE.scalar(value, offset);
        logCreationIfNecessary(ret);
        return ret;

    }

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    public static INDArray scalar(Number value) {
        INDArray ret = INSTANCE.scalar(value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     *              =     * @return the scalar nd array
     */
    public static INDArray scalar(double value) {
        INDArray ret = INSTANCE.scalar(value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     *              =     * @return the scalar nd array
     */
    public static INDArray scalar(float value) {
        INDArray ret = INSTANCE.scalar(value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray scalar(IComplexNumber value, long offset) {
        IComplexNDArray ret = INSTANCE.scalar(value, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexFloat value) {
        IComplexNDArray ret = INSTANCE.scalar(value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value the value of the scalar
     *              =     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexDouble value) {
        IComplexNDArray ret = INSTANCE.scalar(value);
        logCreationIfNecessary(ret);
        return ret;

    }

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    public static IComplexNDArray scalar(IComplexNumber value) {
        IComplexNDArray ret = INSTANCE.scalar(value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexFloat value, long offset) {
        IComplexNDArray ret = INSTANCE.scalar(value, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a scalar nd array with the specified value and offset
     *
     * @param value  the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexDouble value, long offset) {
        IComplexNDArray ret = INSTANCE.scalar(value, offset);
        logCreationIfNecessary(ret);
        return ret;

    }

    /**
     * Get the strides for the given order and shape
     *
     * @param shape the shape of the ndarray
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
     * @param shape the shape of the ndarray
     * @return the strides for the given shape
     * and order specified by NDArrays.order()
     */
    public static int[] getStrides(int[] shape) {
        return getStrides(shape, Nd4j.order());
    }

    /**
     * Get the strides based on the shape
     * and NDArrays.order()
     *
     * @param shape the shape of the ndarray
     * @return the strides for the given shape
     * and order specified by NDArrays.order()
     */
    public static long[] getStrides(long[] shape) {
        return getStrides(shape, Nd4j.order());
    }

    /**
     * An alias for repmat
     *
     * @param tile   the ndarray to tile
     * @param repeat the shape to repeat
     * @return the tiled ndarray
     */
    public static INDArray tile(INDArray tile, int... repeat) {
        int d = repeat.length;
        long[] shape = ArrayUtil.copy(tile.shape());
        long n = Math.max(tile.length(), 1);
        if (d < tile.rank()) {
            repeat = Ints.concat(ArrayUtil.nTimes(tile.rank() - d, 1), repeat);
        }
        for (int i = 0; i < shape.length; i++) {
            if (repeat[i] != 1) {
                tile = tile.reshape(-1, n).repeat(0, new int[] {repeat[i]});
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
     * @param shape the shape of the ndarray
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
     * @param shape the shape of the ndarray
     * @return the strides for the given shape
     * and order specified by NDArrays.order()
     */
    public static int[] getComplexStrides(int[] shape) {
        return getComplexStrides(shape, Nd4j.order());
    }

    public static long[] getComplexStrides(long[] shape) {
        return getComplexStrides(shape, Nd4j.order());
    }


    /**
     * Linspace with complex numbers
     * @param i
     * @param i1
     * @param i2
     * @return
     */
    public static IComplexNDArray complexLinSpace(int i, int i1, int i2) {
        return Nd4j.createComplex(Nd4j.linspace(i, i1, i2));

    }

    /**
     * Initializes nd4j
     */
    public synchronized void initContext() {
        try {
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

            String otherDtype = pp.toString(DTYPE);
            dtype = otherDtype.equals("float") ? DataBuffer.Type.FLOAT
                    : otherDtype.equals("half") ? DataBuffer.Type.HALF : DataBuffer.Type.DOUBLE;

            if (dtype == DataBuffer.Type.HALF && backend.getClass().getName().equals("CpuBackend")) {
                showAttractiveMessage(getMessageForNativeHalfPrecision());
            }

            if (Nd4j.dataType() != dtype){
                DataTypeUtil.setDTypeForContext(dtype);
            }

            compressDebug = pp.toBoolean(COMPRESSION_DEBUG);
            copyOnOps = pp.toBoolean(COPY_OPS, true);
            shouldInstrument = pp.toBoolean(INSTRUMENTATION);
            resourceManagerOn = pp.toBoolean(RESOURCE_MANGER_ON);
            executionMode = pp.toString(EXECUTION_MODE, "java").equals("java") ? OpExecutioner.ExecutionMode.JAVA
                    : OpExecutioner.ExecutionMode.NATIVE;
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


            instrumentationClazz = (Class<? extends Instrumentation>) Class
                    .forName(pp.toString(INSTRUMENTATION_CLASS, InMemoryInstrumentation.class.getName()));

            opFactoryClazz = (Class<? extends OpFactory>) Class
                    .forName(pp.toString(OP_FACTORY, DefaultOpFactory.class.getName()));

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

            instrumentation = instrumentationClazz.newInstance();
            OP_EXECUTIONER_INSTANCE = opExecutionerClazz.newInstance();
            Constructor c2 = ndArrayFactoryClazz.getConstructor(DataBuffer.Type.class, char.class);
            INSTANCE = (NDArrayFactory) c2.newInstance(dtype, ORDER);
            SPARSE_INSTANCE = sparseNDArrayClazz.newInstance();
            CONVOLUTION_INSTANCE = convolutionInstanceClazz.newInstance();
            BLAS_WRAPPER_INSTANCE = blasWrapperClazz.newInstance();
            SPARSE_BLAS_WRAPPER_INSTANCE = sparseBlasWrapperClazz.newInstance();
            DATA_BUFFER_FACTORY_INSTANCE = dataBufferFactoryClazz.newInstance();
            OP_FACTORY_INSTANCE = opFactoryClazz.newInstance();


            UNIT = Nd4j.createFloat(1, 0);
            ZERO = Nd4j.createFloat(0, 0);
            NEG_UNIT = Nd4j.createFloat(-1, 0);
            ENFORCE_NUMERICAL_STABILITY = pp.toBoolean(NUMERICAL_STABILITY);
            DISTRIBUTION_FACTORY = distributionFactoryClazz.newInstance();
            getExecutioner().setExecutionMode(executionMode);

            if (isFallback()) {
                fallbackMode.set(true);
                showAttractiveMessage(getMessageForFallback());
            } else {
                fallbackMode.set(false);
            }

            OP_EXECUTIONER_INSTANCE.printEnvironmentInformation();
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
        String fallback = System.getenv("ND4J_FALLBACK");
        if (fallback == null) {
            return false;
        }
        return (fallback.equalsIgnoreCase("true") || fallback.equalsIgnoreCase("1"));
    }
    
    /**
     * Get ensured shapes that wind up being scalar end up with the write shape
     * @param shape
     * @return 
     */
    private static int[] getEnsuredShape(int[] shape) {
        if (shape.length == 1 && shape[0] == 0) {
            return new int[]{1, 1};
        }
        return shape;
    }


    private static long[] getEnsuredShape(long[] shape) {
        if (shape.length == 1 && shape[0] == 0) {
            return new long[]{1, 1};
        }
        return shape;
    }
    
    /**
     * Get ensured shapes that wind up being scalar end up with the write shape
     * @param rows
     * @param columns
     * @return 
     */
    private static int[] getEnsuredShape(int rows, int columns) {
        return getEnsuredShape(new int[] {rows, columns});
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

    public static INDArray typeConversion(INDArray array, DataBuffer.TypeEx targetType) {
        return null;
    }

    /**
     * This method returns sizeOf(currentDataType), in bytes
     *
     * @return number of bytes per element
     */
    public static int sizeOfDataType() {
        return sizeOfDataType(Nd4j.dataType());
    }

    /**
     * This method returns size of element for specified dataType, in bytes
     *
     * @param dtype number of bytes per element
     * @return
     */
    public static int sizeOfDataType(DataBuffer.Type dtype) {
        switch (dtype) {
            case FLOAT:
                return 4;
            case HALF:
                return 2;
            case INT:
                return 4;
            default:
            case DOUBLE:
                return 8;
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
    public static INDArray pile(INDArray... arrays) {
        // if we have vectors as input, it's just vstack use case
        if (arrays[0].isRowVector() && arrays[0].rank() == 2) {
            return Nd4j.vstack(arrays);
        }


        long[] shape = arrays[0].shape();
        long[] newShape = ArrayUtils.add(shape, 0, 1);

        boolean shouldReshape = true;
        if (arrays[0].size(0) == 1)
            shouldReshape = false;

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
    public static INDArray[] tear(INDArray tensor, int... dimensions) {
        if (dimensions.length >= tensor.rank())
            throw new ND4JIllegalStateException("Target dimensions number should be less tensor rank");

        for (int e = 0; e < dimensions.length; e++)
            if (dimensions[e] < 0)
                throw new ND4JIllegalStateException("Target dimensions can't have negative values");

        return factory().tear(tensor, dimensions);
    }


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

        Nd4j.getExecutioner().exec(op);

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

        Nd4j.getExecutioner().exec(op);

        return ret;
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
     * Create from a given numpy file.
     *
     * @param file the file to create the ndarray from
     * @return the created ndarray
     */
    public static INDArray createFromNpyFile(File file) {
        return INSTANCE.createFromNpyFile(file);
    }

    /**
     * Create a numpy array based on the passed in
     * input stream
     * @param is the input stream to read
     * @return the loaded ndarray
     * @throws IOException
     */
    public static INDArray createNpyFromInputStream(InputStream is) throws IOException {
        byte[] content = IOUtils.toByteArray(is);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(content.length);
        byteBuffer.put(content);
        byteBuffer.rewind();
        Pointer pointer = new Pointer(byteBuffer);
        return createFromNpyPointer(pointer);
    }





    /**
     * Create an {@link INDArray}
     * from a flatbuffers {@link FlatArray}
     * @param array the array to create the {@link INDArray} from
     * @return the created {@link INDArray}
     */
    public static INDArray createFromFlatArray(FlatArray array) {
        val dtype = array.dtype();
        val order = array.byteOrder();
        val rank = (int) array.shape(0);
        val shape = new long[rank * 2 + 4];
        for (int e = 0; e < shape.length; e++)
            shape[e] = array.shape(e);

        char ordering = shape[shape.length - 1] == 99 ? 'c' : 'f';

        val shapeOf = Shape.shapeOf(shape);
        val stridesOf = Shape.stridesOf(shape);

        val _dtype = SameDiff.getDataTypeFromByte(dtype);
        val _order = SameDiff.getOrderFromByte(order);
        val prod = rank > 0 ? ArrayUtil.prod(shapeOf) : 1;
        val doubles = new double[prod];

        val bb = array.bufferAsByteBuffer();
        switch (_dtype) {
            case DOUBLE: {
                val db = bb.order(_order).asDoubleBuffer();
                for (int e = 0; e < prod; e++)
                    doubles[e] = db.get(e);
            }
            break;
            case FLOAT: {
                val fb = bb.order(_order).asFloatBuffer();
                for (int e = 0; e < prod; e++)
                    doubles[e] = (double) fb.get(e);
            }
            break;
            case HALF: {
                val sb = bb.order(_order).asShortBuffer();
                for (int e = 0; e < prod; e++)
                    doubles[e] = (double) HalfIndexer.toFloat((int) sb.get(e));
            }
            break;
            default:
                throw new UnsupportedOperationException("Unknown datatype: [" + _dtype + "]");
        }

        return Nd4j.create(doubles, shapeOf, stridesOf, 0, ordering);
    }
}
