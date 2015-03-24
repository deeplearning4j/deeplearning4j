/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.factory;

import com.google.common.base.Function;
import com.google.common.primitives.Ints;
import org.nd4j.linalg.api.buffer.BufferReaper;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.api.buffer.factory.DefaultDataBufferFactory;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.instrumentation.InMemoryInstrumentation;
import org.nd4j.linalg.api.instrumentation.Instrumentation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.factory.DefaultOpFactory;
import org.nd4j.linalg.api.ops.factory.OpFactory;
import org.nd4j.linalg.api.resources.ResourceManager;
import org.nd4j.linalg.api.resources.DefaultResourceManager;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.factory.DefaultDistributionFactory;
import org.nd4j.linalg.api.rng.distribution.factory.DistributionFactory;
import org.nd4j.linalg.convolution.ConvolutionInstance;
import org.nd4j.linalg.convolution.DefaultConvolutionInstance;
import org.nd4j.linalg.fft.DefaultFFTInstance;
import org.nd4j.linalg.fft.FFTInstance;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.Shape;
import org.springframework.core.io.ClassPathResource;

import java.io.*;
import java.lang.ref.ReferenceQueue;
import java.lang.reflect.Constructor;
import java.util.*;

/**
 * Creation of ndarrays via classpath discovery.
 *
 * @author Adam Gibson
 */
public class Nd4j {

    public final static String LINALG_PROPS = "/nd4j.properties";
    public final static String NUMERICAL_STABILITY = "force.stability";
    public final static String FFT_OPS = "fft";
    public final static String DATA_BUFFER_OPS = "databufferfactory";
    public final static String CONVOLUTION_OPS = "convops";
    public final static String DTYPE = "dtype";
    public final static String BLAS_OPS = "blas.ops";
    public final static String ORDER_KEY = "ndarray.order";
    public final static String NDARRAY_FACTORY_CLASS = "ndarrayfactory.class";
    public final static String COPY_OPS = "ndarray.copyops";
    public final static String OP_EXECUTIONER = "opexec";
    public final static String OP_FACTORY = "opfactory";
    public final static String RANDOM = "rng";
    public final static String DISTRIBUTION = "dist";
    public final static String INSTRUMENTATION = "instrumentation";
    public final static String RESOURCE_MANAGER = "resourcemanager";


    public static int dtype = DataBuffer.FLOAT;
    public static char ORDER = 'c';
    public static IComplexNumber UNIT;
    public static IComplexNumber ZERO;
    public static IComplexNumber NEG_UNIT;
    public static double EPS_THRESHOLD = 1e-12f;
    //number of elements to print in begin and end
    public static int MAX_ELEMENTS_PER_SLICE = 3;
    public static int MAX_SLICES_TO_PRINT = 3;
    public static boolean ENFORCE_NUMERICAL_STABILITY = false;
    public static boolean copyOnOps = true;
    public static boolean shouldInstrument = false;

    protected static Class<? extends BlasWrapper> blasWrapperClazz;
    protected static Class<? extends NDArrayFactory> ndArrayFactoryClazz;
    protected static Class<? extends FFTInstance> fftInstanceClazz;
    protected static Class<? extends ConvolutionInstance> convolutionInstanceClazz;
    protected static Class<? extends DataBufferFactory> dataBufferFactoryClazz;
    protected static Class<? extends OpExecutioner> opExecutionerClazz;
    protected static Class<? extends OpFactory> opFactoryClazz;
    protected static Class<? extends org.nd4j.linalg.api.rng.Random> randomClazz;
    protected static Class<? extends DistributionFactory> distributionFactoryClazz;
    protected static Class<? extends Instrumentation> instrumentationClazz;
    protected static Class<? extends ResourceManager> resourceManagerClazz;

    protected static DataBufferFactory DATA_BUFFER_FACTORY_INSTANCE;
    protected static BlasWrapper BLAS_WRAPPER_INSTANCE;
    protected static NDArrayFactory INSTANCE;
    protected static FFTInstance FFT_INSTANCE;
    protected static ConvolutionInstance CONVOLUTION_INSTANCE;
    protected static OpExecutioner OP_EXECUTIONER_INSTANCE;
    protected static DistributionFactory DISTRIBUTION_FACTORY;
    protected static OpFactory OP_FACTORY_INSTANCE;
    protected static org.nd4j.linalg.api.rng.Random random;
    protected static Instrumentation instrumentation;
    protected static ResourceManager resourceManager;

    protected static Properties props = new Properties();
    protected static ReferenceQueue<INDArray> referenceQueue = new ReferenceQueue<>();
    protected static ReferenceQueue<DataBuffer> bufferQueue = new ReferenceQueue<>();

    static {
        Nd4j nd4j = new Nd4j();
        nd4j.initContext();
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

    /**
     * Get the current random generator
     *
     * @return the current random generator
     */
    public static org.nd4j.linalg.api.rng.Random getRandom() {
        return random;
    }

    /**
     * Get the resource manager
     * @return the resource manager
     *
     */
    public static ResourceManager getResourceManager() {
        return resourceManager;
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
     * Returns the fft instance
     *
     * @return the fft instance
     */
    public static FFTInstance getFFt() {
        return FFT_INSTANCE;
    }

    /**
     * @param fftInstance
     */
    public static void setFft(FFTInstance fftInstance) {
        if (fftInstance == null)
            throw new IllegalArgumentException("No null instances allowed");
        FFT_INSTANCE = fftInstance;
    }

    /**
     * Given a sequence of Iterators over a applyTransformToDestination of matrices, fill in all of
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
     * Given a sequence of Iterators over a applyTransformToDestination of matrices, fill in all of
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
        Nd4j.getResourceManager().register(log);
    }

    /**
     * The factory used for creating ndarrays
     *
     * @return the factory instance used for creating ndarrays
     */
    public static NDArrayFactory factory() {
        return INSTANCE;
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
     * Create a buffer equal of length prod(shape)
     *
     * @param shape the shape of the buffer to create
     * @param type  the type to create
     * @return the created buffer
     */
    public static DataBuffer createBuffer(int[] shape, int type) {
        int length = ArrayUtil.prod(shape);
        return type == DataBuffer.DOUBLE ? createBuffer(new double[length]) : createBuffer(new float[length]);
    }

    /**
     * Create a buffer equal of length prod(shape)
     *
     * @param shape the shape of the buffer to create
     * @return the created buffer
     */
    public static DataBuffer createBuffer(int[] shape) {
        int length = ArrayUtil.prod(shape);
        return createBuffer(length);
    }

    /**
     * Creates a buffer of the specified length based on the data type
     *
     * @param length the length of te buffer
     * @return the buffer to create
     */
    public static DataBuffer createBuffer(long length) {
        if (dataType() == DataBuffer.FLOAT) {
            return createBuffer(new float[(int) length]);

        }
        return createBuffer(new double[(int) length]);
    }

    /**
     * Create a buffer based on the data type
     *
     * @param data the data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBuffer(float[] data) {
        DataBuffer ret;
        if (dataType() == DataBuffer.FLOAT)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createFloat(data);
        else
            ret = DATA_BUFFER_FACTORY_INSTANCE.createDouble(ArrayUtil.toDoubles(data));
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a buffer based on the data type
     *
     * @param data the data to create the buffer with
     * @return the created buffer
     */
    public static DataBuffer createBuffer(double[] data) {
        DataBuffer ret;
        if (dataType() == DataBuffer.DOUBLE)
            ret = DATA_BUFFER_FACTORY_INSTANCE.createDouble(data);
        else
            ret = DATA_BUFFER_FACTORY_INSTANCE.createFloat(ArrayUtil.toFloats(data));
        logCreationIfNecessary(ret);
        return ret;
    }


    public static void setFactory(NDArrayFactory factory) {
        INSTANCE = factory;
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
     * Returns the data type used for the runtime
     *
     * @return the datatype used for the runtime
     */
    public static int dataType() {
        return dtype;
    }

    public static BlasWrapper getBlasWrapper() {
        return BLAS_WRAPPER_INSTANCE;
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
     * Create a complex ndarray based on the
     * real and imaginary
     *
     * @param real the real numbers
     * @param imag the imaginary components
     * @return the complex
     */
    public static IComplexNDArray createComplex(INDArray real, INDArray imag) {
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
        List<IComplexNDArray> list = new ArrayList<>();
        for (int i = 0; i < num; i++)
            list.add(n.dup());
        IComplexNDArray ret = Nd4j.createComplex(list, Ints.concat(new int[]{num}, n.shape()));
        logCreationIfNecessary(ret);
        return ret;
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

            final IComplexNumber[] data = new IComplexNumber[vec.length()];
            final Double[] index = new Double[vec.length()];

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

                        return Double.compare(
                                data[idx1].absoluteValue().doubleValue(),
                                data[idx2].absoluteValue().doubleValue());
                    }
                });

            else
                Arrays.sort(index, new Comparator<Double>() {
                    @Override
                    public int compare(Double o1, Double o2) {
                        int idx1 = (int) o1.doubleValue();
                        int idx2 = (int) o2.doubleValue();

                        return -Double.compare(
                                data[idx1].absoluteValue().doubleValue(),
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
     * Sort an ndarray along a particular dimension
     *
     * @param ndarray   the ndarray to sort
     * @param dimension the dimension to sort
     * @return the indices and the sorted ndarray
     */
    public static INDArray[] sortWithIndices(INDArray ndarray, int dimension, boolean ascending) {
        INDArray indices = Nd4j.create(ndarray.shape());
        INDArray[] ret = new INDArray[2];

        for (int i = 0; i < ndarray.vectorsAlongDimension(dimension); i++) {
            INDArray vec = ndarray.vectorAlongDimension(i, dimension);
            INDArray indexVector = indices.vectorAlongDimension(i, dimension);
            final Double[] data = new Double[vec.length()];
            final Double[] index = new Double[vec.length()];

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
                        return Double.compare(
                                o1.asDouble().absoluteValue().doubleValue(),
                                o2.asDouble().absoluteValue().doubleValue());
                    }
                });

            else
                Arrays.sort(data, new Comparator<IComplexNumber>() {
                    @Override
                    public int compare(IComplexNumber o1, IComplexNumber o2) {
                        return -Double.compare(
                                o1.asDouble().absoluteValue().doubleValue(),
                                o2.asDouble().absoluteValue().doubleValue());
                    }
                });

            for (int j = 0; j < vec.length(); j++)
                vec.putScalar(j, data[j]);


        }

        return ndarray;
    }

    /**
     * Sort an ndarray along a particular dimension
     *
     * @param ndarray   the ndarray to sort
     * @param dimension the dimension to sort
     * @return the sorted ndarray
     */
    public static INDArray sort(INDArray ndarray, int dimension, boolean ascending) {
        for (int i = 0; i < ndarray.vectorsAlongDimension(dimension); i++) {
            INDArray vec = ndarray.vectorAlongDimension(i, dimension);
            double[] data = new double[vec.length()];
            for (int j = 0; j < vec.length(); j++) {
                data[j] = vec.getDouble(j);
            }

            Arrays.sort(data);

            if (ascending)
                for (int j = 0; j < vec.length(); j++)
                    vec.putScalar(j, data[j]);
            else {
                int count = data.length - 1;
                for (int j = 0; j < vec.length(); j++) {
                    vec.putScalar(j, data[count--]);
                }
            }

        }

        return ndarray;
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
        int[] shape = n.isColumnVector() ? new int[]{n.shape()[0]} : n.shape();
        int[] retShape = Ints.concat(new int[]{num}, shape);
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
    public static INDArray linspace(int lower, int upper, int num) {
        return INSTANCE.linspace(lower, upper, num);
    }

    public static INDArray toFlattened(Collection<INDArray> matrices) {
        INDArray ret = INSTANCE.toFlattened(matrices);
        logCreationIfNecessary(ret);
        return ret;

    }

    public static IComplexNDArray complexFlatten(List<IComplexNDArray> flatten) {
        IComplexNDArray ret = INSTANCE.complexFlatten(flatten);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray complexFlatten(IComplexNDArray... flatten) {
        IComplexNDArray ret = INSTANCE.complexFlatten(flatten);
        logCreationIfNecessary(ret);
        return ret;
    }

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

    public static INDArray toFlattened(INDArray... matrices) {
        return INSTANCE.toFlattened(matrices);
    }

    /**
     * Create the identity ndarray
     *
     * @param n the number for the identity
     * @return
     */
    public static INDArray eye(int n) {
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
     * Read line via input streams
     *
     * @param filePath the input stream ndarray
     * @param split    the split separator
     * @return the read txt method
     */
    public static void writeTxt(INDArray write, String filePath, String split) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filePath));
        for (int i = 0; i < write.rows(); i++) {
            StringBuffer sb = new StringBuffer();
            INDArray row = write.getRow(i);
            for (int j = 0; j < row.columns(); j++) {
                sb.append(row.getDouble(j));
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
    public static INDArray readTxt(InputStream filePath, String split) throws IOException {
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
            data2.add(read(data));


        }

        ret = Nd4j.create(data2.size(), numColumns);
        for (int i = 0; i < data2.size(); i++)
            ret.putRow(i, Nd4j.create(Nd4j.createBuffer(data2.get(i))));
        return ret;
    }

    private static float[] read(String[] split) {
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
    public static INDArray readTxt(String filePath, String split) throws IOException {
        return readTxt(new FileInputStream(filePath), split);
    }

    /**
     * Read line via input streams
     *
     * @param filePath the input stream ndarray
     * @return the read txt method
     */
    public static INDArray readTxt(String filePath) throws IOException {
        return readTxt(filePath, "\t");
    }

    private static INDArray loadRow(String[] data) {
        INDArray ret = Nd4j.create(data.length);
        for (int i = 0; i < data.length; i++) {
            ret.putScalar(i, Double.parseDouble(data[i]));
        }

        return ret;
    }

    /**
     * Read in an ndarray from a data input stream
     *
     * @param dis the data input stream to read from
     * @return the ndarray
     * @throws IOException
     */
    public static INDArray read(DataInputStream dis) throws IOException {
        int dimensions = dis.readInt();
        int[] shape = new int[dimensions];
        int[] stride = new int[dimensions];

        for (int i = 0; i < dimensions; i++)
            shape[i] = dis.readInt();
        for (int i = 0; i < dimensions; i++)
            stride[i] = dis.readInt();
        String dataType = dis.readUTF();
        String type = dis.readUTF();

        if (!type.equals("real"))
            throw new IllegalArgumentException("Trying to read in a complex ndarray");

        if (dataType.equals("double")) {
            double[] data = ArrayUtil.readDouble(ArrayUtil.prod(shape), dis);
            return create(data, shape, stride, 0);
        }

        else {
            float[] data = ArrayUtil.readFloat(ArrayUtil.prod(shape), dis);
            return create(data, shape, stride, 0);
        }

    }

    /**
     * Write an ndarray to the specified outputstream
     *
     * @param arr              the array to write
     * @param dataOutputStream the data output stream to write to
     * @throws IOException
     */
    public static void write(INDArray arr, DataOutputStream dataOutputStream) throws IOException {
        dataOutputStream.writeInt(arr.shape().length);
        for (int i = 0; i < arr.shape().length; i++)
            dataOutputStream.writeInt(arr.size(i));
        for (int i = 0; i < arr.stride().length; i++)
            dataOutputStream.writeInt(arr.stride()[i]);

        dataOutputStream.writeUTF(dataType() == DataBuffer.FLOAT ? "float" : "double");

        dataOutputStream.writeUTF("real");

        if (dataType() == DataBuffer.DOUBLE)
            ArrayUtil.write(arr.data().asDouble(), dataOutputStream);
        else
            ArrayUtil.write(arr.data().asFloat(), dataOutputStream);

    }

    /**
     * Clear nans from an ndarray
     *
     * @param arr the array to clear
     */
    public static void clearNans(INDArray arr) {
        BooleanIndexing.applyWhere(arr, Conditions.isNan(), new Value(Nd4j.EPS_THRESHOLD));
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
        dataOutputStream.writeInt(arr.shape().length);
        for (int i = 0; i < arr.shape().length; i++)
            dataOutputStream.writeInt(arr.size(i));
        for (int i = 0; i < arr.stride().length; i++)
            dataOutputStream.writeInt(arr.stride()[i]);
        dataOutputStream.writeUTF(dataType() == DataBuffer.FLOAT ? "float" : "double");

        dataOutputStream.writeUTF("complex");

        if (dataType() == DataBuffer.DOUBLE)
            ArrayUtil.write(arr.data().asDouble(), dataOutputStream);
        else
            ArrayUtil.write(arr.data().asFloat(), dataOutputStream);

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
        INDArray ret = INSTANCE.reverse(reverse);
        logCreationIfNecessary(ret);
        return ret;
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
     * Create double
     *
     * @param real real component
     * @param imag imag component
     * @return
     */
    public static IComplexNumber createComplexNumber(Number real, Number imag) {
        if (dataType() == DataBuffer.FLOAT)
            return INSTANCE.createFloat(real.floatValue(), imag.floatValue());
        return INSTANCE.createDouble(real.doubleValue(), imag.doubleValue());

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
     * Generates a random matrix between min and max
     *
     * @param shape the number of rows of the matrix
     * @param min   the minimum number
     * @param max   the maximum number
     * @param rng   the rng to use
     * @return a drandom matrix of the specified shape and range
     */
    public static INDArray rand(int[] shape, double min, double max, org.nd4j.linalg.api.rng.Random rng) {
        INDArray ret = INSTANCE.rand(shape, min, max, rng);
        logCreationIfNecessary(ret);
        return ret;
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
        INDArray ret = INSTANCE.rand(rows, columns, min, max, rng);
        logCreationIfNecessary(ret);
        return ret;
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
            IComplexNDArray ret = Nd4j.createComplex(new int[]{vectorLength, 1});
            for (int i = 0; i < vectorLength; i++) {
                ret.putScalar(i, x.getComplex(i, i));
            }

            return ret;
        }


        throw new IllegalArgumentException("Illegal input for diagonal of shape " + x.shape().length);

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
        if (x.isScalar())
            return x.dup();

        if (x.isVector()) {
            INDArray m = Nd4j.create(x.length(), x.length());
            INDArray xLinear = x.linearView();

            for (int i = 0; i < x.length(); i++)
                m.put(i, i, xLinear.getDouble(i));

            return m;

        } else if (x.isMatrix()) {
            int vectorLength = x.rows() - k;
            INDArray ret = Nd4j.create(new int[]{vectorLength, 1});
            for (int i = 0; i < vectorLength; i++) {
                ret.putScalar(i, x.getDouble(i, i));
            }

            return ret;
        }


        throw new IllegalArgumentException("Illegal input for diagonal of shape " + x.shape().length);
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
     * Create a complex ndarray from the passed in indarray
     *
     * @param data the data to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(IComplexNumber[] data, int[] shape) {
        IComplexNDArray ret = INSTANCE.createComplex(data, shape);
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
    public static IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int offset, char ordering) {
        IComplexNDArray ret = INSTANCE.createComplex(data, shape, offset, ordering);
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
        IComplexNDArray ret = INSTANCE.createComplex(arrs, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r       the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows, int columns, org.nd4j.linalg.api.rng.Random r) {
        INDArray ret = INSTANCE.rand(rows, columns, r);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param seed    the  seed to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows, int columns, long seed) {
        INDArray ret = INSTANCE.rand(rows, columns, seed);
        logCreationIfNecessary(ret);
        return ret;
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
        INDArray ret = INSTANCE.rand(rows, columns);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Random normal using the given rng
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r       the random generator to use
     * @return
     */
    public static INDArray randn(int rows, int columns, org.nd4j.linalg.api.rng.Random r) {
        INDArray ret = INSTANCE.randn(rows, columns, r);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Random normal using the current time stamp
     * as the seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    public static INDArray randn(int rows, int columns) {
        INDArray ret = INSTANCE.randn(rows, columns);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Random normal using the specified seed
     *
     * @param rows    the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    public static INDArray randn(int rows, int columns, long seed) {
        INDArray ret = INSTANCE.randn(rows, columns, seed);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape, Distribution r) {
        INDArray ret = INSTANCE.rand(shape, r);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape, org.nd4j.linalg.api.rng.Random r) {
        INDArray ret = INSTANCE.rand(shape, r);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     *
     * @param shape the shape of the ndarray
     * @param seed  the  seed to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape, long seed) {
        INDArray ret = INSTANCE.rand(shape, seed);
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
        INDArray ret = INSTANCE.rand(shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Random normal using the given rng
     *
     * @param shape the shape of the ndarray
     * @param r     the random generator to use
     * @return
     */
    public static INDArray randn(int[] shape, org.nd4j.linalg.api.rng.Random r) {
        INDArray ret = INSTANCE.randn(shape, r);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Random normal using the current time stamp
     * as the seed
     *
     * @param shape the shape of the ndarray
     * @return
     */
    public static INDArray randn(int[] shape) {
        INDArray ret = INSTANCE.randn(shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Random normal using the specified seed
     *
     * @param shape the shape of the ndarray
     * @return
     */
    public static INDArray randn(int[] shape, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return randn(shape, Nd4j.getRandom());
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
        IComplexNDArray ret = INSTANCE.createComplex(data);
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
        INDArray ret = INSTANCE.create(columns);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray
     *
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray createComplex(int columns, char order) {
        IComplexNDArray ret = INSTANCE.createComplex(columns);
        logCreationIfNecessary(ret);
        return ret;
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
     * @param data the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(double[] data) {
        return create(data, order());
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

    private static IComplexNDArray createComplex(float[] data, Character order) {
        IComplexNDArray ret = INSTANCE.createComplex(data, order);
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
     * Returns true if the given ndarray has either
     * an infinite or nan
     *
     * @param num the ndarray to test
     * @return true if the given ndarray has either infinite or nan
     * false otherwise
     */
    public static boolean hasInvalidNumber(INDArray num) {
        INDArray linear = num.linearView();
        for (int i = 0; i < linear.length(); i++) {
            if (Double.isInfinite(linear.getDouble(i)) || Double.isNaN(linear.getDouble(i)))
                return true;
        }
        return false;
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the created ndarray
     */
    public static INDArray zeros(int rows, int columns) {
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
    public static INDArray complexZeros(int columns) {
        return INSTANCE.complexZeros(columns);
    }

    public static IComplexNDArray complexValueOf(int num, IComplexNumber value) {
        IComplexNDArray ret = INSTANCE.complexValueOf(num, value);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray complexValueOf(int[] shape, IComplexNumber value) {
        IComplexNDArray ret = INSTANCE.complexValueOf(shape, value);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray complexValueOf(int num, double value) {
        return INSTANCE.complexValueOf(num, value);
    }

    public static IComplexNDArray complexValueOf(int[] shape, double value) {
        IComplexNDArray ret = INSTANCE.complexValueOf(shape, value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray
     *
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return the created ndarray
     */
    public static INDArray valueArrayOf(int[] shape, double value) {
        INDArray ret = INSTANCE.valueArrayOf(shape, value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray
     *
     * @param num   number of columns
     * @param value the value to assign
     * @return the created ndarray
     */
    public static INDArray valueArrayOf(int num, double value) {
        INDArray ret = INSTANCE.valueArrayOf(new int[]{1, num}, value);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a row vector with the specified number of columns
     *
     * @param rows    the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param value   the value to assign
     * @return the created ndarray
     */
    public static INDArray valueArrayOf(int rows, int columns, double value) {
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
    public static INDArray complexOnes(int rows, int columns) {
        INDArray ret = INSTANCE.complexOnes(rows, columns);
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
     * Concatneate ndarrays along a dimension
     *
     * @param dimension the dimension to concatneate along
     * @param toConcat  the ndarrays to concat
     * @return the concatted ndarrays with an output shape of
     * the ndarray shapes save the dimension shape specified
     * which is then the sum of the sizes along that dimension
     */
    public static INDArray concat(int dimension, INDArray... toConcat) {
        INDArray ret = INSTANCE.concat(dimension, toConcat);
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
    public static INDArray zeros(int[] shape) {
        INDArray ret = INSTANCE.zeros(shape);
        logCreationIfNecessary(ret);
        return ret;

    }

    /**
     * Create an ndarray of ones
     *
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static IComplexNDArray complexZeros(int[] shape) {
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
    public static INDArray ones(int[] shape) {
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
    public static IComplexNDArray complexOnes(int[] shape) {
        IComplexNDArray ret = INSTANCE.complexOnes(shape);
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
    public static IComplexNDArray createComplex(float[] data, int rows, int columns, int[] stride, int offset) {
        IComplexNDArray ret = INSTANCE.createComplex(data, rows, columns, stride, offset);
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
    public static INDArray create(float[] data, int rows, int columns, int[] stride, int offset) {
        INDArray ret = INSTANCE.create(data, rows, columns, stride, offset);
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
    public static IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset) {
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
    public static INDArray create(double[] data, int[] shape, int[] stride, int offset) {
        INDArray ret = INSTANCE.create(data, shape, stride, offset);
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
    public static INDArray create(float[] data, int[] shape) {
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
    public static IComplexNDArray createComplex(double[] data, int[] shape) {
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
        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride);
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
    public static IComplexNDArray createComplex(double[] data, int rows, int columns, int[] stride, int offset) {
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
    public static INDArray create(double[] data, int rows, int columns, int[] stride, int offset) {
        INDArray ret = INSTANCE.create(data, rows, columns, stride, offset);
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
    public static IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset) {
        IComplexNDArray ret = INSTANCE.createComplex(data, shape, stride, offset);
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
    public static INDArray create(float[] data, int[] shape, int offset) {
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
    public static INDArray create(double[] data, int[] shape, int offset, char ordering) {
        INDArray ret = INSTANCE.create(data, shape, Nd4j.getStrides(shape), offset);
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
    public static INDArray create(float[] data, int[] shape, int[] stride, int offset) {
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
    public static IComplexNDArray createComplex(int rows, int columns, int[] stride, int offset) {
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
    public static INDArray create(int rows, int columns, int[] stride, int offset) {
        INDArray ret = INSTANCE.create(rows, columns, stride, offset);
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
    public static IComplexNDArray createComplex(int[] shape, int[] stride, int offset) {
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
    public static INDArray create(int[] shape, int[] stride, int offset) {
        INDArray ret = INSTANCE.create(shape, stride, offset);
        logCreationIfNecessary(ret);
        return ret;

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
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    public static INDArray scalar(Number value, int offset) {
        return INSTANCE.scalar(value, offset);
    }

    /**
     * Create a scalar ndarray with the specified offset
     *
     * @param value  the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray complexScalar(Number value, int offset) {
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
    public static INDArray scalar(double value, int offset) {
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
    public static INDArray scalar(float value, int offset) {
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
    public static IComplexNDArray scalar(IComplexNumber value, int offset) {
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
    public static IComplexNDArray scalar(IComplexFloat value, int offset) {
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
    public static IComplexNDArray scalar(IComplexDouble value, int offset) {
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
        if (Shape.isRowVectorShape(shape) && shape.length > 1)
            shape = new int[]{shape[1]};
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
     * An alias for repmat
     *
     * @param tile   the ndarray to tile
     * @param repeat the shape to repeat
     * @return the tiled ndarray
     */
    public static INDArray tile(INDArray tile, int[] repeat) {
        return tile.repmat(repeat);
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
    public static IComplexNDArray createComplex(float[] data, int rows, int columns, int[] stride, int offset, char ordering) {
        IComplexNDArray ret = INSTANCE.createComplex(data, rows, columns, stride, offset);
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
    public static INDArray create(float[] data, int rows, int columns, int[] stride, int offset, char ordering) {
        INDArray ret = INSTANCE.create(data, rows, columns, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(int[] shape, int dataType) {
        INDArray ret = INSTANCE.create(shape, dataType);
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
    public static IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset, char ordering) {
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
    public static INDArray create(double[] data, int[] shape, int[] stride, int offset, char ordering) {
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
    public static IComplexNDArray createComplex(float[] data, int[] shape, char ordering) {
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
    public static IComplexNDArray createComplex(double[] data, int rows, int columns, int[] stride, int offset, char ordering) {
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
    public static INDArray create(double[] data, int rows, int columns, int[] stride, int offset, char ordering) {
        INDArray ret = INSTANCE.create(data, rows, columns, stride, offset);
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
    public static IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset, char ordering) {
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
    public static INDArray create(float[] data, int[] shape, int[] stride, int offset, char ordering) {
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
    public static IComplexNDArray createComplex(int rows, int columns, int[] stride, int offset, char ordering) {
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
    public static INDArray create(int rows, int columns, int[] stride, int offset, char ordering) {
        INDArray ret = INSTANCE.create(rows, columns, stride, offset);
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
    public static IComplexNDArray createComplex(int[] shape, int[] stride, int offset, char ordering) {
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
    public static INDArray create(int[] shape, int[] stride, int offset, char ordering) {
        INDArray ret = INSTANCE.create(shape, stride, offset);
        logCreationIfNecessary(ret);
        return ret;

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
        IComplexNDArray ret = INSTANCE.createComplex(rows, columns, stride);
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
        INDArray ret = INSTANCE.create(rows, columns, stride);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape, int[] stride, char ordering) {
        return createComplex(shape, stride, 0);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape, int[] stride, char ordering) {
        INDArray ret = INSTANCE.create(shape, stride);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param rows    the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows, int columns, char ordering) {
        IComplexNDArray ret = INSTANCE.createComplex(rows, columns);
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
    public static INDArray create(int rows, int columns, char ordering) {
        INDArray ret = INSTANCE.create(rows, columns, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    /**
     * Creates a complex ndarray with the specified shape
     *
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape, char ordering) {
        IComplexNDArray ret = INSTANCE.createComplex(createBuffer(ArrayUtil.prod(shape) * 2), shape, 0, ordering);
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
        INDArray ret = INSTANCE.create(shape, ordering);
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
    public static IComplexNDArray createComplex(float[] data, int[] shape, int offset, char ordering) {
        IComplexNDArray ret = INSTANCE.createComplex(data, shape, ArrayUtil.calcStrides(shape, 2), offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray createComplex(double[] data, int[] shape, int offset) {
        return createComplex(data, shape, offset, Nd4j.order());
    }

    public static INDArray create(double[] data, int[] shape, int offset) {
        INDArray ret = INSTANCE.create(data, shape, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray createComplex(double[] data, int[] ints, int offset, char ordering) {
        IComplexNDArray ret = INSTANCE.createComplex(data, ints, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray createComplex(double[] dim) {
        IComplexNDArray ret = INSTANCE.createComplex(dim, new int[]{1, dim.length / 2});
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray createComplex(float[] data, int[] shape, int offset) {
        IComplexNDArray ret = INSTANCE.createComplex(data, shape, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(float[][] doubles) {
        return INSTANCE.create(doubles);
    }

    public static IComplexNDArray complexLinSpace(int i, int i1, int i2) {
        return Nd4j.createComplex(Nd4j.linspace(i, i1, i2));

    }

    public static INDArray create(float[] data, int[] shape, int[] stride, char ordering, int offset) {
        INDArray ret = INSTANCE.create(data, shape, stride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(float[] data, int[] shape, char ordering, int offset) {
        INDArray ret = INSTANCE.create(data, shape, getStrides(shape, ordering), offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(DataBuffer data, int[] shape, int[] strides, int offset) {
        INDArray ret = INSTANCE.create(data, shape, strides, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(DataBuffer data, int[] shape, int offset) {
        INDArray ret = INSTANCE.create(data, shape, getStrides(shape), offset);
        logCreationIfNecessary(ret);
        return ret;

    }

    public static INDArray create(DataBuffer data, int[] newShape, int[] newStride, int offset, char ordering) {
        INDArray ret = INSTANCE.create(data, newShape, newStride, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray createComplex(DataBuffer data, int[] newShape, int[] newStrides, int offset) {
        IComplexNDArray ret = INSTANCE.createComplex(data, newShape, newStrides, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray createComplex(DataBuffer data, int[] shape, int offset) {
        IComplexNDArray ret = INSTANCE.createComplex(data, shape, offset);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray createComplex(DataBuffer data, int[] newDims, int[] newStrides, int offset, char ordering) {
        IComplexNDArray ret = INSTANCE.createComplex(data, newDims, newStrides, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray createComplex(DataBuffer data, int[] shape, int offset, char ordering) {
        IComplexNDArray ret = INSTANCE.createComplex(data, shape, offset, ordering);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(DataBuffer data, int[] shape) {
        INDArray ret = INSTANCE.create(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static IComplexNDArray createComplex(DataBuffer data, int[] shape) {
        IComplexNDArray ret = INSTANCE.createComplex(data, shape);
        logCreationIfNecessary(ret);
        return ret;
    }

    public static INDArray create(DataBuffer buffer) {
        INDArray ret = INSTANCE.create(buffer);
        logCreationIfNecessary(ret);
        return ret;
    }


    /**
     * Initializes nd4j
     */
    public void initContext() {
        try {
            ClassPathResource c = new ClassPathResource(LINALG_PROPS);
            props = new Properties();
            props.load(c.getInputStream());
            for (String key : props.stringPropertyNames())
                System.setProperty(key, props.getProperty(key));
            String otherDtype = System.getProperty(DTYPE, props.get(DTYPE).toString());
            dtype = otherDtype.equals("float") ? DataBuffer.FLOAT : DataBuffer.DOUBLE;
            copyOnOps = Boolean.parseBoolean(props.getProperty(COPY_OPS, "true"));
            shouldInstrument = Boolean.parseBoolean(props.getProperty(INSTRUMENTATION, "false"));
            ORDER = System.getProperty(ORDER_KEY, props.getProperty(ORDER_KEY, "c").toString()).charAt(0);
            if (opExecutionerClazz == null)
                opExecutionerClazz = (Class<? extends OpExecutioner>) Class.forName(System.getProperty(OP_EXECUTIONER, DefaultOpExecutioner.class.getName()));
            if (fftInstanceClazz == null)
                fftInstanceClazz = (Class<? extends FFTInstance>) Class.forName(System.getProperty(FFT_OPS, DefaultFFTInstance.class.getName()));
            if (ndArrayFactoryClazz == null)
                ndArrayFactoryClazz = (Class<? extends NDArrayFactory>) Class.forName(System.getProperty(NDARRAY_FACTORY_CLASS, props.get(NDARRAY_FACTORY_CLASS).toString()));
            if (convolutionInstanceClazz == null)
                convolutionInstanceClazz = (Class<? extends ConvolutionInstance>) Class.forName(System.getProperty(CONVOLUTION_OPS, DefaultConvolutionInstance.class.getName()));
            if (dataBufferFactoryClazz == null) {
                String defaultName = props.getProperty(DATA_BUFFER_OPS, DefaultDataBufferFactory.class.getName());
                dataBufferFactoryClazz = (Class<? extends DataBufferFactory>) Class.forName(System.getProperty(DATA_BUFFER_OPS, defaultName));
            }
            if(resourceManagerClazz == null) {
                resourceManagerClazz = (Class<? extends ResourceManager>) Class.forName(props.getProperty(RESOURCE_MANAGER, DefaultResourceManager.class.getName()));
            }

            if (randomClazz == null) {
                String rand = props.getProperty(RANDOM, DefaultRandom.class.getName());
                randomClazz = (Class<? extends org.nd4j.linalg.api.rng.Random>) Class.forName(rand);
            }

            if (instrumentationClazz == null)
                instrumentationClazz = (Class<? extends Instrumentation>) Class.forName(props.getProperty(INSTRUMENTATION, InMemoryInstrumentation.class.getName()));

            if (opFactoryClazz == null)
                opFactoryClazz = (Class<? extends OpFactory>) Class.forName(System.getProperty(OP_FACTORY, DefaultOpFactory.class.getName()));

            if (blasWrapperClazz == null)
                blasWrapperClazz = (Class<? extends BlasWrapper>) Class.forName(System.getProperty(BLAS_OPS, props.get(BLAS_OPS).toString()));
            if (distributionFactoryClazz == null) {
                String clazzName = props.getProperty(DISTRIBUTION, DefaultDistributionFactory.class.getName());
                distributionFactoryClazz = (Class<? extends DistributionFactory>) Class.forName(clazzName);

            }

            resourceManager = resourceManagerClazz.newInstance();
            instrumentation = instrumentationClazz.newInstance();
            OP_EXECUTIONER_INSTANCE = opExecutionerClazz.newInstance();
            FFT_INSTANCE = fftInstanceClazz.newInstance();
            Constructor c2 = ndArrayFactoryClazz.getConstructor(int.class, char.class);
            INSTANCE = (NDArrayFactory) c2.newInstance(dtype, ORDER);
            CONVOLUTION_INSTANCE = convolutionInstanceClazz.newInstance();
            BLAS_WRAPPER_INSTANCE = blasWrapperClazz.newInstance();
            DATA_BUFFER_FACTORY_INSTANCE = dataBufferFactoryClazz.newInstance();
            OP_FACTORY_INSTANCE = opFactoryClazz.newInstance();
            random = randomClazz.newInstance();
            UNIT = Nd4j.createFloat(1, 0);
            ZERO = Nd4j.createFloat(0, 0);
            NEG_UNIT = Nd4j.createFloat(-1, 0);
            ENFORCE_NUMERICAL_STABILITY = Boolean.parseBoolean(System.getProperty(NUMERICAL_STABILITY, String.valueOf(false)));
            DISTRIBUTION_FACTORY = distributionFactoryClazz.newInstance();
            //start the buffer reaper
            new BufferReaper(refQueue(),bufferRefQueue()).start();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }


}
