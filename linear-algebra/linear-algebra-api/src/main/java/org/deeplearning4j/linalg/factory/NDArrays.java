package org.deeplearning4j.linalg.factory;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.linalg.api.complex.IComplexDouble;
import org.deeplearning4j.linalg.api.complex.IComplexFloat;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.deeplearning4j.linalg.util.ReflectionUtil;
import org.springframework.core.io.ClassPathResource;

import java.lang.reflect.Constructor;
import java.util.*;

/**
 *
 * Creation of ndarrays via classpath discovery.
 *
 *
 * @author Adam Gibson
 */
public class NDArrays {

    private static Class<? extends INDArray> clazz;
    private static Class<? extends IComplexNDArray> complexClazz;
    private static Class<? extends IComplexDouble> complexDoubleClazz;
    private static Class<? extends IComplexFloat> complexFloatClazz;
    private static Class<? extends BlasWrapper> blasWrapperClazz;
    private static BlasWrapper BLAS_WRAPPER_INSTANCE;
    public final static String LINALG_PROPS = "/dl4j-linalg.properties";
    public final static String REAL_CLASS_PROP = "real.class";
    public final static String COMPLEX_CLASS_PROP = "complex.class";
    public final static String DTYPE = "dtype";
    public final static String COMPLEX_DOUBLE = "complex.double.class";
    public final static String COMPLEX_FLOAT = "complex.float.class";
    public final static String BLAS_OPS = "blas.ops";
    public static String dtype;
    private static Properties props = new Properties();


    static {
        try {
            ClassPathResource c = new ClassPathResource(LINALG_PROPS);
            props.load(c.getInputStream());
            String realType = props.get(REAL_CLASS_PROP + "." + props.get(DTYPE)).toString();
            String complexType = props.get(COMPLEX_CLASS_PROP + "." + props.get(DTYPE)).toString();
            dtype = props.get(DTYPE).toString();
            complexDoubleClazz = (Class<? extends IComplexDouble>) Class.forName(props.get(COMPLEX_DOUBLE).toString());
            complexFloatClazz = (Class<? extends IComplexFloat>) Class.forName(props.get(COMPLEX_FLOAT).toString());
            blasWrapperClazz = (Class<? extends BlasWrapper>) Class.forName(props.get(BLAS_OPS).toString());
            clazz = (Class<? extends INDArray>) Class.forName(realType);
            complexClazz = (Class<? extends IComplexNDArray>) Class.forName(complexType);
            BLAS_WRAPPER_INSTANCE = blasWrapperClazz.newInstance();
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }


    public static BlasWrapper getBlasWrapper() {
        return BLAS_WRAPPER_INSTANCE;
    }


    /**
     * Generate a linearly spaced vector
     * @param lower upper bound
     * @param upper lower bound
     * @param num the step size
     * @return the linearly spaced vector
     */
    public static INDArray linspace(int lower,int upper,int num) {
        double[] length = new double[num];
        for (int i = 0; i < num; i++) {
            double t = (double) i / (num - 1);
            length[i]  =  lower * (1 - t) + t * upper;

        }
        return create(length);
    }



    public static INDArray toFlattened(Collection<INDArray> matrices) {
        int length = 0;
        for(INDArray m : matrices)  length += m.length();
        INDArray ret = NDArrays.create(length);
        int linearIndex = 0;
        for(INDArray d : matrices) {
            for(int i = 0; i < d.length(); i++) {
                ret.put(linearIndex++,d.getScalar(i));
            }
        }

        return ret;

    }


    public static INDArray toFlattened(int length,Iterator<? extends INDArray>...matrices) {

        INDArray ret = NDArrays.create(length);
        int linearIndex = 0;

        List<double[]> gradient = new ArrayList<>();
        for(Iterator<? extends INDArray> iter : matrices) {
            while(iter.hasNext()) {
                INDArray d = iter.next();
                gradient.add(d.data());
            }
        }




        INDArray ret2 = NDArrays.create(ArrayUtil.combine(gradient));
        return ret2.reshape(1,ret2.length());
    }


    /**
     * Returns a column vector where each entry is the nth bilinear
     * product of the nth slices of the two tensors.
     */
    public static INDArray bilinearProducts(INDArray curr,INDArray in) {
        if (in.columns() != 1) {
            throw new AssertionError("Expected a column vector");
        }
        if (in.rows() != curr.columns()) {
            throw new AssertionError("Number of rows in the input does not match number of columns in tensor");
        }
        if (curr.rows() != curr.columns()) {
            throw new AssertionError("Can only perform this operation on a SimpleTensor with square slices");
        }

        INDArray inT = in.transpose();
        INDArray out = NDArrays.create(curr.slices(), 1);
        for (int slice = 0; slice < curr.slices(); ++slice) {
            float result = (float) inT.mul(curr.slice(slice)).mul(in).getScalar(0).element();
            out.putScalar(slice, result);
        }

        return out;
    }


    public static INDArray toFlattened(INDArray...matrices) {
        int length = 0;
        for(INDArray m : matrices)  length += m.length();
        INDArray ret = NDArrays.create(1,length);
        int linearIndex = 0;
        for(INDArray d : matrices) {
            for(int i = 0; i < d.length(); i++) {
                ret.put(linearIndex++,d.getScalar(i));
            }
        }

        return ret;
    }


    /**
     * Create the identity ndarray
     * @param n the number for the identity
     * @return
     */
    public static INDArray eye(int n) {
        INDArray ret = NDArrays.create((int) Math.pow(n,2));

        for (int i = 0; i < n; i++) {
            ret.put(i, i, 1.0);
        }

        return ret.reshape(n,n);

    }

    /**
     * Rotate a matrix 90 degrees
     * @param toRotate the matrix to rotate
     * @return the rotated matrix
     */
    public static void rot90(INDArray toRotate) {
        if(!toRotate.isMatrix())
            throw new IllegalArgumentException("Only rotating matrices");

        INDArray start = toRotate.transpose();
        for(int i = 0; i < start.rows(); i++)
            start.putRow(i,reverse(start.getRow(i)));

    }

    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     * @param reverse the matrix to reverse
     * @return the reversed matrix
     */
    public static INDArray rot(INDArray reverse) {
        INDArray ret = NDArrays.create(reverse.shape());
        if(reverse.isVector())
            return reverse(reverse);
        else {
            for(int i = 0; i < reverse.slices(); i++) {
                ret.putSlice(i,reverse(reverse.slice(i)));
            }
        }
        return ret.reshape(reverse.shape());
    }

    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     * @param reverse the matrix to reverse
     * @return the reversed matrix
     */
    public static INDArray reverse(INDArray reverse) {
         INDArray rev = reverse.ravel();
         INDArray ret = NDArrays.create(rev.shape());
         int count = 0;
         for(int i = rev.length() - 1; i >= 0; i--) {
            ret.put(count++,rev.getScalar(i));

        }

        return ret.reshape(reverse.shape());
    }


    /**
     *  Array of evenly spaced values.
     * @param begin the begin of the range
     * @param end the end of the range
     * @return the range vector
     */
    public static INDArray arange(double begin, double end) {
        return NDArrays.create(ArrayUtil.toDoubles(ArrayUtil.range((int) begin,(int)end))).transpose();
    }


    /**
     * Create float
     * @param real real component
     * @param imag imag component
     * @return
     */
    public static IComplexFloat createFloat(float real,float imag) {
        try {
            Constructor c = complexDoubleClazz.getConstructor(double.class,double.class);
            return (IComplexFloat) c.newInstance(real,imag);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Create an instance of a complex double
     * @param real the real component
     * @param imag the imaginary component
     * @return a new imaginary double with the specified real and imaginary components
     */
    public static IComplexDouble createDouble(double real,double imag) {
        try {
            Constructor c = complexDoubleClazz.getConstructor(double.class,double.class);
            return (IComplexDouble) c.newInstance(real,imag);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Copy a to b
     * @param a the origin matrix
     * @param b the destination matrix
     */
    public static void copy(INDArray a,INDArray b) {
        a = a.reshape(new int[]{1,a.length()});
        b = b.reshape(new int[]{1,b.length()});
        for(int i = 0; i < a.length(); i++) {
            b.put(i,a.getScalar(i));
        }
    }


    /**
     * Generates a random matrix between min and max
     * @param shape the number of rows of the matrix
     * @param min the minimum number
     * @param max the maximum number
     * @param rng the rng to use
     * @return a drandom matrix of the specified shape and range
     */
    public static INDArray rand(int[] shape,float min,float max,RandomGenerator rng) {
        INDArray ret = NDArrays.create(shape).ravel();
        float r = max - min;
        for(int i = 0; i < ret.length(); i++) {
            ret.putScalar(i, r * rng.nextFloat() + min);
        }
        return ret.reshape(shape);
    }
    /**
     * Generates a random matrix between min and max
     * @param rows the number of rows of the matrix
     * @param columns the number of columns in the matrix
     * @param min the minimum number
     * @param max the maximum number
     * @param rng the rng to use
     * @return a drandom matrix of the specified shape and range
     */
    public static INDArray rand(int rows, int columns,float min,float max,RandomGenerator rng) {
        INDArray ret = NDArrays.create(rows, columns).ravel();
        float r = max - min;
        for(int i = 0; i < ret.length(); i++) {
            ret.putScalar(i, r * rng.nextFloat() + min);
        }
        return ret.reshape(rows,columns);
    }


    public static INDArray appendBias(INDArray...vectors) {
        int size = 0;
        for (INDArray vector : vectors) {
            size += vector.rows();
        }
        // one extra for the bias
        size++;

        INDArray result = NDArrays.create(size, 1);
        int index = 0;
        for (INDArray vector : vectors) {
            result.put(new NDArrayIndex[] {NDArrayIndex.interval(index, index + vector.rows()),NDArrayIndex.interval(0,result.columns())},vector);
            index += vector.rows();
        }

        result.put(new NDArrayIndex[] {NDArrayIndex.interval(index,result.rows()),NDArrayIndex.interval(0,result.columns())},NDArrays.ones(1,result.columns()));
        return result;
    }


    /**
     * Create a complex ndarray from the passed in indarray
     * @param arr the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(INDArray arr) {
        try {
            Constructor c = complexClazz.getConstructor(INDArray.class);
            return (IComplexNDArray) c.newInstance(arr);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Create a complex ndarray from the passed in indarray
     * @param data the data to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(IComplexNumber[] data,int[] shape) {
        try {
            Constructor c = complexClazz.getConstructor(ReflectionUtil.classesFor(new Object[]{data,shape}));
            return (IComplexNDArray) c.newInstance(data,shape);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Create a complex ndarray from the passed in indarray
     * @param arrs the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(List<IComplexNDArray> arrs,int[] shape) {
        try {
            Constructor c = complexClazz.getConstructor(ReflectionUtil.classesFor(new Object[]{arrs,shape}));
            return (IComplexNDArray) c.newInstance(arrs,shape);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Create a random ndarray with the given shape using the given rng
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows,int columns,RandomGenerator r) {
        return rand(new int[]{rows,columns},r);
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     * @param rows the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param seed the  seed to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows,int columns,long seed) {

        return randn(new int[]{rows,columns}, new MersenneTwister(seed));
    }
    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows,int columns) {
        return randn(new int[]{rows,columns}, System.currentTimeMillis());
    }

    /**
     * Random normal using the given rng
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r the random generator to use
     * @return
     */
    public static INDArray randn(int rows,int columns,RandomGenerator r) {
        return randn(new int[]{rows,columns},r);
    }

    /**
     * Random normal using the current time stamp
     * as the seed
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    public static INDArray randn(int rows,int columns) {
        return randn(new int[]{rows,columns}, System.currentTimeMillis());
    }

    /**
     * Random normal using the specified seed
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    public static INDArray randn(int rows,int columns,long seed) {
        return randn(new int[]{rows,columns}, new MersenneTwister(seed));
    }





    /**
     * Create a random ndarray with the given shape using the given rng
     * @param shape the shape of the ndarray
     * @param r the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape,RealDistribution r) {
        INDArray ret = create(new int[]{1,ArrayUtil.prod(shape)});
        for(int i = 0; i < ret.length(); i++)
            ret.put(i,NDArrays.scalar(r.sample()));
        return ret.reshape(shape);
    }
    /**
     * Create a random ndarray with the given shape using the given rng
     * @param shape the shape of the ndarray
     * @param r the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape,RandomGenerator r) {
        INDArray ret = create(new int[]{1,ArrayUtil.prod(shape)});
        for(int i = 0; i < ret.length(); i++)
            ret.put(i,NDArrays.scalar(r.nextDouble()));
        return ret.reshape(shape);
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     * @param shape the shape of the ndarray
     * @param seed the  seed to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape,long seed) {
        return randn(shape, new MersenneTwister(seed));
    }
    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape) {
        return randn(shape, System.currentTimeMillis());
    }

    /**
     * Random normal using the given rng
     * @param shape the shape of the ndarray
     * @param r the random generator to use
     * @return
     */
    public static INDArray randn(int[] shape,RandomGenerator r) {
        INDArray ret = create(new int[]{1,ArrayUtil.prod(shape)});
        for(int i = 0; i < ret.length(); i++)
            ret.put(i,NDArrays.scalar(r.nextGaussian()));
        return ret.reshape(shape);
    }

    /**
     * Random normal using the current time stamp
     * as the seed
     * @param shape the shape of the ndarray
     * @return
     */
    public static INDArray randn(int[] shape) {
        return randn(shape, System.currentTimeMillis());
    }

    /**
     * Random normal using the specified seed
     * @param shape the shape of the ndarray
     * @return
     */
    public static INDArray randn(int[] shape,long seed) {
        return randn(shape, new MersenneTwister(seed));
    }



    /**
     * Creates a row vector with the data
     * @param data the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray create(double[] data) {
        return create(data,new int[]{data.length});
    }

    /**
     * Creates a row vector with the data
     * @param data the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray create(float[] data) {
        return create(data,new int[]{data.length});
    }

    /**
     * Creates an ndarray with the specified data
     * @param data the number of columns in the row vector
     * @return ndarray
     */
    public static INDArray createComplex(double[] data) {
        assert data.length % 2 == 0 : "Length of data must be even. A complex ndarray is made up of pairs of real and imaginary components";
        return createComplex(data,new int[]{data.length / 2});
    }



    /**
     * Creates a row vector with the specified number of columns
     * @param columns the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray create(int columns) {
        return create(new int[]{columns});
    }

    /**
     * Creates an ndarray
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static INDArray createComplex(int columns) {
        return createComplex(new int[]{columns});
    }


    /**
     * Creates a row vector with the specified number of columns
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray zeros(int rows,int columns) {
        return zeros(new int[]{rows, columns});
    }

    /**
     * Creates a matrix of zeros
     * @param rows te number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static INDArray complexZeros(int rows,int columns) {
        return createComplex(new int[]{rows,columns});
    }


    /**
     * Creates a row vector with the specified number of columns
     * @param columns the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray zeros(int columns) {
        return zeros(new int[]{columns});
    }

    /**
     * Creates an ndarray
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static INDArray complexZeros(int columns) {
        return createComplex(new int[]{columns});
    }


    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return  the created ndarray
     */
    public static INDArray valueArrayOf(int[] shape,double value) {
        INDArray create = create(shape);
        create.assign(value);
        return create;
    }


    /**
     * Creates a row vector with the specified number of columns
     * @param rows the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param value the value to assign
     * @return  the created ndarray
     */
    public static INDArray valueArrayOf(int rows,int columns,double value) {
        INDArray create = create(rows,columns);
        create.assign(value);
        return create;
    }




    /**
     * Creates a row vector with the specified number of columns
     * @param rows the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray ones(int rows,int columns) {
        return ones(new int[]{rows,columns});
    }


    /**
     * Creates an ndarray
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static INDArray complexOnes(int rows,int columns) {
        return createComplex(new int[]{rows,columns});
    }

    /**
     * Creates a row vector with the specified number of columns
     * @param columns the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray ones(int columns) {
        return ones(new int[]{columns});
    }

    /**
     * Creates an ndarray
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static INDArray complexOnes(int columns) {
        return createComplex(new int[]{columns});
    }



    /**
     * Concatenates two matrices horizontally. Matrices must have identical
     * numbers of rows.
     */
    public static INDArray concatHorizontally(INDArray A, INDArray B) {
        if (A.rows() != B.rows()) {
            throw new IllegalArgumentException("Matrices don't have same number of rows.");
        }

        INDArray result = NDArrays.create(A.rows(), A.columns() + B.columns());
        copy(A,result);

        int count = 0;

        for(int i = A.rows(); i < B.rows(); i++) {
            result.putRow(i,B.getRow(count++));
        }
        return result;
    }

    /**
     * Concatenates two matrices vertically. Matrices must have identical
     * numbers of columns.
     */
    public static INDArray concatVertically(INDArray A, INDArray B) {
        if (A.columns() != B.columns()) {
            throw new IllegalArgumentException("Matrices don't have same number of columns (" + A.columns() + " != " + B.columns() + ".");
        }

        INDArray result = NDArrays.create(A.rows() + B.rows(), A.columns());
        copy(A,result);

        int count = 0;

        for(int i = A.columns(); i < B.columns(); i++) {
            result.putColumn(i,B.getColumn(count++));
        }
        return result;
    }







    /**
     * Create an ndarray of zeros
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static INDArray zeros(int[] shape) {
        INDArray ret = create(shape);
        return ret;

    }

    /**
     * Create an ndarray of ones
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static IComplexNDArray complexZeros(int[] shape) {
        IComplexNDArray ret = createComplex(shape);
        return ret;

    }


    /**
     * Create an ndarray of ones
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static INDArray ones(int[] shape) {
        INDArray ret = create(shape);
        ret.assign(1);
        return ret;

    }

    /**
     * Create an ndarray of ones
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static IComplexNDArray complexOnes(int[] shape) {
        IComplexNDArray ret = createComplex(shape);
        ret.assign(1);
        return ret;

    }


    /**
     * Creates a complex ndarray with the specified shape
     * @param data the data to use with the ndarray
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(float[] data,int rows,int columns,int[] stride,int offset) {
        return createComplex(data,new int[]{rows,columns},stride,offset);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param data  the data to use with the ndarray
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(float[] data,int rows,int columns,int[] stride,int offset) {
        return create(data,new int[]{rows,columns},stride,offset);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param data the data to use with the ndarray
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(float[] data,int[] shape,int[] stride,int offset) {
        try {
            Constructor c = complexClazz.getConstructor(float[].class,int[].class,int[].class,int.class);
            return (IComplexNDArray) c.newInstance(data,shape,stride,offset);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }




    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(float[] data,int[] shape,int[] stride,int offset) {
        try {
            Constructor c = clazz.getConstructor(float[].class,int[].class,int[].class,int.class);
            return (INDArray) c.newInstance(data,shape,stride,offset);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(double[] data,int[] shape) {
        return create(data,shape,ArrayUtil.calcStrides(shape),0);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(float[] data,int[] shape) {
        return create(data,shape,ArrayUtil.calcStrides(shape),0);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(double[] data,int[] shape) {
        return createComplex(data,shape,ArrayUtil.calcStrides(shape,2),0);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(float[] data,int[] shape) {
        return createComplex(data,shape,ArrayUtil.calcStrides(shape,2),0);
    }



    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(double[] data,int[] shape,int[] stride) {
        return createComplex(data,shape,stride,0);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(float[] data,int[] shape,int[] stride) {
        return createComplex(data,shape,stride,0);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(double[] data,int rows,int columns,int[] stride,int offset) {
        return createComplex(data,new int[]{rows,columns},stride,offset);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param data the data to use with tne ndarray
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data,int rows,int columns,int[] stride,int offset) {
        return create(data,new int[]{rows,columns},stride,offset);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(double[] data,int[] shape,int[] stride,int offset) {
        try {
            Constructor c = complexClazz.getConstructor(double[].class,int[].class,int[].class,int.class);
            return (IComplexNDArray) c.newInstance(data,shape,stride,offset);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data,int[] shape,int[] stride,int offset) {
        try {
            Constructor c = clazz.getConstructor(double[].class,int[].class,int[].class,int.class);
            return (INDArray) c.newInstance(data,shape,stride,offset);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
       * @return the instance
     */
    public static INDArray create(List<INDArray> list,int[] shape) {
        try {
            Constructor c = clazz.getConstructor(List.class,int[].class);
            return (INDArray) c.newInstance(list,shape);
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows,int columns,int[] stride,int offset) {
        if(dtype.equals("double"))
            return createComplex(new double[rows * columns * 2],new int[]{rows,columns},stride,offset);
        if(dtype.equals("float"))
            return createComplex(new float[rows * columns * 2],new int[]{rows,columns},stride,offset);
        throw new IllegalStateException("Illegal data type " + dtype);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(int rows,int columns,int[] stride,int offset) {
        if(dtype.equals("double"))
            return create(new double[rows * columns],new int[]{rows,columns},stride,offset);
        if(dtype.equals("float"))
            return create(new float[rows * columns],new int[]{rows,columns},stride,offset);
        throw new IllegalStateException("Illegal data type " + dtype);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape,int[] stride,int offset) {
        if(dtype.equals("double"))
            return createComplex(new double[ArrayUtil.prod(shape) * 2],shape,stride,offset);
        if(dtype.equals("float"))
            return createComplex(new float[ArrayUtil.prod(shape) * 2],shape,stride,offset);
        throw new IllegalStateException("Illegal data type " + dtype);

    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape,int[] stride,int offset) {
        if(dtype.equals("double"))
            return create(new double[ArrayUtil.prod(shape)],shape,stride,offset);
        if(dtype.equals("float"))
            return create(new float[ArrayUtil.prod(shape)],shape,stride,offset);
        throw new IllegalStateException("Illegal data type " + dtype);

    }







    /**
     * Creates a complex ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows,int columns,int[] stride) {
        return createComplex(new int[]{rows,columns},stride);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static INDArray create(int rows,int columns,int[] stride) {
        return create(new int[]{rows,columns},stride);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape,int[] stride) {
        return createComplex(shape,stride,0);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape,int[] stride) {
        return create(shape,stride,0);
    }




    /**
     * Creates a complex ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows,int columns) {
        return createComplex(new int[]{rows, columns});
    }


    /**
     * Creates an ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    public static INDArray create(int rows,int columns) {
        return create(new int[]{rows,columns});
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape) {
        return createComplex(shape, ArrayUtil.calcStrides(shape,2),0);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape) {
        return create(shape,ArrayUtil.calcStrides(shape),0);
    }


    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    public static INDArray scalar(Number value,int offset) {
        if(dtype.equals("double"))
            return scalar(value.doubleValue(),offset);
        if(dtype.equals("float"))
            return scalar(value.floatValue(),offset);
        throw new IllegalStateException("Illegal data type " + dtype);
    }


    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray complexScalar(Number value,int offset) {
        if(dtype.equals("double"))
            return scalar(createDouble(value.doubleValue(),0),offset);
        if(dtype.equals("float"))
            return scalar(createFloat(value.floatValue(),0),offset);
        throw new IllegalStateException("Illegal data type " + dtype);
    }


    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    public static IComplexNDArray complexScalar(Number value) {
        return complexScalar(value,0);
    }




    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static INDArray scalar(float value,int offset) {
        return create(new float[]{value},new int[]{1},new int[]{1},offset);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static INDArray scalar(double value,int offset) {
        return create(new double[]{value},new int[]{1},new int[]{1},offset);

    }



    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    public static INDArray scalar(Number value) {
        if(dtype.equals("double"))
            return scalar(value.doubleValue(),0);
        if(dtype.equals("float"))
            return scalar(value.floatValue(),0);
        throw new IllegalStateException("Illegal data type " + dtype);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
    =     * @return the scalar nd array
     */
    public static INDArray scalar(float value) {
        if(dtype.equals("float"))
            return create(new float[]{value},new int[]{1},new int[]{1},0);
        else
            return scalar((double) value);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
    =     * @return the scalar nd array
     */
    public static INDArray scalar(double value) {
        if(dtype.equals("double"))
            return create(new double[]{value},new int[]{1},new int[]{1},0);
        else
            return scalar((float) value);
    }


    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray scalar(IComplexNumber value,int offset) {
        if(dtype.equals("double"))
            return scalar(value.asDouble(),offset);
        if(dtype.equals("float"))
            return scalar(value.asFloat(),offset);
        throw new IllegalStateException("Illegal data type " + dtype);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexFloat value) {
        return createComplex(new float[]{value.realComponent(),value.imaginaryComponent()},new int[]{1},new int[]{1},0);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
    =     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexDouble value) {
        return createComplex(new double[]{value.realComponent(),value.imaginaryComponent()},new int[]{1},new int[]{1},0);

    }


    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    public static IComplexNDArray scalar(IComplexNumber value) {
        if(dtype.equals("double"))
            return scalar(value.asDouble(),0);
        if(dtype.equals("float"))
            return scalar(value.asFloat(),0);
        throw new IllegalStateException("Illegal data type " + dtype);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexFloat value,int offset) {
        return createComplex(new float[]{value.realComponent(),value.imaginaryComponent()},new int[]{1},new int[]{1},offset);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexDouble value,int offset) {
        return createComplex(new double[]{value.realComponent(),value.imaginaryComponent()},new int[]{1},new int[]{1},offset);

    }



}
