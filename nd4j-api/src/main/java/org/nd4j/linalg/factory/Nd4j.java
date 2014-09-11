package org.nd4j.linalg.factory;

import com.google.common.primitives.Ints;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.springframework.core.io.ClassPathResource;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.*;

/**
 *
 * Creation of ndarrays via classpath discovery.
 *
 *
 * @author Adam Gibson
 */
public class Nd4j {

    private static Class<? extends BlasWrapper> blasWrapperClazz;
    private static Class<? extends NDArrayFactory> ndArrayFactoryClazz;

    private static BlasWrapper BLAS_WRAPPER_INSTANCE;
    public final static String LINALG_PROPS = "/nd4j.properties";
    public final static String REAL_CLASS_PROP = "real.class";
    public final static String COMPLEX_CLASS_PROP = "complex.class";
    public final static String DTYPE = "dtype";
    public final static String BLAS_OPS = "blas.ops";
    public static String dtype;
    public static char ORDER = 'c';
    public final static String ORDER_KEY = "ndarray.order";
    public final static String NDARRAY_FACTORY_CLASS = "ndarrayfactory.class";
    private static NDArrayFactory INSTANCE;
    private static Properties props = new Properties();
    public final static IComplexNumber UNIT;
    public final static IComplexNumber ZERO;
    public final static IComplexNumber NEG_UNIT;


    static {
        try {
            ClassPathResource c = new ClassPathResource(LINALG_PROPS);
            props.load(c.getInputStream());
            dtype = props.get(DTYPE).toString();
            ORDER = props.getProperty(ORDER_KEY,"c").toString().charAt(0);

            ndArrayFactoryClazz = (Class<? extends NDArrayFactory>) Class.forName(props.get(NDARRAY_FACTORY_CLASS).toString());
            Constructor c2 = ndArrayFactoryClazz.getConstructor(String.class,Character.class);
            INSTANCE = (NDArrayFactory) c2.newInstance(dtype,ORDER);
            blasWrapperClazz = (Class<? extends BlasWrapper>) Class.forName(props.get(BLAS_OPS).toString());
            BLAS_WRAPPER_INSTANCE = blasWrapperClazz.newInstance();
            UNIT = Nd4j.createFloat(1, 0);
            ZERO = Nd4j.createFloat(1, 0);
            NEG_UNIT = Nd4j.createFloat(-1, 0);
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }


    public static NDArrayFactory factory() {
        return INSTANCE;
    }




    /**
     * Returns the ordering of the ndarrays
     * @return the ordering of the ndarrays
     */
    public static Character order() {
        return factory().order();
    }

    /**
     * Returns the data type used for the runtime
     * @return the datatype used for the runtime
     */
    public static String dataType() {
        return dtype;
    }

    public static BlasWrapper getBlasWrapper() {
        return BLAS_WRAPPER_INSTANCE;
    }


    /**
     * Create an n x (shape)
     * ndarray where the ndarray is repeated num times
     * @param n the ndarray to replicate
     * @param num the number of copies to repeat
     * @return the repeated ndarray
     */
    public static IComplexNDArray repeat(IComplexNDArray n,int num) {
        List<IComplexNDArray> list = new ArrayList<>();
        for(int i = 0; i < num; i++)
            list.add(n);
        return Nd4j.createComplex(list, Ints.concat(new int[]{num},n.shape()));
    }

    /**
     * Create an n x (shape)
     * ndarray where the ndarray is repeated num times
     * @param n the ndarray to replicate
     * @param num the number of copies to repeat
     * @return the repeated ndarray
     */
    public static INDArray repeat(INDArray n,int num) {
        List<INDArray> list = new ArrayList<>();
        for(int i = 0; i < num; i++)
            list.add(n);
        return Nd4j.create(list, Ints.concat(new int[]{num},n.shape()));
    }

    /**
     * Generate a linearly spaced vector
     * @param lower upper bound
     * @param upper lower bound
     * @param num the step size
     * @return the linearly spaced vector
     */
    public static INDArray linspace(int lower,int upper,int num) {
        return INSTANCE.linspace(lower,upper,num);
    }



    public static INDArray toFlattened(Collection<INDArray> matrices) {
        return INSTANCE.toFlattened(matrices);

    }

    public static IComplexNDArray complexFlatten(List<IComplexNDArray> flatten) {
        return INSTANCE.complexFlatten(flatten);
    }

    public static IComplexNDArray complexFlatten(IComplexNDArray...flatten) {
        return INSTANCE.complexFlatten(flatten);
    }


    public static INDArray toFlattened(int length,Iterator<? extends INDArray>...matrices) {
        return INSTANCE.toFlattened(length,matrices);
    }


    /**
     * Returns a column vector where each entry is the nth bilinear
     * product of the nth slices of the two tensors.
     */
    public static INDArray bilinearProducts(INDArray curr,INDArray in) {
        return INSTANCE.bilinearProducts(curr,in);
    }


    public static INDArray toFlattened(INDArray...matrices) {
        return INSTANCE.toFlattened(matrices);
    }


    /**
     * Create the identity ndarray
     * @param n the number for the identity
     * @return
     */
    public static INDArray eye(int n) {
        return INSTANCE.eye(n);

    }

    /**
     * Rotate a matrix 90 degrees
     * @param toRotate the matrix to rotate
     * @return the rotated matrix
     */
    public static void rot90(INDArray toRotate) {
        INSTANCE.rot90(toRotate);

    }


    /**
     * Read in an ndarray from a data input stream
     * @param dis the data input stream to read from
     * @return the ndarray
     * @throws IOException
     */
    public static INDArray read(DataInputStream dis) throws IOException {
        int dimensions = dis.readInt();
        int[] shape = new int[dimensions];
        int[] stride = new int[dimensions];

        for(int i = 0; i < dimensions; i++)
            shape[i] = dis.readInt();
        for(int  i = 0; i < dimensions; i++)
            stride[i] = dis.readInt();
        String dataType = dis.readUTF();
        String type = dis.readUTF();

        if(!type.equals("real"))
            throw new IllegalArgumentException("Trying to read in a complex ndarray");

        if(dataType.equals("float")) {
            float[] data = ArrayUtil.readFloat(ArrayUtil.prod(shape),dis);
            return create(data,shape,stride,0);
        }

        float[] data = ArrayUtil.readFloat(ArrayUtil.prod(shape),dis);
        return create(data,shape,stride,0);
    }


    /**
     * Write an ndarray to the specified outputs tream
     * @param arr the array to write
     * @param dataOutputStream the data output stream to write to
     * @throws IOException
     */
    public static void write(INDArray arr,DataOutputStream dataOutputStream) throws IOException {
        dataOutputStream.writeInt(arr.shape().length);
        for(int i = 0; i < arr.shape().length; i++)
            dataOutputStream.writeInt(arr.size(i));
        for(int i = 0; i < arr.stride().length; i++)
            dataOutputStream.writeInt(arr.stride()[i]);

        dataOutputStream.writeUTF(dataType());

        dataOutputStream.writeUTF("real");

        if(dataType().equals("float"))
            ArrayUtil.write(arr.floatData(),dataOutputStream);
        else
            ArrayUtil.write(arr.data(),dataOutputStream);

    }


    /**
     * Create an ndarray based on the given data layout
     * @param data the data to use
     * @return an ndarray with the given data layout
     */
    public static INDArray create(double[][] data) {
        return INSTANCE.create(data);
    }


    /**
     * Read in an ndarray from a data input stream
     * @param dis the data input stream to read from
     * @return the ndarray
     * @throws IOException
     */
    public static IComplexNDArray readComplex(DataInputStream dis) throws IOException {
        int dimensions = dis.readInt();
        int[] shape = new int[dimensions];
        int[] stride = new int[dimensions];

        for(int i = 0; i < dimensions; i++)
            shape[i] = dis.readInt();
        for(int  i = 0; i < dimensions; i++)
            stride[i] = dis.readInt();
        String dataType = dis.readUTF();

        String type = dis.readUTF();

        if(!type.equals("complex"))
            throw new IllegalArgumentException("Trying to read in a real ndarray");

        if(dataType.equals("float")) {
            float[] data = ArrayUtil.readFloat(ArrayUtil.prod(shape),dis);
            return createComplex(data,shape,stride,0);
        }

        double[] data = ArrayUtil.read(ArrayUtil.prod(shape),dis);
        return createComplex(data,shape,stride,0);
    }


    /**
     * Write an ndarray to the specified outputs tream
     * @param arr the array to write
     * @param dataOutputStream the data output stream to write to
     * @throws IOException
     */
    public static void writeComplex(IComplexNDArray arr,DataOutputStream dataOutputStream) throws IOException {
        dataOutputStream.writeInt(arr.shape().length);
        for(int i = 0; i < arr.shape().length; i++)
            dataOutputStream.writeInt(arr.size(i));
        for(int i = 0; i < arr.stride().length; i++)
            dataOutputStream.writeInt(arr.stride()[i]);
        dataOutputStream.writeUTF(dataType());

        dataOutputStream.writeUTF("complex");

        if(dataType().equals("float"))
            ArrayUtil.write(arr.floatData(),dataOutputStream);
        else
            ArrayUtil.write(arr.data(),dataOutputStream);

    }


    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     * @param reverse the matrix to reverse
     * @return the reversed matrix
     */
    public static INDArray rot(INDArray reverse) {
        return INSTANCE.rot(reverse);
    }

    /**
     * Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc
     * @param reverse the matrix to reverse
     * @return the reversed matrix
     */
    public static INDArray reverse(INDArray reverse) {
        return INSTANCE.reverse(reverse);
    }


    /**
     *  Array of evenly spaced values.
     * @param begin the begin of the range
     * @param end the end of the range
     * @return the range vector
     */
    public static INDArray arange(double begin, double end) {
        return INSTANCE.arange(begin,end);
    }

    /**
     * Create float
     * @param real real component
     * @param imag imag component
     * @return
     */
    public static IComplexFloat createComplexNumber(Number real,Number imag) {
        return INSTANCE.createFloat(real.floatValue(),imag.floatValue());
    }
    /**
     * Create float
     * @param real real component
     * @param imag imag component
     * @return
     */
    public static IComplexFloat createFloat(float real,float imag) {
        return INSTANCE.createFloat(real,imag);
    }


    /**
     * Create an instance of a complex double
     * @param real the real component
     * @param imag the imaginary component
     * @return a new imaginary double with the specified real and imaginary components
     */
    public static IComplexDouble createDouble(double real,double imag) {
        return INSTANCE.createDouble(real,imag);
    }


    /**
     * Copy a to b
     * @param a the origin matrix
     * @param b the destination matrix
     */
    public static void copy(INDArray a,INDArray b) {
        INSTANCE.copy(a,b);
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
        return INSTANCE.rand(shape,min,max,rng);
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
        return INSTANCE.rand(rows,columns,min,max,rng);
    }


    public static INDArray appendBias(INDArray...vectors) {
        return INSTANCE.appendBias(vectors);
    }


    /**
     * Create a complex ndarray from the passed in indarray
     * @param arr the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(INDArray arr) {
        if(arr instanceof  IComplexNDArray)
            return (IComplexNDArray) arr;
        return INSTANCE.createComplex(arr);
    }


    /**
     * Create a complex ndarray from the passed in indarray
     * @param data the data to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(IComplexNumber[] data,int[] shape) {
        return INSTANCE.createComplex(data, shape);
    }


    /**
     * Create a complex ndarray from the passed in indarray
     * @param data the data to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(IComplexNumber[] data,int[] shape,int offset,char ordering) {
        return INSTANCE.createComplex(data, shape,offset,ordering);
    }


    /**
     * Create a complex ndarray from the passed in indarray
     * @param arrs the arr to wrap
     * @return the complex ndarray with the specified ndarray as the
     * real components
     */
    public static IComplexNDArray createComplex(List<IComplexNDArray> arrs,int[] shape) {
        return INSTANCE.createComplex(arrs,shape);
    }


    /**
     * Create a random ndarray with the given shape using the given rng
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows,int columns,RandomGenerator r) {
        return INSTANCE.rand(rows,columns,r);
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     * @param rows the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param seed the  seed to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows,int columns,long seed) {
        return INSTANCE.rand(rows,columns,seed);
    }
    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int rows,int columns) {
        return INSTANCE.rand(rows,columns);
    }

    /**
     * Random normal using the given rng
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @param r the random generator to use
     * @return
     */
    public static INDArray randn(int rows,int columns,RandomGenerator r) {
        return INSTANCE.randn(rows,columns,r);
    }

    /**
     * Random normal using the current time stamp
     * as the seed
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    public static INDArray randn(int rows,int columns) {
        return INSTANCE.rand(rows,columns);
    }

    /**
     * Random normal using the specified seed
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the matrix
     * @return
     */
    public static INDArray randn(int rows,int columns,long seed) {
        return INSTANCE.randn(rows, columns, seed);
    }





    /**
     * Create a random ndarray with the given shape using the given rng
     * @param shape the shape of the ndarray
     * @param r the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape,RealDistribution r) {
        return INSTANCE.rand(shape,r);
    }
    /**
     * Create a random ndarray with the given shape using the given rng
     * @param shape the shape of the ndarray
     * @param r the random generator to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape,RandomGenerator r) {
        return INSTANCE.rand(shape,r);
    }

    /**
     * Create a random ndarray with the given shape using the given rng
     * @param shape the shape of the ndarray
     * @param seed the  seed to use
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape,long seed) {
        return INSTANCE.rand(shape,seed);
    }
    /**
     * Create a random ndarray with the given shape using
     * the current time as the seed
     * @param shape the shape of the ndarray
     * @return the random ndarray with the specified shape
     */
    public static INDArray rand(int[] shape) {
        return INSTANCE.rand(shape);
    }

    /**
     * Random normal using the given rng
     * @param shape the shape of the ndarray
     * @param r the random generator to use
     * @return
     */
    public static INDArray randn(int[] shape,RandomGenerator r) {
        return INSTANCE.randn(shape, r);
    }

    /**
     * Random normal using the current time stamp
     * as the seed
     * @param shape the shape of the ndarray
     * @return
     */
    public static INDArray randn(int[] shape) {
        return INSTANCE.randn(shape);
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
    public static INDArray create(double[] data,char order) {
        return INSTANCE.create(data);
    }

    /**
     * Creates a row vector with the data
     * @param data the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray create(float[] data,char order) {
        return INSTANCE.create(data);
    }

    /**
     * Creates an ndarray with the specified data
     * @param data the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray createComplex(double[] data,char order) {
        return INSTANCE.createComplex(data);
    }



    /**
     * Creates a row vector with the specified number of columns
     * @param columns the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray create(int columns,char order) {
        return INSTANCE.create(columns);
    }

    /**
     * Creates an ndarray
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray createComplex(int columns,char order) {
        return INSTANCE.createComplex(columns);
    }




    /**
     * Creates a row vector with the data
     * @param data the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray create(double[] data) {
        return create(data,order());
    }

    /**
     * Creates a row vector with the data
     * @param data the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray create(float[] data) {
        return create(data,order());
    }

    /**
     * Creates an ndarray with the specified data
     * @param data the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray createComplex(double[] data) {
        return createComplex(data, Nd4j.order());

    }



    /**
     * Creates a row vector with the specified number of columns
     * @param columns the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray create(int columns) {
        return create(columns,order());
    }

    /**
     * Creates an ndarray
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray createComplex(int columns)  {
        return createComplex(columns,order());
    }


    /**
     * Returns true if the given ndarray has either
     * an infinite or nan
     * @param num the ndarray to test
     * @return true if the given ndarray has either infinite or nan
     * false otherwise
     */
    public static boolean hasInvalidNumber(INDArray num) {
        INDArray linear = num.linearView();
        for(int i = 0;i < linear.length(); i++) {
            if(Float.isInfinite(linear.get(i)) || Float.isNaN(linear.get(i)))
                return true;
        }
        return false;
    }


    /**
     * Creates a row vector with the specified number of columns
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray zeros(int rows,int columns) {
        return INSTANCE.zeros(rows, columns);
    }

    /**
     * Creates a matrix of zeros
     * @param rows te number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static INDArray complexZeros(int rows,int columns) {
        return INSTANCE.complexZeros(rows,columns);
    }


    /**
     * Creates a row vector with the specified number of columns
     * @param columns the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray zeros(int columns) {
        return INSTANCE.zeros(columns);
    }

    /**
     * Creates an ndarray
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static INDArray complexZeros(int columns) {
        return INSTANCE.complexZeros(columns);
    }


    public static IComplexNDArray complexValueOf(int num,IComplexNumber value) {
        return INSTANCE.complexValueOf(num,value);
    }

    public static IComplexNDArray complexValueOf(int[] shape,IComplexNumber value) {
        return INSTANCE.complexValueOf(shape,value);
    }

    public static IComplexNDArray complexValueOf(int num,double value) {
        return INSTANCE.complexValueOf(num,value);
    }

    public static IComplexNDArray complexValueOf(int[] shape,double value) {
        return INSTANCE.complexValueOf(shape,value);
    }

    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray
     * @param shape the shape of the ndarray
     * @param value the value to assign
     * @return  the created ndarray
     */
    public static INDArray valueArrayOf(int[] shape,double value) {
        return INSTANCE.valueArrayOf(shape,value);
    }


    /**
     * Creates an ndarray with the specified value
     * as the  only value in the ndarray
     * @param num number of columns
     * @param value the value to assign
     * @return  the created ndarray
     */
    public static INDArray valueArrayOf(int num,double value) {
        return INSTANCE.valueArrayOf(new int[]{1,num},value);
    }


    /**
     * Creates a row vector with the specified number of columns
     * @param rows the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @param value the value to assign
     * @return  the created ndarray
     */
    public static INDArray valueArrayOf(int rows,int columns,double value) {
        return INSTANCE.valueArrayOf(rows, columns, value);
    }




    /**
     * Creates a row vector with the specified number of columns
     * @param rows the number of rows in the matrix
     * @param columns the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray ones(int rows,int columns) {
        return INSTANCE.ones(rows,columns);
    }


    /**
     * Creates an ndarray
     * @param rows the number of rows in the matrix
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static INDArray complexOnes(int rows,int columns) {
        return INSTANCE.complexOnes(rows, columns);
    }

    /**
     * Creates a row vector with the specified number of columns
     * @param columns the columns of the ndarray
     * @return  the created ndarray
     */
    public static INDArray ones(int columns) {
        return INSTANCE.ones(columns);
    }

    /**
     * Creates an ndarray
     * @param columns the number of columns in the row vector
     * @return ndarray
     */
    public static IComplexNDArray complexOnes(int columns) {
        return INSTANCE.complexOnes(columns);
    }



    /**
     * Concatenates two matrices horizontally. Matrices must have identical
     * numbers of rows.
     * @param A the first matrix to concat
     * @param B  the second matrix to concat
     */
    public static INDArray hstack(INDArray A, INDArray B) {
        return INSTANCE.hstack(A, B);
    }

    /**
     * Concatenates two matrices vertically. Matrices must have identical
     * numbers of columns.
     * @param A
     * @param B
     */
    public static INDArray vstack(INDArray A, INDArray B) {
        return INSTANCE.vstack(A, B);
    }


    /**
     * Concatneate ndarrays along a dimension
     * @param dimension the dimension to concatneate along
     * @param toConcat the ndarrays to concat
     * @return the concatted ndarrays with an output shape of
     * the ndarray shapes save the dimension shape specified
     * which is then the sum of the sizes along that dimension
     */
    public static INDArray concat(int dimension,INDArray...toConcat) {
        return INSTANCE.concat(dimension,toConcat);
    }

    /**
     * Concatneate ndarrays along a dimension
     * @param dimension the dimension to concatneate along
     * @param toConcat the ndarrays to concat
     * @return the concatted ndarrays with an output shape of
     * the ndarray shapes save the dimension shape specified
     * which is then the sum of the sizes along that dimension
     */
    public static IComplexNDArray concat(int dimension,IComplexNDArray...toConcat) {
        return INSTANCE.concat(dimension,toConcat);
    }



    /**
     * Create an ndarray of zeros
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static INDArray zeros(int[] shape) {
        return INSTANCE.zeros(shape);


    }

    /**
     * Create an ndarray of ones
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static IComplexNDArray complexZeros(int[] shape) {
        return INSTANCE.complexZeros(shape);

    }


    /**
     * Create an ndarray of ones
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static INDArray ones(int[] shape) {
        return INSTANCE.ones(shape);

    }

    /**
     * Create an ndarray of ones
     * @param shape the shape of the ndarray
     * @return an ndarray with ones filled in
     */
    public static IComplexNDArray complexOnes(int[] shape) {
        return INSTANCE.complexOnes(shape);

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
        return INSTANCE.createComplex(data, rows, columns, stride, offset);
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
        return INSTANCE.create(data, rows, columns, stride, offset);
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
        return INSTANCE.createComplex(data, shape, stride, offset);
    }




    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(float[] data,int[] shape,int[] stride,int offset) {
        return INSTANCE.create(data,shape,stride,offset);
    }


    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(double[] data,int[] shape) {
        return  INSTANCE.create(data,shape);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(float[] data,int[] shape) {
        return INSTANCE.create(data,shape);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(double[] data,int[] shape) {
        return INSTANCE.createComplex(data,shape);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(float[] data,int[] shape) {
        return INSTANCE.createComplex(data,shape);
    }



    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(double[] data,int[] shape,int[] stride) {
        return INSTANCE.createComplex(data, shape, stride);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(float[] data,int[] shape,int[] stride) {
        return INSTANCE.createComplex(data,shape,stride);
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
        return INSTANCE.createComplex(data,rows,columns,stride,offset);
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
        return  INSTANCE.create(data,rows,columns,stride,offset);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(double[] data,int[] shape,int[] stride,int offset) {
        return INSTANCE.createComplex(data,shape,stride,offset);
    }



    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data,int[] shape,int offset) {
        return create(data, shape, offset, Nd4j.order());
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data,int[] shape,int offset,char ordering) {
        return INSTANCE.create(data, shape, Nd4j.getStrides(shape), offset);
    }

    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data,int[] shape,int[] stride,int offset) {
        return INSTANCE.create(data,shape,stride,offset);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray create(List<INDArray> list,int[] shape) {
        return INSTANCE.create(list,shape);
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
        return INSTANCE.createComplex(rows,columns,stride,offset);
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
        return INSTANCE.create(rows, columns, stride, offset);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape,int[] stride,int offset) {
        return INSTANCE.createComplex(shape, stride, offset);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape,int[] stride,int offset) {
        return INSTANCE.create(shape,stride,offset);

    }







    /**
     * Creates a complex ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows,int columns,int[] stride) {
        return createComplex(rows,columns,stride,order());
    }


    /**
     * Creates an ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static INDArray create(int rows,int columns,int[] stride) {
        return create(rows,columns,stride,order());
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape,int[] stride) {
        return createComplex(shape,stride,order());
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape,int[] stride) {
        return create(shape,stride,order());
    }




    /**
     * Creates a complex ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows,int columns) {
        return createComplex(rows,columns,order());
    }


    /**
     * Creates an ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    public static INDArray create(int rows,int columns) {
        return create(rows,columns,order());
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape) {
        return createComplex(shape,order());
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape) {
        return create(shape,order());
    }


    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    public static INDArray scalar(Number value,int offset) {
        return INSTANCE.scalar(value,offset);
    }


    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray complexScalar(Number value,int offset) {
        return INSTANCE.complexScalar(value,offset);
    }


    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    public static IComplexNDArray complexScalar(Number value) {
        return INSTANCE.complexScalar(value);
    }




    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static INDArray scalar(float value,int offset) {
        return INSTANCE.scalar(value,offset);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static INDArray scalar(double value,int offset) {
        return INSTANCE.scalar(value,offset);

    }



    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    public static INDArray scalar(Number value) {
        return INSTANCE.scalar(value);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
    =     * @return the scalar nd array
     */
    public static INDArray scalar(float value) {
        return INSTANCE.scalar(value);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
    =     * @return the scalar nd array
     */
    public static INDArray scalar(double value) {
        return INSTANCE.scalar(value);
    }


    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @param offset the offset of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray scalar(IComplexNumber value,int offset) {
        return INSTANCE.scalar(value,offset);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexFloat value) {
        return INSTANCE.scalar(value);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
    =     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexDouble value) {
        return INSTANCE.scalar(value);

    }


    /**
     * Create a scalar ndarray with the specified offset
     * @param value the value to initialize the scalar with
     * @return the created ndarray
     */
    public static IComplexNDArray scalar(IComplexNumber value) {
        return INSTANCE.scalar(value);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexFloat value,int offset) {
        return INSTANCE.scalar(value,offset);
    }

    /**
     * Create a scalar nd array with the specified value and offset
     * @param value the value of the scalar
     * @param offset the offset of the ndarray
     * @return the scalar nd array
     */
    public static IComplexNDArray scalar(IComplexDouble value,int offset) {
        return INSTANCE.scalar(value,offset);

    }


    /**
     * Get the strides for the given order and shape
     * @param shape the shape of the ndarray
     * @param order the order to getScalar the strides for
     * @return the strides for the given shape and order
     */
    public static int[] getStrides(int[] shape,char order) {
        if(order == NDArrayFactory.FORTRAN)
            return ArrayUtil.calcStridesFortran(shape);
        return ArrayUtil.calcStrides(shape);
    }

    /**
     * Get the strides based on the shape
     * and NDArrays.order()
     * @param shape the shape of the ndarray
     * @return the strides for the given shape
     * and order specified by NDArrays.order()
     */
    public static int[] getStrides(int[] shape) {
        return getStrides(shape, Nd4j.order());
    }




    /**
     * Get the strides for the given order and shape
     * @param shape the shape of the ndarray
     * @param order the order to getScalar the strides for
     * @return the strides for the given shape and order
     */
    public static int[] getComplexStrides(int[] shape,char order) {
        if(order == NDArrayFactory.FORTRAN)
            return ArrayUtil.calcStridesFortran(shape,2);
        return ArrayUtil.calcStrides(shape,2);
    }

    /**
     * Get the strides based on the shape
     * and NDArrays.order()
     * @param shape the shape of the ndarray
     * @return the strides for the given shape
     * and order specified by NDArrays.order()
     */
    public static int[] getComplexStrides(int[] shape) {
        return getComplexStrides(shape, Nd4j.order());
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
    public static IComplexNDArray createComplex(float[] data,int rows,int columns,int[] stride,int offset,char ordering) {
        return INSTANCE.createComplex(data, rows, columns, stride, offset);
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
    public static INDArray create(float[] data,int rows,int columns,int[] stride,int offset,char ordering) {
        return INSTANCE.create(data, rows, columns, stride, offset);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param data the data to use with the ndarray
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(float[] data,int[] shape,int[] stride,int offset,char ordering) {
        return INSTANCE.createComplex(data, shape, stride, offset);
    }




    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(float[] data,int[] shape,int[] stride,int offset,char ordering) {
        return INSTANCE.create(data,shape,stride,offset,ordering);
    }


    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(double[] data,int[] shape,char ordering) {
        return  INSTANCE.create(data,shape);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static INDArray create(float[] data,int[] shape,char ordering) {
        return INSTANCE.create(data,shape);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(double[] data,int[] shape,char ordering) {
        return INSTANCE.createComplex(data,shape);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(float[] data,int[] shape,char ordering) {
        return INSTANCE.createComplex(data,shape);
    }



    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(double[] data,int[] shape,int[] stride,char ordering) {
        return INSTANCE.createComplex(data, shape, stride);
    }

    /**
     * Create an ndrray with the specified shape
     * @param data the data to use with tne ndarray
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the created ndarray
     */
    public static IComplexNDArray createComplex(float[] data,int[] shape,int[] stride,char ordering) {
        return INSTANCE.createComplex(data,shape,stride);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(double[] data,int rows,int columns,int[] stride,int offset,char ordering) {
        return INSTANCE.createComplex(data,rows,columns,stride,offset);
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
    public static INDArray create(double[] data,int rows,int columns,int[] stride,int offset,char ordering) {
        return  INSTANCE.create(data,rows,columns,stride,offset);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(double[] data,int[] shape,int[] stride,int offset,char ordering) {
        return INSTANCE.createComplex(data,shape,stride,offset);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(double[] data,int[] shape,int[] stride,int offset,char ordering) {
        return INSTANCE.create(data,shape,stride,offset);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray create(List<INDArray> list,int[] shape,char ordering) {
        return INSTANCE.create(list,shape);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows,int columns,int[] stride,int offset,char ordering) {
        return INSTANCE.createComplex(rows,columns,stride,offset);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(int rows,int columns,int[] stride,int offset,char ordering) {
        return INSTANCE.create(rows, columns, stride, offset);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset  the offset of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape,int[] stride,int offset,char ordering) {
        return INSTANCE.createComplex(shape, stride, offset);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape,int[] stride,int offset,char ordering) {
        return INSTANCE.create(shape,stride,offset);

    }







    /**
     * Creates a complex ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows,int columns,int[] stride,char ordering) {
        return INSTANCE.createComplex(rows, columns, stride);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static INDArray create(int rows,int columns,int[] stride,char ordering) {
        return INSTANCE.create(rows, columns, stride);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape,int[] stride,char ordering) {
        return createComplex(shape,stride,0);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @param stride the stride for the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape,int[] stride,char ordering) {
        return INSTANCE.create(shape,stride);
    }




    /**
     * Creates a complex ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int rows,int columns,char ordering) {
        return INSTANCE.createComplex(rows,columns);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param rows the rows of the ndarray
     * @param columns the columns of the ndarray
     * @return the instance
     */
    public static INDArray create(int rows,int columns,char ordering) {
        return INSTANCE.create(rows,columns);
    }



    /**
     * Creates a complex ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static IComplexNDArray createComplex(int[] shape,char ordering) {
        return INSTANCE.createComplex(shape);
    }


    /**
     * Creates an ndarray with the specified shape
     * @param shape the shape of the ndarray
     * @return the instance
     */
    public static INDArray create(int[] shape,char ordering) {
        return INSTANCE.create(shape);
    }


    public static IComplexNDArray createComplex(double[] data, int[] ints, int offset, char ordering) {
        return INSTANCE.createComplex(data,ints,ArrayUtil.calcStrides(ints,2),offset,ordering);
    }

    public static IComplexNDArray createComplex(double[] data, int[] shape, int offset) {
        return createComplex(data, shape, offset, Nd4j.order());
    }

    public static INDArray create(float[] data, int[] shape, int offset) {
        return INSTANCE.create(data,shape,offset);
    }

    public static IComplexNDArray createComplex(float[] data, int[] ints, int offset, char ordering) {
        return INSTANCE.createComplex(data,ints,offset,ordering);
    }

    public static IComplexNDArray createComplex(float[] dim) {
        return INSTANCE.createComplex(dim);
    }

    public static IComplexNDArray createComplex(float[] data, int[] shape, int offset) {
        return INSTANCE.createComplex(data,shape,offset);
    }

    public static INDArray create(float[][] floats) {
        return INSTANCE.create(floats);
    }

    public static IComplexNDArray complexLinSpace(int i, int i1, int i2) {
        return Nd4j.createComplex(Nd4j.linspace(1,8,8));

    }


    public static INDArray create(float[] data, int[] shape, int[] stride,char ordering, int offset) {
        return INSTANCE.create(data,shape,stride,offset,ordering);
    }

    public static INDArray create(float[] data, int[] shape, char ordering, int offset) {
        return INSTANCE.create(data,shape,getStrides(shape,ordering),offset,ordering);
    }


}
