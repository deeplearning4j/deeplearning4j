package org.deeplearning4j.linalg.factory;

import org.deeplearning4j.linalg.api.complex.IComplexDouble;
import org.deeplearning4j.linalg.api.complex.IComplexFloat;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.springframework.core.io.ClassPathResource;

import java.lang.reflect.Constructor;
import java.util.Properties;

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

    public final static String LINALG_PROPS = "/dl4j-linalg.properties";
    public final static String REAL_CLASS_PROP = "real.class";
    public final static String COMPLEX_CLASS_PROP = "complex.class";
    public final static String DTYPE = "dtype";
    public final static String COMPLEX_DOUBLE = "complex.double.class";
    public final static String COMPLEX_FLOAT = "complex.float.class";

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

            clazz = (Class<? extends INDArray>) Class.forName(realType);
            complexClazz = (Class<? extends IComplexNDArray>) Class.forName(complexType);
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
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
    public static INDArray complexScalar(Number value,int offset) {
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
    public static INDArray complexScalar(Number value) {
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
