package org.nd4j.examples;

import jcuda.jcublas.JCublas;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.BlasWrapper;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jblas.JblasNDArrayFactory;
import org.nd4j.linalg.jcublas.JCublasNDArrayFactory;
import org.nd4j.linalg.jcublas.JCublasWrapper;

/**
 * Created by agibsonccc on 9/15/14.
 */
public class MultiINDArrayInterop {

    static NDArrayFactory jblas = new JblasNDArrayFactory("double",'f');
    static NDArrayFactory jcublas = new JCublasNDArrayFactory("double",'f');
    static BlasWrapper wrapper = new org.nd4j.linalg.jblas.BlasWrapper();
    static  BlasWrapper jcublasWrapper = new JCublasWrapper();


    public static void main(String[] args) {

        INDArray jblasLinspace = jblas.linspace(1,8,8);
        INDArray jcublasLinspace = jcublas.linspace(1,8,8);

        setJblas();

        INDArray transpose = jblasLinspace.transpose();

        setJcublas();

        INDArray jcublastranspose = jcublasLinspace.transpose();
        setJblas();

        INDArray mmul = jblasLinspace.mmul(transpose);
        setJcublas();

        INDArray mmul2 = jcublastranspose.mmul(jcublastranspose);
        assert mmul.equals(mmul2);



        setJblas();
        INDArray jblasreshape = jblasLinspace.reshape(2,4);
        INDArray reshapetranspose = jblasreshape.transpose();
        INDArray jblasmmul = reshapetranspose.mmul(jblasreshape);
        setJcublas();
        INDArray jcublasreshape = jcublasLinspace.reshape(2, 4);
        INDArray jcublasreshapetranspose = jblasreshape.transpose();
        assert reshapetranspose.equals(jcublasreshapetranspose);
        assert jcublasreshape.equals(jblasreshape);

        INDArray jcublasmmul = jcublasreshapetranspose.mmul(jcublasreshape);

        assert jblasmmul.equals(jcublasmmul);




        setJblas();
        INDArray toTransposeJblas = jblas.create(new float[]{1,2,3,4},new int[]{2,2});
        setJcublas();
        INDArray toTransposeJcublas = jcublas.create(new float[]{1,2,3,4},new int[]{2,2});
        setJblas();
        INDArray transposeToTransposeJblas = toTransposeJblas.transpose();
        setJcublas();
        INDArray transposeToTransposeJcublas = toTransposeJcublas.transpose();
        assert toTransposeJblas.equals(toTransposeJcublas);
        assert transposeToTransposeJblas.equals(transposeToTransposeJcublas);

        setJblas();
        float dotjblas = Nd4j.getBlasWrapper().dot(jblasLinspace,jblasLinspace);
        setJcublas();
        float dotjcublas = Nd4j.getBlasWrapper().dot(jcublasLinspace,jcublasLinspace);
        assert dotjblas == dotjcublas;



    }

    public static void setJcublas() {
        Nd4j.setFactory(jcublas);
        Nd4j.setBlasWrapper(jcublasWrapper);

    }



    public static void setJblas() {
        Nd4j.setFactory(jblas);
        Nd4j.setBlasWrapper(wrapper);

    }

}
