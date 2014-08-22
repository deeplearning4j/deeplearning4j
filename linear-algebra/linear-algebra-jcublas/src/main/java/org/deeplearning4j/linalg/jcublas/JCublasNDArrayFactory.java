package org.deeplearning4j.linalg.jcublas;

import org.deeplearning4j.linalg.api.complex.IComplexDouble;
import org.deeplearning4j.linalg.api.complex.IComplexFloat;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.BaseNDArrayFactory;
import org.deeplearning4j.linalg.jcublas.complex.JCublasComplexDouble;
import org.deeplearning4j.linalg.jcublas.complex.JCublasComplexFloat;
import org.deeplearning4j.linalg.jcublas.complex.JCublasComplexNDArray;

import java.util.List;

/**
 * Created by mjk on 8/21/14.
 */
public class JCublasNDArrayFactory extends BaseNDArrayFactory {
    public JCublasNDArrayFactory(String dtype) {
        super(dtype);
    }

    @Override
    public IComplexFloat createFloat(float real, float imag) {
        return new JCublasComplexFloat(real,imag);
    }

    @Override
    public IComplexDouble createDouble(double real, double imag) {
        return new JCublasComplexDouble(real,imag);
    }

    @Override
    public IComplexNDArray createComplex(INDArray arr) {
        return new JCublasComplexNDArray(arr);
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape) {
        return new JCublasComplexNDArray(data,shape);
    }

    @Override
    public IComplexNDArray createComplex(List<IComplexNDArray> arrs, int[] shape) {
        return new JCublasComplexNDArray(arrs,shape);
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset) {
        return new JCublasComplexNDArray(data,shape,stride,offset);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, int offset) {
        return new JCublasNDArray(data,shape,stride,offset);
    }

    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset) {
        return new JCublasComplexNDArray(data,shape,stride,offset);

    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, int offset) {
        return new JCublasNDArray(data,shape,stride,offset);
    }

    @Override
    public INDArray create(List<INDArray> list, int[] shape) {
        return new JCublasNDArray(list,shape);
    }
}
