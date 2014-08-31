package org.deeplearning4j.linalg.api.activation;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrayFactory;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.ArrayOps;
import org.deeplearning4j.linalg.ops.ElementWiseOp;

/**
 * Softmax function 
 * @author Adam Gibson
 *
 */
public class SoftMax extends BaseActivationFunction {
    //whether to take row wise or column wise maxes on softmax calculation
    private boolean rows;

    /**
     * Initialize softmax with whether to use row wise or column wise features
     * @param rows whether to use row wise or column wise features for calculation
     */
    public SoftMax(boolean rows) {
        this.rows = rows;
    }

    /**
     * Initializes softmax with column wise features
     */
    public SoftMax() {
        this(false);
    }


    /**
     * Soft max function
     * row_maxes is a row vector (max for each row)
     * row_maxes = rowmaxes(input)
     * diff = exp(input - max) / diff.rowSums()
     *
     * @param input the input for the softmax
     * @param row whether the row maxes should be taken or the column maxes,
     *            this is dependent on whether the features are column wise or row wise

     * @return the softmax output (a probability matrix) scaling each row to between
     * 0 and 1
     */
    public static  INDArray softmax(INDArray input,boolean row) {
        //leveraging row sums and row maxes
        if(row) {
            if(input.ordering() == NDArrayFactory.FORTRAN) {
                INDArray max = input.max(1);
                INDArray diff = input.subColumnVector(max);
                new ArrayOps()
                        .from(diff)
                        .op(org.deeplearning4j.linalg.ops.transforms.Exp.class)
                        .build().exec();
                diff.diviColumnVector(diff.sum(1).transpose());
                return diff;
            }
            else {
                INDArray max = input.max(1);

                INDArray diff = input.subColumnVector(max);
                new ArrayOps()
                        .from(diff)
                        .op(org.deeplearning4j.linalg.ops.transforms.Exp.class)
                        .build().exec();
                diff.diviColumnVector(diff.sum(1).transpose());
                return diff;
            }


        }

        //column sums and column maxes
        else {

            if(input.ordering() == NDArrayFactory.FORTRAN) {
                INDArray max = input.max(0).transpose();
                INDArray diff =  input.subRowVector(max);
                new ArrayOps()
                        .from(diff)
                        .op(org.deeplearning4j.linalg.ops.transforms.Exp.class)
                        .build().exec();
                diff.diviRowVector(diff.sum(0));
                return diff;
            }
            else {
                INDArray max = input.max(0).transpose();
                INDArray diff =  input.subRowVector(max);
                new ArrayOps()
                        .from(diff)
                        .op(org.deeplearning4j.linalg.ops.transforms.Exp.class)
                        .build().exec();
                diff.diviRowVector(diff.sum(0));
                return diff;
            }




        }
    }


    /**
     *
     */
    private static final long serialVersionUID = -3407472284248637360L;

    @Override
    public INDArray apply(INDArray input) {
        return softmax(input,rows);
    }

    /**
     * The class used for transformation
     *
     * @return the class used for transformation
     */
    @Override
    public Class<? extends ElementWiseOp> transformClazz() {
        return null;
    }

    @Override
    public INDArray applyDerivative(INDArray input) {
        if(input instanceof IComplexNDArray)
            return softmax(input,rows).mul(NDArrays.complexOnes(input.shape()).subi(softmax(input, rows)));
        else
            return softmax(input,rows).mul(NDArrays.ones(input.shape()).subi(softmax(input, rows)));

    }

}
