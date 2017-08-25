package org.nd4j.linalg.api.ops.impl.transforms.gradient;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseGradientOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 */
public class SigmoidDerivative extends BaseGradientOp  {
    public SigmoidDerivative(INDArray x, INDArray z) {
        super(x, z);
    }

    public SigmoidDerivative() {
    }

    public SigmoidDerivative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public SigmoidDerivative(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, z.lengthLong());
    }

    public SigmoidDerivative(INDArray x) {
        super(x);
    }

    /**
     * An op number
     *
     * @return
     */
    @Override
    public int opNum() {
        return 0;
    }

    /**
     * The name of this operation
     *
     * @return the name of this operation
     */
    @Override
    public String name() {
        return "sigmoidderivative";
    }



    /**
     * A copy of this operation for a particular dimension of the input
     *
     * @param index     the index of the op to iterate over
     * @param dimension the dimension to ge the input for
     * @return the operation for that dimension
     */
    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative(x.vectorAlongDimension(index, dimension),
                    y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension),
                    xAlongDimension.length());
        else
            return new org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative(x.vectorAlongDimension(index, dimension),
                    z.vectorAlongDimension(index, dimension), xAlongDimension.length());
    }

    /**
     * A copy of this operation for a particular dimension of the input
     *
     * @param index     the index of the op to iterate over
     * @param dimension the dimension to ge the input for
     * @return the operation for that dimension
     */
    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative(x.tensorAlongDimension(index, dimension),
                    y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension),
                    xAlongDimension.length());
        else
            return new org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative(x.tensorAlongDimension(index, dimension),
                    z.tensorAlongDimension(index, dimension), xAlongDimension.length());
    }

    @Override
    public void exec() {
        INDArray sigmoid = Nd4j.getExecutioner().execAndReturn(new Sigmoid(x));
        INDArray rsubbed = sigmoid.rsub(1);
        INDArray mulled = y.mul(sigmoid).muli(rsubbed);
        if(mulled != z)
            z.assign(mulled);

    }

    @Override
    public void exec(int... dimensions) {
        super.exec(dimensions);
    }
}
