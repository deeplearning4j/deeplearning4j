package org.deeplearning4j.nn.learning;

import org.deeplearning4j.nn.linalg.FourDTensor;
import org.jblas.DoubleMatrix;

import static org.jblas.MatrixFunctions.abs;
import static org.jblas.MatrixFunctions.pow;
import static org.jblas.MatrixFunctions.sqrt;

/**
 * Four dimensional tensor ada grad.
 * This allows for tensor and tensor by slice wise learning rates
 */
public class FourDTensorAdaGrad extends TensorAdaGrad {

    protected int tensors;

    public FourDTensorAdaGrad(int rows, int cols, int slices,int tensors, double gamma) {
        super(rows, cols, slices, gamma);
        this.tensors = tensors;
    }

    /**
     * Initializes the tensor ada grad with a gamma
     * of 1e-2
     *
     * @param rows   number of rows for the gradients
     * @param cols   the number of columns for the gradient
     * @param slices the number of slices for the gradient
     * @param tensors the number of tensors to use
     */
    public FourDTensorAdaGrad(int rows, int cols, int slices,int tensors) {
        super(rows, cols, slices);
        this.tensors = tensors;
    }



    /**
     * Tensor/Slice wise learning rates
     * @param tensor the tensor to use
     * @param slice the slice to use
     * @param gradient the gradient slice
     * @return the slice wise learning rates
     */
    public DoubleMatrix getLearningRates(int tensor,int slice,DoubleMatrix gradient) {
        this.gradient = gradient.dup();
        FourDTensor histGrad = (FourDTensor) this.historicalGradient;
        FourDTensor grad = (FourDTensor) this.gradient;
        FourDTensor adjustedGrad = (FourDTensor) adjustedGradient;
        DoubleMatrix squaredGradient = pow(grad.getSliceOfTensor(tensor,slice),2);
        if(this.historicalGradient.length != this.gradient.length)
            this.historicalGradient = FourDTensor.zeros(grad.rows(),this.gradient.columns,grad.slices());
        histGrad.put(tensor,slice,histGrad.getSlice(slice).add(squaredGradient));
        this.historicalGradient.addi(squaredGradient);
        numIterations++;
        DoubleMatrix sqrtGradient = sqrt(squaredGradient).add(fudgeFactor);
        DoubleMatrix div = abs(gradient).div(sqrtGradient);
        adjustedGrad.put(tensor,slice,div.mul(masterStepSize));
        //ensure no zeros
        return adjustedGrad.getSliceOfTensor(tensor,slice);
    }

    @Override
    protected void createHistoricalGradient() {
        if(rows > 0 && cols > 0 && slices > 0 && tensors > 0)
            this.historicalGradient = new FourDTensor(rows,cols,slices,tensors);
    }

    @Override
    protected void createAdjustedGradient() {
        if(rows > 0 && cols > 0 && slices > 0 && tensors > 0)
            this.adjustedGradient = new FourDTensor(rows,cols,slices,tensors);

    }
}
