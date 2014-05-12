package org.deeplearning4j.nn.learning;

import org.deeplearning4j.nn.Tensor;
import org.jblas.DoubleMatrix;

import static org.jblas.MatrixFunctions.abs;
import static org.jblas.MatrixFunctions.pow;
import static org.jblas.MatrixFunctions.sqrt;

/**
 * Tensor ada grad
 * This is for handling slicewise learning rates
 * @author Adam Gibson
 */
public class TensorAdaGrad extends AdaGrad {

    protected int slices;

    public TensorAdaGrad(int rows, int cols,int slices, double gamma) {
        super(rows, cols, gamma);
        this.slices = slices;
    }

    /**
     * Initializes the tensor adad grad with a gamma
     * of 1e-2
     * @param rows number of rows for the gradients
     * @param cols the number of columns for the gradient
     * @param slices the number of slices for the gradient
     */
    public TensorAdaGrad(int rows, int cols,int slices) {
        super(rows, cols);
        this.slices = slices;
    }

    @Override
    protected void createHistoricalGradient() {
       this.historicalGradient = new Tensor(rows,cols,slices);
    }

    @Override
    protected void createAdjustedGradient() {
       this.adjustedGradient = new Tensor(rows,cols,slices);
    }


    /**
     * Slice wise learning rates
     * @param slice the slice to use
     * @param gradient the gradient slice
     * @return the slice wise learning rates
     */
    public DoubleMatrix getLearningRates(int slice,DoubleMatrix gradient) {
        this.gradient = gradient.dup();
        Tensor histGrad = (Tensor) this.historicalGradient;
        Tensor grad = (Tensor) this.gradient;
        Tensor adjustedGrad = (Tensor) adjustedGradient;
        DoubleMatrix squaredGradient = pow(grad.getSlice(slice),2);
        if(this.historicalGradient.length != this.gradient.length)
            this.historicalGradient = Tensor.zeros(grad.rows(),this.gradient.columns,grad.slices());
        histGrad.setSlice(slice,histGrad.getSlice(slice).add(squaredGradient));
        this.historicalGradient.addi(squaredGradient);
        numIterations++;
        DoubleMatrix sqrtGradient = sqrt(squaredGradient).add(fudgeFactor);
        DoubleMatrix div = abs(gradient).div(sqrtGradient);
        adjustedGrad.setSlice(slice,div.mul(masterStepSize));
        //ensure no zeros
        return adjustedGrad.getSlice(slice);
    }

    /**
     * Gets feature specific learning rates
     * Adagrad keeps a history of gradients being passed in.
     * Note that each gradient passed in becomes adapted over time, hence
     * the name adagrad
     *
     * @param gradient the gradient to get learning rates for
     * @return the feature specific learning rates
     */
    @Override
    public DoubleMatrix getLearningRates(DoubleMatrix gradient) {
        return super.getLearningRates(gradient);
    }
}
