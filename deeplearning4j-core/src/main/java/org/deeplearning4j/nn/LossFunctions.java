package org.deeplearning4j.nn;

import org.jblas.DoubleMatrix;

import static org.deeplearning4j.util.MatrixUtil.log;
import static org.deeplearning4j.util.MatrixUtil.oneMinus;
import static org.jblas.MatrixFunctions.pow;

/**
 * Central class for loss functions
 */
public class LossFunctions {


    /**
     * MSE: Mean Squared Error: Linear Regression
     * EXPLL: Exponential log likelihood: Poisson Regression
     * XENT: Cross Entropy: Binary Classification
     * SOFTMAX: Softmax Regression
     * RMSE_XENT: RMSE Cross Entropy
     *
     *
     */
    public static enum LossFunction {
        MSE,EXPLL,XENT,MCXENT,RMSE_XENT,SQUARED_LOSS,RECONSTRUCTION_CROSSENTROPY,NEGATIVELOGLIKELIHOOD
    }


    /**
     * Generic scoring function
     * @param input the inputs to score
     * @param labels the labels to score
     * @param lossFunction the loss function to use
     * @param output the output function
     * @param l2 the l2 coefficient
     * @param useRegularization  whether to use regularization
     * @return the score for the given parameters
     */
    public static double score(DoubleMatrix input,DoubleMatrix labels,LossFunction lossFunction,Output output,double l2,boolean useRegularization) {
        double ret = 0.0;
        double reg = 0.5 * l2;
        DoubleMatrix z = output.output(input);
        switch (lossFunction) {
            case MCXENT:
                DoubleMatrix mcXEntLogZ = log(z);
                ret = - labels.mul(mcXEntLogZ).columnSums().sum() / labels.rows;
                break;
            case XENT:
                DoubleMatrix xEntLogZ = log(z);
                DoubleMatrix xEntOneMinusLabelsOut = oneMinus(labels);
                DoubleMatrix xEntOneMinusLogOneMinusZ = oneMinus(log(z));
                ret = -labels.mul(xEntLogZ).add(xEntOneMinusLabelsOut).mul(xEntOneMinusLogOneMinusZ).columnSums().sum() / labels.rows;
                break;
            case RMSE_XENT:
                ret = pow(labels.sub(z),2).columnSums().sum() / labels.rows;
                break;
            case MSE:
                DoubleMatrix mseDelta = labels.sub(z);
                ret = 0.5 * pow(mseDelta, 2).columnSums().sum() / labels.rows;
                break;
            case EXPLL:
                DoubleMatrix expLLLogZ = log(z);
                ret = -z.sub(labels.mul(expLLLogZ)).columnSums().sum() / labels.rows;
                break;
            case SQUARED_LOSS:
                ret = pow(labels.sub(z),2).columnSums().sum() / labels.rows;


        }

        if(useRegularization && l2 > 0)
            ret += reg;


        return ret;

    }

}
