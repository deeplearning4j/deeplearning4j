package org.deeplearning4j.linalg.lossfunctions;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.ops.transforms.Transforms;

import static org.deeplearning4j.linalg.ops.transforms.Transforms.*;

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
        MSE,
        EXPLL,
        XENT,
        MCXENT,
        RMSE_XENT,
        SQUARED_LOSS,
        RECONSTRUCTION_CROSSENTROPY,
        NEGATIVELOGLIKELIHOOD
    }


    /**
     * Generic scoring function
     * @param labels the labels to score
     * @param lossFunction the loss function to use
     * @param output the output function
     * @param l2 the l2 coefficient
     * @param useRegularization  whether to use regularization
     * @return the score for the given parameters
     */
    public static float score(INDArray labels,LossFunction lossFunction,INDArray output,double l2,boolean useRegularization) {
        float ret = 0.0f;
        double reg = 0.5 * l2;
        INDArray z = output;
        switch (lossFunction) {
            case RECONSTRUCTION_CROSSENTROPY:
                INDArray xEntLogZ2 = Transforms.log(z.dup());
                INDArray xEntOneMinusLabelsOut2 = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ2 = Transforms.log(z.dup()).rsubi(1);
                ret = - labels.mul(xEntLogZ2).add(xEntOneMinusLabelsOut2).mul(xEntOneMinusLogOneMinusZ2).sum(1).sum(Integer.MAX_VALUE).get(0) / labels.rows();
                break;
            case MCXENT:
                INDArray mcXEntLogZ =  log(z);
                ret = -  labels.mul(mcXEntLogZ).sum(1).sum(Integer.MAX_VALUE).get(0) / labels.rows();
                break;
            case XENT:
                INDArray xEntLogZ =  log(z);
                INDArray xEntOneMinusLabelsOut = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ =  log(z).rsubi(1);
                ret = -labels.mul(xEntLogZ).add(xEntOneMinusLabelsOut).mul(xEntOneMinusLogOneMinusZ).sum(1).sum(Integer.MAX_VALUE).get(0) / labels.rows();
                break;
            case RMSE_XENT:
                ret =   sqrt( pow(labels.sub(z),2)).sum(1).sum(Integer.MAX_VALUE).get(0) / labels.rows();
                break;
            case MSE:
                INDArray mseDelta = labels.sub(z);
                ret = 0.5f *   pow(mseDelta, 2).sum(1).sum(Integer.MAX_VALUE).get(0) / labels.rows();
                break;
            case EXPLL:
                INDArray expLLLogZ = log(z);
                ret =  -   z.sub(labels.mul(expLLLogZ)).sum(1).sum(Integer.MAX_VALUE).get(0) / labels.rows();
                break;
            case SQUARED_LOSS:
                ret = (float)  pow(labels.sub(z), 2).sum(1).sum(Integer.MAX_VALUE).element() / labels.rows();


        }

        if(useRegularization && l2 > 0)
            ret += reg;


        return ret;

    }

    /**
     * Reconstruction entropy for Denoising AutoEncoders and RBMs
     * @param input the input ndarray
     * @param hBias the hidden bias of the neural network
     * @param vBias the visible bias of the neural network
     * @param W the weight matrix of the neural network
     * @return the reconstruction cross entropy for the given parameters
     */
    public static float reconEntropy(INDArray input,INDArray hBias,INDArray vBias,INDArray W) {
        INDArray inputTimesW = input.mmul(W);
        INDArray preSigH = input.mmul(W).addRowVector(hBias);
        INDArray sigH = sigmoid(preSigH.dup());

        //transpose doesn't go in right
        INDArray preSigV = sigH.mmul(W.transpose()).addRowVector(vBias);
        INDArray sigV = sigmoid(preSigV.dup());
        INDArray inner =
                input.mul(log(sigV.dup()))
                        .addi(input.rsub(1)
                                .muli(log(sigV.rsub(1))));


        INDArray rows = inner.sum(0);
        INDArray mean = rows.mean(Integer.MAX_VALUE);

        float ret = (float) mean.element();

        ret /= input.rows();

        return  ret;
    }



}
