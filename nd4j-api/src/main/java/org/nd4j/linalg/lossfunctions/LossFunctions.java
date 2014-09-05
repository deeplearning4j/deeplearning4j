package org.nd4j.linalg.lossfunctions;

import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

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
        assert !NDArrays.hasInvalidNumber(output) : "Invalid output on labels. Must not contain nan or infinite numbers.";
        float ret = 0.0f;
        double reg = 0.5 * l2;
        INDArray z = output;
        switch (lossFunction) {
            case RECONSTRUCTION_CROSSENTROPY:
                INDArray xEntLogZ2 = Transforms.log(z.dup());
                INDArray xEntOneMinusLabelsOut2 = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ2 = Transforms.log(z).rsubi(1);
                ret = - labels.mul(xEntLogZ2).add(xEntOneMinusLabelsOut2).mul(xEntOneMinusLogOneMinusZ2).sum(1).sum(Integer.MAX_VALUE).get(0) / labels.rows();
                break;
            case MCXENT:
                INDArray columnSums = labels.mul(log(z));
                ret = - columnSums.sum(1).sum(Integer.MAX_VALUE).get(0) / labels.rows();
                break;
            case XENT:
                INDArray xEntLogZ =  log(z);
                INDArray xEntOneMinusLabelsOut = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ =  log(z).rsubi(1);
                ret = -labels.mul(xEntLogZ).add(xEntOneMinusLabelsOut).mul(xEntOneMinusLogOneMinusZ).sum(1).sum(Integer.MAX_VALUE).get(0) / labels.rows();
                break;
            case RMSE_XENT:
                INDArray rmseXentDiff = labels.sub(z);
                INDArray squaredrmseXentDiff = pow(rmseXentDiff,2);
                INDArray sqrt = sqrt(squaredrmseXentDiff);
                ret =   sqrt.sum(1).sum(Integer.MAX_VALUE).get(0) / labels.rows();
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
    public static float reconEntropy(INDArray input,INDArray hBias,INDArray vBias,INDArray W,ActivationFunction activationFunction) {
        INDArray preSigH = input.mmul(W).addRowVector(hBias);
        INDArray sigH = activationFunction.apply(preSigH);
        assert !NDArrays.hasInvalidNumber(sigH);
        //transpose doesn't go in right
        INDArray preSigV = sigH.mmul(W.transpose()).addRowVector(vBias);
        INDArray sigV = activationFunction.apply(preSigV);
        assert !NDArrays.hasInvalidNumber(sigH);

        INDArray inner =
                input.mul(log(sigV))
                        .add(input.rsub(1)
                                .mul(log(sigV.rsub(1))));


        INDArray rows = inner.sum(1);
        INDArray mean = rows.mean(Integer.MAX_VALUE);

        float ret = (float) mean.element();

        ret /= input.rows();

        return  ret;
    }



}
