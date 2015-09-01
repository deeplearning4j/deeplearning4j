package org.nd4j.linalg.lossfunctions;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.nd4j.linalg.ops.transforms.Transforms.log;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

/**
 * @author Adam Gibson
 */
public @Data @Builder
class LossCalculation {
    private INDArray labels;
    private INDArray z;
    /** L1/L2 values: before division by miniBatchSize, but after multiplication by l1Coeff or 0.5*l2Coeff */
    private double l1,l2;
    private LossFunctions.LossFunction lossFunction;
    private boolean useRegularization;
    private INDArray delta;
    private boolean miniBatch = false;
    private int miniBatchSize;

    public double score() {
        double ret = 0.0;
        switch (lossFunction) {
            case CUSTOM: throw new IllegalStateException("Unable to score custom operation. Please define an alternative mechanism");
            case RECONSTRUCTION_CROSSENTROPY:
                INDArray xEntLogZ2 = log(z);
                INDArray xEntOneMinusLabelsOut2 = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ2 = log(z).rsubi(1);
                ret = -labels.mul(xEntLogZ2).add(xEntOneMinusLabelsOut2).muli(xEntOneMinusLogOneMinusZ2).sumNumber().doubleValue();
                break;
            case MCXENT:
                INDArray sums = labels.mul(log(z));
                ret = -sums.sumNumber().doubleValue();
                break;
            case XENT:
                INDArray xEntLogZ = log(z);
                INDArray xEntOneMinusLabelsOut = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ = log(z).rsubi(1);
                ret = labels.mul(xEntLogZ).add(xEntOneMinusLabelsOut).muli(xEntOneMinusLogOneMinusZ).sum(1).sumNumber().doubleValue();
                break;
            case RMSE_XENT:
                INDArray rmseXentDiff = delta == null ? labels.sub(z) : delta;
                INDArray squaredrmseXentDiff = pow(rmseXentDiff, 2.0);
                INDArray sqrt = sqrt(squaredrmseXentDiff);
                ret = sqrt.sumNumber().doubleValue();
                break;
            case MSE:
                INDArray mseDelta = delta == null ? labels.sub(z) : delta;
                ret = 0.5 * pow(mseDelta, 2).sum(1).sumNumber().doubleValue();
                break;
            case EXPLL:
                INDArray expLLLogZ = log(z);
                ret = z.sub(labels.mul(expLLLogZ)).sumNumber().doubleValue();
                break;
            case SQUARED_LOSS:
                ret = pow(delta == null ? labels.sub(z) : delta, 2).sumNumber().doubleValue();
                break;
            case NEGATIVELOGLIKELIHOOD:
                INDArray log = log(z);
                INDArray sums2 = labels.mul(log);
                ret = -sums2.sumNumber().doubleValue();
                break;
        }

        if (useRegularization) {
            ret += l1 + l2;
        }

        if(miniBatch)
            ret /= (double) miniBatchSize;
        return ret;
    }



}
