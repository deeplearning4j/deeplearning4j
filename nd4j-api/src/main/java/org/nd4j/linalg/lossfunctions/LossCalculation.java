package org.nd4j.linalg.lossfunctions;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
    private double l1,l1Magnitude,l2,l2Magnitude;
    private LossFunctions.LossFunction lossFunction;
    private boolean useRegularization;
    private INDArray delta;

    public double score() {
        double ret = 0.0;
        switch (lossFunction) {
            case CUSTOM: throw new IllegalStateException("Unable to score custom operation. Please define an alternative mechanism");
            case RECONSTRUCTION_CROSSENTROPY:
                INDArray xEntLogZ2 = log(z);
                INDArray xEntOneMinusLabelsOut2 = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ2 = log(z).rsubi(1);
                ret = -labels.mul(xEntLogZ2).add(xEntOneMinusLabelsOut2).muli(xEntOneMinusLogOneMinusZ2).sum(Integer.MAX_VALUE).getDouble(0);
                break;
            case MCXENT:
                INDArray sums = log(z);
                INDArray columnSums = labels.mul(sums);
                ret = -columnSums.sum(Integer.MAX_VALUE).getDouble(0);
                break;
            case XENT:
                INDArray xEntLogZ = log(z);
                INDArray xEntOneMinusLabelsOut = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ = log(z).rsubi(1);
                ret = labels.mul(xEntLogZ).add(xEntOneMinusLabelsOut).muli(xEntOneMinusLogOneMinusZ).sum(1).sum(Integer.MAX_VALUE).getDouble(0);
                break;
            case RMSE_XENT:
                INDArray rmseXentDiff = delta == null ? labels.sub(z) : delta;
                INDArray squaredrmseXentDiff = pow(rmseXentDiff, 2.0);
                INDArray sqrt = sqrt(squaredrmseXentDiff);
                ret = sqrt.sum(1).sum(Integer.MAX_VALUE).getDouble(0);
                break;
            case MSE:
                INDArray mseDelta = delta == null ? labels.sub(z) : delta;
                ret = 0.5 * pow(mseDelta, 2).sum(1).sum(Integer.MAX_VALUE).getDouble(0);
                break;
            case EXPLL:
                INDArray expLLLogZ = log(z);
                ret = z.sub(labels.mul(expLLLogZ)).sum(Integer.MAX_VALUE).getDouble(0);
                break;
            case SQUARED_LOSS:
                ret = pow(delta == null ? labels.sub(z) : delta, 2).sum(Integer.MAX_VALUE).getDouble(0);
                break;
            case NEGATIVELOGLIKELIHOOD:
                ret = -Nd4j.sum(
                        labels.mul(log(z))
                                .addi(labels.rsub(1).muli(log(z.rsub(1))))
                        , 1).getDouble(0);



        }

        if (useRegularization) {
            ret += l2 * l2Magnitude;
            ret += l1 * l1Magnitude;
        }


        ret /= (double) labels.rows();
        return ret;
    }



}
