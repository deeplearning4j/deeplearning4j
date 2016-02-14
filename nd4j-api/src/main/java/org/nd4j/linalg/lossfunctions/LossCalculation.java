package org.nd4j.linalg.lossfunctions;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LogSoftMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.conditions.Or;
import org.nd4j.linalg.indexing.functions.StableNumber;
import org.nd4j.linalg.indexing.functions.Value;

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
    private boolean miniBatch = false;
    private int miniBatchSize;
    private String activationFn;
    private INDArray preOut;
    private INDArray mask;

    /** Score the entire (mini)batch */
    public double score(){
        INDArray exampleScores = scoreArray();
        double ret = exampleScores.sumNumber().doubleValue();
        switch(lossFunction){
            case MCXENT:
            case NEGATIVELOGLIKELIHOOD:
            case RECONSTRUCTION_CROSSENTROPY:
                ret *= -1;
                break;
            case MSE:
                ret *= 0.5;
                break;
        }

        if (useRegularization) {
            ret += l1 + l2;
        }

        if(miniBatch)
            ret /= (double) miniBatchSize;

        return ret;
    }

    /** Calculate the score for each example individually.
     * @return If labels are shape [miniBatchSize,nOut] then return shape is [miniBatchSize,1] with value at position i
     * being the score for example i
     */
    public INDArray scoreExamples(){
        INDArray exampleScores = scoreArray().sum(1);

        switch(lossFunction){
            case MCXENT:
            case NEGATIVELOGLIKELIHOOD:
            case RECONSTRUCTION_CROSSENTROPY:
                exampleScores.muli(-1);
                break;
            case MSE:
                exampleScores.muli(0.5);
                break;
        }

        double l = l1+l2;
        if (useRegularization && l != 0.0) {
            exampleScores.addi(l);
        }

        return exampleScores;
    }

    private INDArray scoreArray() {
        INDArray scoreArray;    //shape: [batchSize,nOut]
        switch (lossFunction) {
            case CUSTOM: throw new IllegalStateException("Unable to score custom operation. Please define an alternative mechanism");
            case RECONSTRUCTION_CROSSENTROPY:
                INDArray xEntLogZ2 = logZ(z);
                INDArray xEntOneMinusLabelsOut2 = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ2 = xEntLogZ2.rsubi(1);
                INDArray temp = labels.mul(xEntLogZ2).add(xEntOneMinusLabelsOut2).muli(xEntOneMinusLogOneMinusZ2);
                if(mask != null) temp.muliColumnVector(mask);
                scoreArray = temp;
                break;
            case NEGATIVELOGLIKELIHOOD:
            case MCXENT:
                if(preOut != null && "softmax".equals(activationFn)){
                    //Use LogSoftMax op to avoid numerical issues when calculating score
                    INDArray logsoftmax = Nd4j.getExecutioner().execAndReturn(new LogSoftMax(preOut.dup()), 1);
                    INDArray sums = labels.mul(logsoftmax);
                    if(mask != null) sums.muliColumnVector(mask);
                    scoreArray = sums;
                } else {
                    //Standard calculation
                    INDArray sums = labels.mul(logZ(z));
                    if(mask != null) sums.muliColumnVector(mask);
                    scoreArray = sums;
                }
                break;
            case XENT:
                INDArray xEntLogZ = logZ(z);
                INDArray xEntOneMinusLabelsOut = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ = xEntLogZ.dup().rsubi(1);
                INDArray temp2 = labels.mul(xEntLogZ).add(xEntOneMinusLabelsOut).muli(xEntOneMinusLogOneMinusZ);
                if(mask != null) temp2.muliColumnVector(mask);
                scoreArray = temp2;
                break;
            case RMSE_XENT:
                INDArray rmseXentDiff = labels.sub(z);
                INDArray squaredrmseXentDiff = pow(rmseXentDiff, 2.0);
                INDArray sqrt = sqrt(squaredrmseXentDiff);
                if(mask != null) sqrt.muliColumnVector(mask);
                scoreArray = sqrt;
                break;
            case MSE:
                INDArray mseDeltaSquared = labels.sub(z);
                mseDeltaSquared.muli(mseDeltaSquared);
                if(mask != null) mseDeltaSquared.muliColumnVector(mask);
                scoreArray = mseDeltaSquared;
                break;
            case EXPLL:
                INDArray expLLLogZ = logZ(z);
                INDArray temp3 = z.sub(labels.mul(expLLLogZ));
                if(mask != null) temp3.muliColumnVector(mask);
                scoreArray = temp3;
                break;
            case SQUARED_LOSS:
                INDArray labelsSubZSquared = labels.sub(z);
                labelsSubZSquared.muli(labelsSubZSquared);
                if(mask != null) labelsSubZSquared.muliColumnVector(mask);
                scoreArray = labelsSubZSquared;
                break;
            default:
                throw new RuntimeException("Unknown loss function: " + lossFunction);
        }

        return scoreArray;
    }


    private static INDArray logZ(INDArray z) {
        INDArray log = log(z, true);

        // log approaches -Infinity as z approaches zero.  Replace -Infinity with the least possible value.
        // Caveat: does not handle +Infinity since z is assumed to be 0 <= z <= 1.
        switch(log.data().dataType()) {
            case FLOAT:
                BooleanIndexing.applyWhere(log, new Or(Conditions.isNan(),Conditions.isInfinite()), new StableNumber(StableNumber.Type.FLOAT));
                break;
            case DOUBLE:
                BooleanIndexing.applyWhere(log, new Or(Conditions.isNan(),Conditions.isInfinite()), new StableNumber(StableNumber.Type.DOUBLE));

                break;
            case INT:
                BooleanIndexing.applyWhere(log, new Or(Conditions.isNan(),Conditions.isInfinite()), new Value(-Integer.MAX_VALUE));
                break;
            default:
                throw new RuntimeException("unsupported data type: " + log.data().dataType());
        }
        return log;
    }

}
