package org.deeplearning4j.eval;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.Not;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Map;

/**
 * ROC (Receiver Operating Characteristic) for multi-task binary classifiers, using the specified number of threshold steps.
 * <p>
 * Some ROC implementations will automatically calculate the threshold points based on the data set to give a 'smoother'
 * ROC curve (or optimal cut points for diagnostic purposes). This implementation currently uses fixed steps of size
 * 1.0 / thresholdSteps, as this allows easy implementation for batched and distributed evaluation scenarios (where the
 * full data set is not available in memory on any one machine at once).
 * <p>
 * Unlike {@link ROC} (which supports a single binary label (as a single column probability, or 2 column 'softmax' probability
 * distribution), ROCBinary assumes that all outputs are independent binary variables. This also differs from
 * {@link ROCMultiClass}, which should be used for multi-class (single non-binary) cases.
 * <p>
 * ROCBinary supports per-example and per-output masking: for per-output masking, any particular output may be absent
 * (mask value 0) and hence won't be included in the calculated ROC.
 */
public class ROCBinary extends BaseEvaluation<ROCBinary> {

    private final int thresholdSteps;
    private long[] countActualPositive;
    private long[] countActualNegative;
    private Map<Double,CountsForThreshold> countsForThresholdMap;

    public ROCBinary(int thresholdSteps) {
        this.thresholdSteps = thresholdSteps;
    }


    @Override
    public void eval(INDArray labels, INDArray networkPredictions) {
        eval(labels, networkPredictions, (INDArray) null);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray) {
        if (countActualPositive != null && countActualPositive.length != labels.size(1)) {
            throw new IllegalStateException("Labels array does not match stored state size. Expected labels array with "
                    + "size " + countActualPositive.length + ", got labels array with size " + labels.size(1));
        }

        if(labels.rank() == 3){
            evalTimeSeries(labels, networkPredictions, maskArray);
            return;
        }

        if(countActualPositive == null){
            //Initialize
            countActualPositive = new long[labels.size(1)];
            countActualNegative = new long[labels.size(1)];

            double step = 1.0 / thresholdSteps;
            for (int i = 0; i <= thresholdSteps; i++) {
                double currThreshold = i * step;
                countsForThresholdMap.put(currThreshold, new CountsForThreshold(currThreshold, labels.size(1)));
            }
        }

        //First: need to increment actual positive/negative (label counts) for each output
        INDArray actual1 = labels;
        INDArray actual0 = Nd4j.getExecutioner().execAndReturn( new Not(labels.dup()));
        if(maskArray != null){
            actual1 = actual1.mul(maskArray);
            actual0.muli(maskArray);
        }

        int[] countActualPosThisBatch = actual1.sum(0).data().asInt();
        int[] countActualNegThisBatch = actual0.sum(0).data().asInt();
        addInPlace(countActualPositive, countActualPosThisBatch);
        addInPlace(countActualNegative, countActualNegThisBatch);

        //Here: calculate true positive rate (TPR) vs. false positive rate (FPR) at different threshold
        double step = 1.0 / thresholdSteps;
        for (int i = 0; i <= thresholdSteps; i++) {
            double currThreshold = i * step;

            //Work out true/false positives - do this by replacing probabilities (predictions) with 1 or 0 based on threshold
            Condition condGeq = Conditions.greaterThanOrEqual(currThreshold);
            Condition condLeq = Conditions.lessThanOrEqual(currThreshold);

            Op op = new CompareAndSet(networkPredictions.dup(), 1.0, condGeq);
            INDArray predictedClass1 = Nd4j.getExecutioner().execAndReturn(op);
            op = new CompareAndSet(predictedClass1, 0.0, condLeq);
            predictedClass1 = Nd4j.getExecutioner().execAndReturn(op);

            //True positives: occur whet the predicted and actual are both 1s
            //False positives: occur when predicted 1, actual is 0
            INDArray isTruePositive = predictedClass1.mul(actual1);
            INDArray isFalsePositive = predictedClass1.mul(actual0);

            //Apply mask array:
            if (maskArray != null) {
                if(Arrays.equals(labels.shape(), maskArray.shape())){
                    //Per output masking
                    isTruePositive.muli(maskArray);
                    isFalsePositive.muli(maskArray);
                } else {
                    //Per-example masking
                    isTruePositive.muliColumnVector(maskArray);
                    isFalsePositive.muliColumnVector(maskArray);
                }
            }

            //TP/FP counts for this threshold
            int[] truePositiveCount = isTruePositive.sum(0).data().asInt();
            int[] falsePositiveCount = isFalsePositive.sum(0).data().asInt();

            CountsForThreshold cft = countsForThresholdMap.get(currThreshold);
            cft.incrementTruePositive(truePositiveCount);
            cft.incrementFalsePositive(falsePositiveCount);
        }

    }

    private static void addInPlace(long[] addTo, int[] toAdd){
        for( int i=0; i<addTo.length; i++ ){
            addTo[i] += toAdd[i];
        }
    }

    private static void addInPlace(long[] addTo, long[] toAdd){
        for( int i=0; i<addTo.length; i++ ){
            addTo[i] += toAdd[i];
        }
    }

    @Override
    public void merge(ROCBinary other) {
        if(this.countActualPositive == null){
            this.countActualPositive = other.countActualPositive;
            this.countActualNegative = other.countActualNegative;
            this.countsForThresholdMap = other.countsForThresholdMap;
            return;
        } else if(other.countActualPositive == null){
            return;
        }

        if(this.countActualPositive.length != other.countActualPositive.length){
            throw new IllegalStateException("Cannot merge ROCBinary instances with different number of coulmns. " +
                    "numColumns = " + this.countActualPositive.length + "; other numColumns = " + other.countActualPositive.length);
        }

        //Both have data
        addInPlace(this.countActualPositive, other.countActualPositive);
        addInPlace(this.countActualNegative, other.countActualNegative);
        for(Map.Entry<Double,CountsForThreshold> e : countsForThresholdMap.entrySet()){
            CountsForThreshold o = other.countsForThresholdMap.get(e.getKey());

            e.getValue().incrementTruePositive(o.getCountTruePositive());
            e.getValue().incrementFalsePositive(o.getCountFalsePositive());
        }
    }


    @AllArgsConstructor
    @Data
    public static class CountsForThreshold implements Serializable, Cloneable {
        private double threshold;
        private long[] countTruePositive;
        private long[] countFalsePositive;

        public CountsForThreshold(double threshold, int size) {
            this(threshold, new long[size], new long[size]);
        }

        public void incrementTruePositive(int[] counts){
            addInPlace(countTruePositive, counts);
        }

        public void incrementFalsePositive(int[] counts){
            addInPlace(countFalsePositive, counts);
        }

        public void incrementTruePositive(long[] counts){
            addInPlace(countTruePositive, counts);
        }

        public void incrementFalsePositive(long[] counts){
            addInPlace(countFalsePositive, counts);
        }

        public void incrementTruePositive(long count, int index) {
            countTruePositive[index] += count;
        }

        public void incrementFalsePositive(long count, int index) {
            countFalsePositive[index] += count;
        }

        @Override
        public ROCBinary.CountsForThreshold clone() {
            long[] ctp = ArrayUtils.clone(countTruePositive);
            long[] tfp = ArrayUtils.clone(countFalsePositive);
            return new ROCBinary.CountsForThreshold(threshold, ctp, tfp);
        }
    }
}
