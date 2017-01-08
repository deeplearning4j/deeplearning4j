package org.deeplearning4j.eval;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.io.Serializable;
import java.util.*;

/**
 * ROC (Receiver Operating Characteristic) for multi-class classifiers, using the specified number of threshold steps.
 * <p>
 * The ROC curves are produced by treating the predictions as a set of one-vs-all classifiers, and then calculating
 * ROC curves for each. In practice, this means for N classes, we get N ROC curves.
 * <p>
 * Some ROC implementations will automatically calculate the threshold points based on the data set to give a 'smoother'
 * ROC curve (or optimal cut points for diagnostic purposes). This implementation currently uses fixed steps of size
 * 1.0 / thresholdSteps, as this allows easy implementation for batched and distributed evaluation scenarios (where the
 * full data set is not available in memory on any one machine at once).
 *
 * @author Alex Black
 */
@Getter
public class ROCMultiClass extends BaseEvaluation<ROCMultiClass> {

    private final int thresholdSteps;

    private long[] countActualPositive;
    private long[] countActualNegative;

    private final Map<Integer,Map<Double, ROC.CountsForThreshold>> counts = new LinkedHashMap<>();

    /**
     * @param thresholdSteps Number of threshold steps to use for the ROC calculation
     */
    public ROCMultiClass(int thresholdSteps) {
        this.thresholdSteps = thresholdSteps;
    }


    /**
     * Evaluate (collect statistics for) the given minibatch of data.
     * For time series (3 dimensions) use {@link #evalTimeSeries(INDArray, INDArray)} or {@link #evalTimeSeries(INDArray, INDArray, INDArray)}
     *
     * @param labels      Labels / true outcomes
     * @param predictions Predictions
     */
    public void eval(INDArray labels, INDArray predictions) {
        if (labels.rank() == 3 && predictions.rank() == 3) {
            //Assume time series input -> reshape to 2d
            evalTimeSeries(labels, predictions);
        }
        if (labels.rank() > 2 || predictions.rank() > 2 || labels.size(1) != predictions.size(1)) {
            throw new IllegalArgumentException("Invalid input data shape: labels shape = " + Arrays.toString(labels.shape()) +
                    ", predictions shape = " + Arrays.toString(predictions.shape()) + "; require rank 2 array with size(1) == 1 or 2");
        }

        double step = 1.0 / thresholdSteps;

        if(countActualPositive == null){
            //This must be the first time eval has been called...
            int size = labels.size(1);
            countActualPositive = new long[size];
            countActualNegative = new long[size];

            for( int i=0; i<size; i++ ){
                Map<Double,ROC.CountsForThreshold> map = new LinkedHashMap<Double, ROC.CountsForThreshold>();
                counts.put(i, map);

                for (int j = 0; j <= thresholdSteps; j++) {
                    double currThreshold = j * step;
                    map.put(currThreshold, new ROC.CountsForThreshold(currThreshold));
                }
            }
        }

        if(countActualPositive.length != labels.size(1)){
            throw new IllegalArgumentException("Cannot evaluate data: number of label classes does not match previous call. " +
                    "Got " + labels.size(1) + " labels (from array shape " + Arrays.toString(labels.shape()) + ")" +
                    " vs. expected number of label classes = " + countActualPositive.length);
        }

        for( int i=0; i<countActualPositive.length; i++ ){
            //Iterate over each class
            INDArray positiveActualColumn = labels.getColumn(i);
            INDArray positivePredictedColumn = predictions.getColumn(i);

            //Increment global counts - actual positive/negative observed
            long currBatchPositiveActualCount = positiveActualColumn.sumNumber().intValue();
            countActualPositive[i] += currBatchPositiveActualCount;
            countActualNegative[i] += positiveActualColumn.length() - currBatchPositiveActualCount;

            //Here: calculate true positive rate (TPR) vs. false positive rate (FPR) at different threshold

            for (int j = 0; j <= thresholdSteps; j++) {
                double currThreshold = j * step;

                //Work out true/false positives - do this by replacing probabilities (predictions) with 1 or 0 based on threshold
                Condition condGeq = Conditions.greaterThanOrEqual(currThreshold);
                Condition condLeq = Conditions.lessThanOrEqual(currThreshold);

                Op op = new CompareAndSet(positivePredictedColumn.dup(), 1.0, condGeq);
                INDArray predictedClass1 = Nd4j.getExecutioner().execAndReturn(op);
                op = new CompareAndSet(predictedClass1, 0.0, condLeq);
                predictedClass1 = Nd4j.getExecutioner().execAndReturn(op);


                //True positives: occur when positive predicted class and actual positive actual class...
                //False positive occurs when positive predicted class, but negative actual class
                INDArray isTruePositive = predictedClass1.mul(positiveActualColumn);       //If predicted == 1 and actual == 1 at this threshold: 1x1 = 1. 0 otherwise
                INDArray negativeActualColumn = positiveActualColumn.rsub(1.0);
                INDArray isFalsePositive = predictedClass1.mul(negativeActualColumn);      //If predicted == 1 and actual == 0 at this threshold: 1x1 = 1. 0 otherwise

                //Counts for this batch:
                int truePositiveCount = isTruePositive.sumNumber().intValue();
                int falsePositiveCount = isFalsePositive.sumNumber().intValue();

                //Increment counts for this threshold
                ROC.CountsForThreshold thresholdCounts = counts.get(i).get(currThreshold);
                thresholdCounts.incrementTruePositive(truePositiveCount);
                thresholdCounts.incrementFalsePositive(falsePositiveCount);
            }
        }
    }

    /**
     * Get the ROC curve, as a set of points
     *
     * @param classIdx Index of the class to get the (one-vs-all) ROC cur
     *
     * @return ROC curve, as a list of points
     */
    public List<ROC.ROCValue> getResults(int classIdx) {
        assertHasBeenFit(classIdx);

        List<ROC.ROCValue> out = new ArrayList<>(counts.size());

        for (Map.Entry<Double, ROC.CountsForThreshold> entry : counts.get(classIdx).entrySet()) {
            double t = entry.getKey();
            ROC.CountsForThreshold c = entry.getValue();
            double tpr = c.getCountTruePositive() / ((double) countActualPositive[classIdx]);
            double fpr = c.getCountFalsePositive() / ((double) countActualNegative[classIdx]);

            out.add(new ROC.ROCValue(t, tpr, fpr));
        }

        return out;
    }

    /**
     * Get the ROC curve, as a set of (falsePositive, truePositive) points
     * <p>
     * Returns a 2d array of {falsePositive, truePositive values}.<br>
     * Size is [2][thresholdSteps], with out[0][.] being false positives, and out[1][.] being true positives
     *
     * @return ROC curve as double[][]
     */
    public double[][] getResultsAsArray(int classIdx) {
        assertHasBeenFit(classIdx);

        double[][] out = new double[2][thresholdSteps+1];
        int i = 0;
        for (Map.Entry<Double, ROC.CountsForThreshold> entry : counts.get(classIdx).entrySet()) {
            ROC.CountsForThreshold c = entry.getValue();
            double tpr = c.getCountTruePositive() / ((double) countActualPositive[classIdx]);
            double fpr = c.getCountFalsePositive() / ((double) countActualNegative[classIdx]);

            out[0][i] = fpr;
            out[1][i] = tpr;
            i++;
        }
        return out;
    }

    /**
     * Calculate the AUC - Area Under Curve<br>
     * Utilizes trapezoidal integration internally
     *
     * @return AUC
     */
    public double calculateAUC(int classIdx) {
        assertHasBeenFit(classIdx);

        //Calculate AUC using trapezoidal rule
        List<ROC.ROCValue> list = getResults(classIdx);

        //Given the points
        double auc = 0.0;
        for (int i = 0; i < list.size() - 1; i++) {
            ROC.ROCValue left = list.get(i);
            ROC.ROCValue right = list.get(i + 1);

            //y axis: TPR
            //x axis: FPR
            double deltaX = Math.abs(right.getFalsePositiveRate() - left.getFalsePositiveRate());   //Iterating in threshold order, so FPR decreases as threshold increases
            double avg = (left.getTruePositiveRate() + right.getTruePositiveRate()) / 2.0;

            auc += deltaX * avg;
        }
        return auc;
    }

    /**
     * Calculate the average (one-vs-all) AUC for all classes
     */
    public double calculateAverageAUC(){
        assertHasBeenFit(0);

        double sum = 0.0;
        for( int i=0; i<countActualPositive.length; i++ ){
            sum += calculateAUC(i);
        }

        return sum / countActualPositive.length;
    }

    public List<ROC.PrecisionRecallPoint> getPrecisionRecallCurve(int classIndex){
        //Precision: (true positive count) / (true positive count + false positive count) == true positive rate
        //Recall: (true positive count) / (true positive count + false negative count) = (TP count) / (total dataset positives)

        List<ROC.PrecisionRecallPoint> out = new ArrayList<>(counts.get(classIndex).size());

        for (Map.Entry<Double, ROC.CountsForThreshold> entry : counts.get(classIndex).entrySet()) {
            double t = entry.getKey();
            ROC.CountsForThreshold c = entry.getValue();
            long tpCount = c.getCountTruePositive();
            long fpCount = c.getCountFalsePositive();
            //For edge cases: http://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases
            //precision == 1 when FP = 0 -> no incorrect positive predictions
            //recall == 1 when no dataset positives are present (got all 0 of 0 positives)
            double precision;
            if(tpCount == 0 && fpCount == 0){
                //At this threshold: no predicted positive cases
                precision = 1.0;
            } else {
                precision = tpCount / (double)(tpCount + fpCount);
            }

            double recall;
            if(countActualPositive[classIndex] == 0){
                recall = 1.0;
            } else {
                recall = tpCount / ((double) countActualPositive[classIndex]);
            }


            out.add(new ROC.PrecisionRecallPoint(c.getThreshold(), precision, recall));
        }

        return out;
    }

    /**
     * Merge this ROCMultiClass instance with another.
     * This ROCMultiClass instance is modified, by adding the stats from the other instance.
     *
     * @param other ROCMultiClass instance to combine with this one
     */
    @Override
    public void merge(ROCMultiClass other){
        if(other.countActualPositive == null){
            //Other has no data
            return;
        } else if(countActualPositive == null){
            //This instance has no data
            this.countActualPositive = Arrays.copyOf(other.countActualPositive, other.countActualPositive.length);
            this.countActualNegative = Arrays.copyOf(other.countActualNegative, other.countActualNegative.length);
            for(Map.Entry<Integer,Map<Double, ROC.CountsForThreshold>> e : other.counts.entrySet()){
                Map<Double,ROC.CountsForThreshold> m = e.getValue();
                Map<Double,ROC.CountsForThreshold> mClone = new LinkedHashMap<>();
                for(Map.Entry<Double,ROC.CountsForThreshold> e2 : m.entrySet()){
                    mClone.put(e2.getKey(), e2.getValue().clone());
                }
                this.counts.put(e.getKey(), mClone);
            }
        } else {
            for( int i=0; i<countActualPositive.length; i++ ){
                this.countActualPositive[i] += other.countActualPositive[i];
                this.countActualNegative[i] += other.countActualNegative[i];
            }

            for(Integer i : counts.keySet()){
                Map<Double,ROC.CountsForThreshold> thisMap = counts.get(i);
                Map<Double,ROC.CountsForThreshold> otherMap = other.counts.get(i);

                for(Double d : thisMap.keySet()){
                    ROC.CountsForThreshold thisC = thisMap.get(d);
                    ROC.CountsForThreshold otherC = otherMap.get(d);
                    thisC.incrementTruePositive(otherC.getCountTruePositive());
                    thisC.incrementFalsePositive(otherC.getCountFalsePositive());
                }
            }
        }
    }


    private void assertHasBeenFit(int classIdx){
        if(countActualPositive == null){
            throw new IllegalStateException("Cannot get results: no data has been collected");
        }
        if(classIdx < 0 || classIdx >= countActualPositive.length){
            throw new IllegalArgumentException("Invalid class index (" + classIdx + "): must be in range 0 to numClasses = " + countActualPositive.length);
        }
    }
}
