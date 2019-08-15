/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.evaluation.classification;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.nd4j.evaluation.BaseEvaluation;
import org.nd4j.evaluation.EvaluationUtils;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.IMetric;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.bool.BroadcastGreaterThan;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.serde.jackson.shaded.NDArrayTextDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArrayTextSerializer;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * EvaluationBinary: used for evaluating networks with binary classification outputs. The typical classification metrics,
 * such as accuracy, precision, recall, F1 score, etc. are calculated for each output.<br>
 * Note that {@link ROCBinary} is also used internally to calculate AUC for each output, but only when using an
 * appropriate constructor, {@link #EvaluationBinary(int, Integer)}
 * <p>
 * Note that EvaluationBinary supports both per-example and per-output masking.<br>
 * EvaluationBinary by default uses a decision threshold of 0.5, however decision thresholds can be set on a per-output
 * basis using {@link #EvaluationBinary(INDArray)}.
 * <p>
 * The most common use case: multi-task networks, where each output is a binary value. This differs from {@link Evaluation}
 * in that {@link Evaluation} is for a single class (binary or non-binary) evaluation.
 *
 * @author Alex Black
 */
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
@Data
public class EvaluationBinary extends BaseEvaluation<EvaluationBinary> {

    public enum Metric implements IMetric {ACCURACY, F1, PRECISION, RECALL, GMEASURE, MCC, FAR;

        @Override
        public Class<? extends IEvaluation> getEvaluationClass() {
            return EvaluationBinary.class;
        }

        @Override
        public boolean minimize() {
            return false;
        }
    }

    public static final int DEFAULT_PRECISION = 4;
    public static final double DEFAULT_EDGE_VALUE = 0.0;

    @EqualsAndHashCode.Exclude      //Exclude axis: otherwise 2 Evaluation instances could contain identical stats and fail equality
    protected int axis = 1;

    //Because we want evaluation to work for large numbers of examples - and with low precision (FP16), we won't
    //use INDArrays to store the counts
    private int[] countTruePositive; //P=1, Act=1
    private int[] countFalsePositive; //P=1, Act=0
    private int[] countTrueNegative; //P=0, Act=0
    private int[] countFalseNegative; //P=0, Act=1
    private ROCBinary rocBinary;

    private List<String> labels;

    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray decisionThreshold;

    protected EvaluationBinary(int axis, ROCBinary rocBinary, List<String> labels, INDArray decisionThreshold){
        this.axis = axis;
        this.rocBinary = rocBinary;
        this.labels = labels;
        this.decisionThreshold = decisionThreshold;
    }

    /**
     * Create an EvaulationBinary instance with an optional decision threshold array.
     *
     * @param decisionThreshold Decision threshold for each output; may be null. Should be a row vector with length
     *                          equal to the number of outputs, with values in range 0 to 1. An array of 0.5 values is
     *                          equivalent to the default (no manually specified decision threshold).
     */
    public EvaluationBinary(INDArray decisionThreshold) {
        if (decisionThreshold != null) {
            if (!decisionThreshold.isRowVectorOrScalar()) {
                throw new IllegalArgumentException(
                                "Decision threshold array must be a row vector; got array with shape "
                                                + Arrays.toString(decisionThreshold.shape()));
            }
            if (decisionThreshold.minNumber().doubleValue() < 0.0) {
                throw new IllegalArgumentException("Invalid decision threshold array: minimum value is less than 0");
            }
            if (decisionThreshold.maxNumber().doubleValue() > 1.0) {
                throw new IllegalArgumentException(
                                "invalid decision threshold array: maximum value is greater than 1.0");
            }

            this.decisionThreshold = decisionThreshold;
        }
    }

    /**
     * This constructor allows for ROC to be calculated in addition to the standard evaluation metrics, when the
     * rocBinarySteps arg is non-null. See {@link ROCBinary} for more details
     *
     * @param size           Number of outputs
     * @param rocBinarySteps Constructor arg for {@link ROCBinary#ROCBinary(int)}
     */
    public EvaluationBinary(int size, Integer rocBinarySteps) {
        countTruePositive = new int[size];
        countFalsePositive = new int[size];
        countTrueNegative = new int[size];
        countFalseNegative = new int[size];
        if (rocBinarySteps != null) {
            rocBinary = new ROCBinary(rocBinarySteps);
        }
    }

    /**
     * Set the axis for evaluation - this is the dimension along which the probability (and label classes) are present.<br>
     * For DL4J, this can be left as the default setting (axis = 1).<br>
     * Axis should be set as follows:<br>
     * For 2D (OutputLayer), shape [minibatch, numClasses] - axis = 1<br>
     * For 3D, RNNs/CNN1D (DL4J RnnOutputLayer), NCW format, shape [minibatch, numClasses, sequenceLength] - axis = 1<br>
     * For 3D, RNNs/CNN1D (DL4J RnnOutputLayer), NWC format, shape [minibatch, sequenceLength, numClasses] - axis = 2<br>
     * For 4D, CNN2D (DL4J CnnLossLayer), NCHW format, shape [minibatch, channels, height, width] - axis = 1<br>
     * For 4D, CNN2D, NHWC format, shape [minibatch, height, width, channels] - axis = 3<br>
     *
     * @param axis Axis to use for evaluation
     */
    public void setAxis(int axis){
        this.axis = axis;
    }

    /**
     * Get the axis - see {@link #setAxis(int)} for details
     */
    public int getAxis(){
        return axis;
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions) {
        eval(labels, networkPredictions, (INDArray) null);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray, List<? extends Serializable> recordMetaData) {
        if(recordMetaData != null){
            throw new UnsupportedOperationException("Evaluation with record metadata not yet implemented for EvaluationBinary");
        }
        eval(labels, networkPredictions, maskArray);
    }

    @Override
    public void eval(INDArray labelsArr, INDArray predictionsArr, INDArray maskArr) {

        //Check for NaNs in predictions - without this, evaulation could silently be intepreted as class 0 prediction due to argmax
        long count = Nd4j.getExecutioner().execAndReturn(new MatchCondition(predictionsArr, Conditions.isNan())).getFinalResult().longValue();
        org.nd4j.base.Preconditions.checkState(count == 0, "Cannot perform evaluation with NaNs present in predictions:" +
                " %s NaNs present in predictions INDArray", count);

        if (countTruePositive != null && countTruePositive.length != labelsArr.size(axis)) {
            throw new IllegalStateException("Labels array does not match stored state size. Expected labels array with "
                            + "size " + countTruePositive.length + ", got labels array with size " + labelsArr.size(axis) + " for axis " + axis);
        }

        Triple<INDArray,INDArray, INDArray> p = BaseEvaluation.reshapeAndExtractNotMasked(labelsArr, predictionsArr, maskArr, axis);
        INDArray labels = p.getFirst();
        INDArray predictions = p.getSecond();
        INDArray maskArray = p.getThird();

        if(labels.dataType() != predictions.dataType())
            labels = labels.castTo(predictions.dataType());

        if(decisionThreshold != null && decisionThreshold.dataType() != predictions.dataType())
            decisionThreshold = decisionThreshold.castTo(predictions.dataType());

        //First: binarize the network prediction probabilities, threshold 0.5 unless otherwise specified
        //This gives us 3 binary arrays: labels, predictions, masks
        INDArray classPredictions;
        if (decisionThreshold != null) {
            classPredictions = Nd4j.createUninitialized(DataType.BOOL, predictions.shape());
            Nd4j.getExecutioner()
                            .exec(new BroadcastGreaterThan(predictions, decisionThreshold, classPredictions, 1));
        } else {
            classPredictions = predictions.gt(0.5);
        }
        classPredictions = classPredictions.castTo(predictions.dataType());

        INDArray notLabels = labels.rsub(1.0);  //If labels are 0 or 1, then rsub(1) swaps
        INDArray notClassPredictions = classPredictions.rsub(1.0);

        INDArray truePositives = classPredictions.mul(labels); //1s where predictions are 1, and labels are 1. 0s elsewhere
        INDArray trueNegatives = notClassPredictions.mul(notLabels); //1s where predictions are 0, and labels are 0. 0s elsewhere
        INDArray falsePositives = classPredictions.mul(notLabels); //1s where predictions are 1, labels are 0
        INDArray falseNegatives = notClassPredictions.mul(labels); //1s where predictions are 0, labels are 1

        if (maskArray != null) {
            //By multiplying by mask, we keep only those 1s that are actually present
            maskArray = maskArray.castTo(truePositives.dataType());
            truePositives.muli(maskArray);
            trueNegatives.muli(maskArray);
            falsePositives.muli(maskArray);
            falseNegatives.muli(maskArray);
        }

        int[] tpCount = truePositives.sum(0).data().asInt();
        int[] tnCount = trueNegatives.sum(0).data().asInt();
        int[] fpCount = falsePositives.sum(0).data().asInt();
        int[] fnCount = falseNegatives.sum(0).data().asInt();

        if (countTruePositive == null) {
            int l = tpCount.length;
            countTruePositive = new int[l];
            countFalsePositive = new int[l];
            countTrueNegative = new int[l];
            countFalseNegative = new int[l];
        }

        addInPlace(countTruePositive, tpCount);
        addInPlace(countFalsePositive, fpCount);
        addInPlace(countTrueNegative, tnCount);
        addInPlace(countFalseNegative, fnCount);

        if (rocBinary != null) {
            rocBinary.eval(labels, predictions, maskArray);
        }
    }

    /**
     * Merge the other evaluation object into this one. The result is that this {@link #EvaluationBinary}  instance contains the counts
     * etc from both
     *
     * @param other EvaluationBinary object to merge into this one.
     */
    @Override
    public void merge(EvaluationBinary other) {
        if (other.countTruePositive == null) {
            //Other is empty - no op
            return;
        }

        if (countTruePositive == null) {
            //This evaluation is empty -> take results from other
            this.countTruePositive = other.countTruePositive;
            this.countFalsePositive = other.countFalsePositive;
            this.countTrueNegative = other.countTrueNegative;
            this.countFalseNegative = other.countFalseNegative;
            this.rocBinary = other.rocBinary;
        } else {
            if (this.countTruePositive.length != other.countTruePositive.length) {
                throw new IllegalStateException("Cannot merge EvaluationBinary instances with different sizes. This "
                                + "size: " + this.countTruePositive.length + ", other size: "
                                + other.countTruePositive.length);
            }

            //Both have stats
            addInPlace(this.countTruePositive, other.countTruePositive);
            addInPlace(this.countTrueNegative, other.countTrueNegative);
            addInPlace(this.countFalsePositive, other.countFalsePositive);
            addInPlace(this.countFalseNegative, other.countFalseNegative);

            if (this.rocBinary != null) {
                this.rocBinary.merge(other.rocBinary);
            }
        }
    }

    @Override
    public void reset() {
        countTruePositive = null;
    }

    private static void addInPlace(int[] addTo, int[] toAdd) {
        for (int i = 0; i < addTo.length; i++) {
            addTo[i] += toAdd[i];
        }
    }

    /**
     * Returns the number of labels - (i.e., size of the prediction/labels arrays) - if known. Returns -1 otherwise
     */
    public int numLabels() {
        if (countTruePositive == null) {
            return -1;
        }

        return countTruePositive.length;
    }

    /**
     * Set the label names, for printing via {@link #stats()}
     */
    public void setLabelNames(List<String> labels) {
        if (labels == null) {
            this.labels = null;
            return;
        }
        this.labels = new ArrayList<>(labels);
    }

    /**
     * Get the total number of values for the specified column, accounting for any masking
     */
    public int totalCount(int outputNum) {
        assertIndex(outputNum);
        return countTruePositive[outputNum] + countTrueNegative[outputNum] + countFalseNegative[outputNum]
                        + countFalsePositive[outputNum];
    }



    /**
     * Get the true positives count for the specified output
     */
    public int truePositives(int outputNum) {
        assertIndex(outputNum);
        return countTruePositive[outputNum];
    }

    /**
     * Get the true negatives count for the specified output
     */
    public int trueNegatives(int outputNum) {
        assertIndex(outputNum);
        return countTrueNegative[outputNum];
    }

    /**
     * Get the false positives count for the specified output
     */
    public int falsePositives(int outputNum) {
        assertIndex(outputNum);
        return countFalsePositive[outputNum];
    }

    /**
     * Get the false negatives count for the specified output
     */
    public int falseNegatives(int outputNum) {
        assertIndex(outputNum);
        return countFalseNegative[outputNum];
    }

    public double averageAccuracy() {
        double ret = 0.0;
        for (int i = 0; i < numLabels(); i++) {
            ret += accuracy(i);
        }

        ret /= (double) numLabels();
        return ret;
    }

    /**
     * Get the accuracy for the specified output
     */
    public double accuracy(int outputNum) {
        assertIndex(outputNum);
        return (countTruePositive[outputNum] + countTrueNegative[outputNum]) / (double) totalCount(outputNum);
    }

    public double averagePrecision() {
        double ret = 0.0;
        for (int i = 0; i < numLabels(); i++) {
            ret += precision(i);
        }

        ret /= (double) numLabels();
        return ret;
    }

    /**
     * Get the precision (tp / (tp + fp)) for the specified output
     */
    public double precision(int outputNum) {
        assertIndex(outputNum);
        //double precision = tp / (double) (tp + fp);
        return countTruePositive[outputNum] / (double) (countTruePositive[outputNum] + countFalsePositive[outputNum]);
    }


    public double averageRecall() {
        double ret = 0.0;
        for (int i = 0; i < numLabels(); i++) {
            ret += recall(i);
        }

        ret /= (double) numLabels();
        return ret;
    }

    /**
     * Get the recall (tp / (tp + fn)) for the specified output
     */
    public double recall(int outputNum) {
        assertIndex(outputNum);
        return countTruePositive[outputNum] / (double) (countTruePositive[outputNum] + countFalseNegative[outputNum]);
    }


    public double averageF1() {
        double ret = 0.0;
        for (int i = 0; i < numLabels(); i++) {
            ret += f1(i);
        }

        ret /= (double) numLabels();
        return ret;
    }

    /**
     * Calculate the F-beta value for the given output
     *
     * @param beta      Beta value to use
     * @param outputNum Output number
     * @return F-beta for the given output
     */
    public double fBeta(double beta, int outputNum) {
        assertIndex(outputNum);
        double precision = precision(outputNum);
        double recall = recall(outputNum);
        return EvaluationUtils.fBeta(beta, precision, recall);
    }

    /**
     * Get the F1 score for the specified output
     */
    public double f1(int outputNum) {
        return fBeta(1.0, outputNum);
    }

    /**
     * Calculate the Matthews correlation coefficient for the specified output
     *
     * @param outputNum Output number
     * @return Matthews correlation coefficient
     */
    public double matthewsCorrelation(int outputNum) {
        assertIndex(outputNum);

        return EvaluationUtils.matthewsCorrelation(truePositives(outputNum), falsePositives(outputNum),
                        falseNegatives(outputNum), trueNegatives(outputNum));
    }

    /**
     * Macro average of the Matthews correlation coefficient (MCC) (see {@link #matthewsCorrelation(int)}) for all labels.
     *
     * @return The macro average of the MCC for all labels.
     */
    public double averageMatthewsCorrelation() {
        double ret = 0.0;
        for (int i = 0; i < numLabels(); i++) {
            ret += matthewsCorrelation(i);
        }

        ret /= (double) numLabels();
        return ret;
    }

    /**
     * Calculate the macro average G-measure for the given output
     *
     * @param output The specified output
     * @return The macro average of the G-measure for the specified output
     */
    public double gMeasure(int output) {
        double precision = precision(output);
        double recall = recall(output);
        return EvaluationUtils.gMeasure(precision, recall);
    }

    /**
     * Average G-measure (see {@link #gMeasure(int)}) for all labels.
     *
     * @return The G-measure for all labels.
     */
    public double averageGMeasure() {
        double ret = 0.0;
        for (int i = 0; i < numLabels(); i++) {
            ret += gMeasure(i);
        }

        ret /= (double) numLabels();
        return ret;
    }

    /**
     * Returns the false positive rate for a given label
     *
     * @param classLabel the label
     * @return fpr as a double
     */
    public double falsePositiveRate(int classLabel) {
        assertIndex(classLabel);
        return falsePositiveRate(classLabel, DEFAULT_EDGE_VALUE);
    }

    /**
     * Returns the false positive rate for a given label
     *
     * @param classLabel the label
     * @param edgeCase   What to output in case of 0/0
     * @return fpr as a double
     */
    public double falsePositiveRate(int classLabel, double edgeCase) {
        double fpCount = falsePositives(classLabel);
        double tnCount = trueNegatives(classLabel);

        return EvaluationUtils.falsePositiveRate((long) fpCount, (long) tnCount, edgeCase);
    }

    /**
     * Returns the false negative rate for a given label
     *
     * @param classLabel the label
     * @return fnr as a double
     */
    public double falseNegativeRate(Integer classLabel) {
        return falseNegativeRate(classLabel, DEFAULT_EDGE_VALUE);
    }

    /**
     * Returns the false negative rate for a given label
     *
     * @param classLabel the label
     * @param edgeCase   What to output in case of 0/0
     * @return fnr as a double
     */
    public double falseNegativeRate(Integer classLabel, double edgeCase) {
        double fnCount = falseNegatives(classLabel);
        double tpCount = truePositives(classLabel);

        return EvaluationUtils.falseNegativeRate((long) fnCount, (long) tpCount, edgeCase);
    }

    /**
     * Returns the {@link ROCBinary} instance, if present
     */
    public ROCBinary getROCBinary() {
        return rocBinary;
    }

    private void assertIndex(int outputNum) {
        if (countTruePositive == null) {
            throw new UnsupportedOperationException(
                            "EvaluationBinary does not have any stats: eval must be called first");
        }
        if (outputNum < 0 || outputNum >= countTruePositive.length) {
            throw new IllegalArgumentException("Invalid input: output number must be between 0 and " + (outputNum - 1)
                            + ". Got index: " + outputNum);
        }
    }

    /**
     * Average False Alarm Rate (FAR) (see {@link #falseAlarmRate(int)}) for all labels.
     *
     * @return The FAR for all labels.
     */
    public double averageFalseAlarmRate() {
        double ret = 0.0;
        for (int i = 0; i < numLabels(); i++) {
            ret += falseAlarmRate(i);
        }

        ret /= (double) numLabels();
        return ret;
    }

    /**
     * False Alarm Rate (FAR) reflects rate of misclassified to classified records
     * <a href="http://ro.ecu.edu.au/cgi/viewcontent.cgi?article=1058&context=isw">http://ro.ecu.edu.au/cgi/viewcontent.cgi?article=1058&context=isw</a><br>
     *
     * @param outputNum Class index to calculate False Alarm Rate (FAR)
     * @return The FAR for the outcomes
     */
    public double falseAlarmRate(int outputNum) {
        assertIndex(outputNum);

        return (falsePositiveRate(outputNum) + falseNegativeRate(outputNum)) / 2.0;
    }

    /**
     * Get a String representation of the EvaluationBinary class, using the default precision
     */
    public String stats() {
        return stats(DEFAULT_PRECISION);
    }

    /**
     * Get a String representation of the EvaluationBinary class, using the specified precision
     *
     * @param printPrecision The precision (number of decimal places) for the accuracy, f1, etc.
     */
    public String stats(int printPrecision) {

        StringBuilder sb = new StringBuilder();

        //Report: Accuracy, precision, recall, F1. Then: confusion matrix

        int maxLabelsLength = 15;
        if (labels != null) {
            for (String s : labels) {
                maxLabelsLength = Math.max(s.length(), maxLabelsLength);
            }
        }

        String subPattern = "%-12." + printPrecision + "f";
        String pattern = "%-" + (maxLabelsLength + 5) + "s" //Label
                        + subPattern + subPattern + subPattern + subPattern //Accuracy, f1, precision, recall
                        + "%-8d%-7d%-7d%-7d%-7d"; //Total count, TP, TN, FP, FN

        String patternHeader = "%-" + (maxLabelsLength + 5) + "s%-12s%-12s%-12s%-12s%-8s%-7s%-7s%-7s%-7s";



        List<String> headerNames = Arrays.asList("Label", "Accuracy", "F1", "Precision", "Recall", "Total", "TP", "TN",
                        "FP", "FN");

        if (rocBinary != null) {
            patternHeader += "%-12s";
            pattern += subPattern;

            headerNames = new ArrayList<>(headerNames);
            headerNames.add("AUC");
        }

        String header = String.format(patternHeader, headerNames.toArray());


        sb.append(header);

        if (countTrueNegative != null) {

            for (int i = 0; i < countTrueNegative.length; i++) {
                int totalCount = totalCount(i);

                double acc = accuracy(i);
                double f1 = f1(i);
                double precision = precision(i);
                double recall = recall(i);

                String label = (labels == null ? String.valueOf(i) : labels.get(i));

                List<Object> args = Arrays.<Object>asList(label, acc, f1, precision, recall, totalCount,
                                truePositives(i), trueNegatives(i), falsePositives(i), falseNegatives(i));
                if (rocBinary != null) {
                    args = new ArrayList<>(args);
                    args.add(rocBinary.calculateAUC(i));
                }

                sb.append("\n").append(String.format(pattern, args.toArray()));
            }

            if (decisionThreshold != null) {
                sb.append("\nPer-output decision thresholds: ")
                                .append(Arrays.toString(decisionThreshold.dup().data().asFloat()));
            }
        } else {
            //Empty evaluation
            sb.append("\n-- No Data --\n");
        }

        return sb.toString();
    }

    /**
     * Calculate specific metric (see {@link Metric}) for a given label.
     *
     * @param metric The Metric to calculate.
     * @param outputNum Class index to calculate.
     *
     * @return Calculated metric.
     */
    public double scoreForMetric(Metric metric, int outputNum){
        switch (metric){
            case ACCURACY:
                return accuracy(outputNum);
            case F1:
                return f1(outputNum);
            case PRECISION:
                return precision(outputNum);
            case RECALL:
                return recall(outputNum);
            case GMEASURE:
                return gMeasure(outputNum);
            case MCC:
                return matthewsCorrelation(outputNum);
            case FAR:
                return falseAlarmRate(outputNum);
            default:
                throw new IllegalStateException("Unknown metric: " + metric);
        }
    }

    public static EvaluationBinary fromJson(String json) {
        return fromJson(json, EvaluationBinary.class);
    }

    public static EvaluationBinary fromYaml(String yaml) {
        return fromYaml(yaml, EvaluationBinary.class);
    }

    @Override
    public double getValue(IMetric metric){
        if(metric instanceof Metric){
            switch ((Metric) metric){
                case ACCURACY:
                    return averageAccuracy();
                case F1:
                    return averageF1();
                case PRECISION:
                    return averagePrecision();
                case RECALL:
                    return averageRecall();
                case GMEASURE:
                    return averageGMeasure();
                case MCC:
                    return averageMatthewsCorrelation();
                case FAR:
                    return averageFalseAlarmRate();
                default:
                    throw new IllegalStateException("Can't get value for non-binary evaluation Metric " + metric);
            }
        } else
            throw new IllegalStateException("Can't get value for non-binary evaluation Metric " + metric);
    }

    @Override
    public EvaluationBinary newInstance() {
        if(rocBinary != null) {
            return new EvaluationBinary(axis, rocBinary.newInstance(), labels, decisionThreshold);
        } else {
            return new EvaluationBinary(axis, null, labels, decisionThreshold);
        }
    }
}
