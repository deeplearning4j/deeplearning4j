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

package org.deeplearning4j.eval;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastGreaterThan;
import org.nd4j.linalg.api.ops.impl.transforms.Not;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

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
    public static final int DEFAULT_PRECISION = 4;
    public static final double DEFAULT_EDGE_VALUE = 0.0;

    //Because we want evaluation to work for large numbers of examples - and with low precision (FP16), we won't
    //use INDArrays to store the counts
    private int[] countTruePositive; //P=1, Act=1
    private int[] countFalsePositive; //P=1, Act=0
    private int[] countTrueNegative; //P=0, Act=0
    private int[] countFalseNegative; //P=0, Act=1
    private ROCBinary rocBinary;

    private List<String> labels;

    @JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    private INDArray decisionThreshold;

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

    @Override
    public void eval(INDArray labels, INDArray networkPredictions) {
        eval(labels, networkPredictions, (INDArray) null);
    }

    @Override
    public void evalTimeSeries(INDArray labels, INDArray predictions, INDArray labelsMask) {
        if (labelsMask == null || labelsMask.rank() == 2) {
            super.evalTimeSeries(labels, predictions, labelsMask);
            return;
        } else if (labelsMask.rank() != 3) {
            throw new IllegalArgumentException("Labels must: must be rank 2 or 3. Got: " + labelsMask.rank());
        }

        //Per output time series masking
        INDArray l2d = EvaluationUtils.reshapeTimeSeriesTo2d(labels);
        INDArray p2d = EvaluationUtils.reshapeTimeSeriesTo2d(predictions);
        INDArray m2d = EvaluationUtils.reshapeTimeSeriesTo2d(labelsMask);

        eval(l2d, p2d, m2d);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray) {

        if (countTruePositive != null && countTruePositive.length != labels.size(1)) {
            throw new IllegalStateException("Labels array does not match stored state size. Expected labels array with "
                            + "size " + countTruePositive.length + ", got labels array with size " + labels.size(1));
        }

        if (labels.rank() == 3) {
            evalTimeSeries(labels, networkPredictions, maskArray);
            return;
        }

        //First: binarize the network prediction probabilities, threshold 0.5 unless otherwise specified
        //This gives us 3 binary arrays: labels, predictions, masks
        INDArray classPredictions;
        if (decisionThreshold != null) {
            classPredictions = Nd4j.createUninitialized(networkPredictions.shape());
            Nd4j.getExecutioner()
                            .exec(new BroadcastGreaterThan(networkPredictions, decisionThreshold, classPredictions, 1));
        } else {
            classPredictions = networkPredictions.gt(0.5);
        }

        INDArray notLabels = Nd4j.getExecutioner().execAndReturn(new Not(labels.dup()));
        INDArray notClassPredictions = Nd4j.getExecutioner().execAndReturn(new Not(classPredictions.dup()));

        INDArray truePositives = classPredictions.mul(labels); //1s where predictions are 1, and labels are 1. 0s elsewhere
        INDArray trueNegatives = notClassPredictions.mul(notLabels); //1s where predictions are 0, and labels are 0. 0s elsewhere
        INDArray falsePositives = classPredictions.mul(notLabels); //1s where predictions are 1, labels are 0
        INDArray falseNegatives = notClassPredictions.mul(labels); //1s where predictions are 0, labels are 1

        if (maskArray != null) {
            //By multiplying by mask, we keep only those 1s that are actually present
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
            rocBinary.eval(labels, networkPredictions, maskArray);
        }
    }

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
     * Calculate the G-measure for the given output
     *
     * @param output The specified output
     * @return The G-measure for the specified output
     */
    public double gMeasure(int output) {
        double precision = precision(output);
        double recall = recall(output);
        return EvaluationUtils.gMeasure(precision, recall);
    }

    /**
     * Returns the false positive rate for a given label
     *
     * @param classLabel the label
     * @return fpr as a double
     */
    public double falsePositiveRate(int classLabel) {
        return recall(classLabel);
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

    public static EvaluationBinary fromJson(String json) {
        return fromJson(json, EvaluationBinary.class);
    }

    public static EvaluationBinary fromYaml(String yaml) {
        return fromYaml(yaml, EvaluationBinary.class);
    }


}
