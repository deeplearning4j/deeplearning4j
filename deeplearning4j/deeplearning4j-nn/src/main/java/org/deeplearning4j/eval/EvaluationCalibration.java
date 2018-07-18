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

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.val;
import org.deeplearning4j.eval.curves.Histogram;
import org.deeplearning4j.eval.curves.ReliabilityDiagram;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;
import org.nd4j.shade.serde.jackson.shaded.NDArrayDeSerializer;
import org.nd4j.shade.serde.jackson.shaded.NDArraySerializer;

/**
 * EvaluationCalibration is an evaluation class designed to analyze the calibration of a classifier.<br>
 * It provides a number of tools for this purpose:
 * - Counts of the number of labels and predictions for each class<br>
 * - Reliability diagram (or reliability curve)<br>
 * - Residual plot (histogram)<br>
 * - Histograms of probabilities, including probabilities for each class separately<br>
 *<br>
 * References:<br>
 * - Reliability diagram: see for example Niculescu-Mizil and Caruana 2005, Predicting Good Probabilities With
 * Supervised Learning<br>
 * - Residual plot: see Wallace and Dahabreh 2012, Class Probability Estimates are Unreliable for Imbalanced Data
 * (and How to Fix Them)<br>
 *
 *
 * @author Alex Black
 */
@Getter
@EqualsAndHashCode
public class EvaluationCalibration extends BaseEvaluation<EvaluationCalibration> {

    public static final int DEFAULT_RELIABILITY_DIAG_NUM_BINS = 10;
    public static final int DEFAULT_HISTOGRAM_NUM_BINS = 50;

    private final int reliabilityDiagNumBins;
    private final int histogramNumBins;
    private final boolean excludeEmptyBins;

    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private INDArray rDiagBinPosCount;
    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private INDArray rDiagBinTotalCount;
    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private INDArray rDiagBinSumPredictions;

    @JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    private INDArray labelCountsEachClass;
    @JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    private INDArray predictionCountsEachClass;

    @JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    private INDArray residualPlotOverall;
    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private INDArray residualPlotByLabelClass;

    @JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    private INDArray probHistogramOverall; //Simple histogram over all probabilities
    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private INDArray probHistogramByLabelClass; //Histogram - for each label class separately

    /**
     * Create an EvaluationCalibration instance with the default number of bins
     */
    public EvaluationCalibration() {
        this(DEFAULT_RELIABILITY_DIAG_NUM_BINS, DEFAULT_HISTOGRAM_NUM_BINS, true);
    }

    /**
     * Create an EvaluationCalibration instance with the specified number of bins
     *
     * @param reliabilityDiagNumBins Number of bins for the reliability diagram (usually 10)
     * @param histogramNumBins       Number of bins for the histograms
     */
    public EvaluationCalibration(int reliabilityDiagNumBins, int histogramNumBins) {
        this(reliabilityDiagNumBins, histogramNumBins, true);
    }

    /**
     * Create an EvaluationCalibration instance with the specified number of bins
     *
     * @param reliabilityDiagNumBins Number of bins for the reliability diagram (usually 10)
     * @param histogramNumBins       Number of bins for the histograms
     * @param excludeEmptyBins       For the reliability diagram,  whether empty bins should be excluded
     */
    public EvaluationCalibration(@JsonProperty("reliabilityDiagNumBins") int reliabilityDiagNumBins,
                    @JsonProperty("histogramNumBins") int histogramNumBins,
                    @JsonProperty("excludeEmptyBins") boolean excludeEmptyBins) {
        this.reliabilityDiagNumBins = reliabilityDiagNumBins;
        this.histogramNumBins = histogramNumBins;
        this.excludeEmptyBins = excludeEmptyBins;
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray) {

        if (labels.rank() == 3) {
            evalTimeSeries(labels, networkPredictions, maskArray);
            return;
        }

        //Stats for the reliability diagram: one reliability diagram for each class
        // For each bin, we need: (a) the number of positive cases AND total cases, (b) the average probability

        val nClasses = labels.size(1);

        if (rDiagBinPosCount == null) {
            //Initialize
            rDiagBinPosCount = Nd4j.create(reliabilityDiagNumBins, nClasses);
            rDiagBinTotalCount = Nd4j.create(reliabilityDiagNumBins, nClasses);
            rDiagBinSumPredictions = Nd4j.create(reliabilityDiagNumBins, nClasses);

            labelCountsEachClass = Nd4j.create(1, nClasses);
            predictionCountsEachClass = Nd4j.create(1, nClasses);

            residualPlotOverall = Nd4j.create(1, histogramNumBins);
            residualPlotByLabelClass = Nd4j.create(histogramNumBins, nClasses);

            probHistogramOverall = Nd4j.create(1, histogramNumBins);
            probHistogramByLabelClass = Nd4j.create(histogramNumBins, nClasses);
        }


        //First: loop over classes, determine positive count and total count - for each bin
        double histogramBinSize = 1.0 / histogramNumBins;
        double reliabilityBinSize = 1.0 / reliabilityDiagNumBins;

        INDArray p = networkPredictions;
        INDArray l = labels;

        if (maskArray != null) {
            //2 options: per-output masking, or
            if (maskArray.isColumnVectorOrScalar()) {
                //Per-example masking
                l = l.mulColumnVector(maskArray);
            } else {
                l = l.mul(maskArray);
            }
        }

        for (int j = 0; j < reliabilityDiagNumBins; j++) {
            INDArray geqBinLower = p.gte(j * reliabilityBinSize);
            INDArray ltBinUpper;
            if (j == reliabilityDiagNumBins - 1) {
                //Handle edge case
                ltBinUpper = p.lte(1.0);
            } else {
                ltBinUpper = p.lt((j + 1) * reliabilityBinSize);
            }

            //Calculate bit-mask over each entry - whether that entry is in the current bin or not
            INDArray currBinBitMask = geqBinLower.muli(ltBinUpper);
            if (maskArray != null) {
                if (maskArray.isColumnVectorOrScalar()) {
                    currBinBitMask.muliColumnVector(maskArray);
                } else {
                    currBinBitMask.muli(maskArray);
                }
            }

            INDArray isPosLabelForBin = l.mul(currBinBitMask);
            INDArray maskedProbs = networkPredictions.mul(currBinBitMask);

            INDArray numPredictionsCurrBin = currBinBitMask.sum(0);

            rDiagBinSumPredictions.getRow(j).addi(maskedProbs.sum(0));
            rDiagBinPosCount.getRow(j).addi(isPosLabelForBin.sum(0));
            rDiagBinTotalCount.getRow(j).addi(numPredictionsCurrBin);
        }


        //Second, we want histograms of:
        //(a) Distribution of label classes: label counts for each class
        //(b) Distribution of prediction classes: prediction counts for each class
        //(c) residual plots, for each class - (i) all instances, (ii) positive instances only, (iii) negative only
        //(d) Histograms of probabilities, for each class

        labelCountsEachClass.addi(labels.sum(0));
        //For prediction counts: do an IsMax op, but we need to take masking into account...
        INDArray isPredictedClass = Nd4j.getExecutioner().execAndReturn(new IsMax(p.dup(), 1));
        if (maskArray != null) {
            LossUtil.applyMask(isPredictedClass, maskArray);
        }
        predictionCountsEachClass.addi(isPredictedClass.sum(0));



        //Residual plots: want histogram of |labels - predicted prob|

        //ND4J's histogram op: dynamically calculates the bin positions, which is not what I want here...
        INDArray labelsSubPredicted = labels.sub(networkPredictions);
        INDArray maskedProbs = networkPredictions.dup();
        Transforms.abs(labelsSubPredicted, false);

        //if masking: replace entries with < 0 to effectively remove them
        if (maskArray != null) {
            //Assume per-example masking
            INDArray newMask = maskArray.mul(-10);
            labelsSubPredicted.addiColumnVector(newMask);
            maskedProbs.addiColumnVector(newMask);
        }

        INDArray notLabels = Transforms.not(labels);
        for (int j = 0; j < histogramNumBins; j++) {
            INDArray geqBinLower = labelsSubPredicted.gte(j * histogramBinSize);
            INDArray ltBinUpper;
            INDArray geqBinLowerProbs = maskedProbs.gte(j * histogramBinSize);
            INDArray ltBinUpperProbs;
            if (j == histogramNumBins - 1) {
                //Handle edge case
                ltBinUpper = labelsSubPredicted.lte(1.0);
                ltBinUpperProbs = maskedProbs.lte(1.0);
            } else {
                ltBinUpper = labelsSubPredicted.lt((j + 1) * histogramBinSize);
                ltBinUpperProbs = maskedProbs.lt((j + 1) * histogramBinSize);
            }

            INDArray currBinBitMask = geqBinLower.muli(ltBinUpper);
            INDArray currBinBitMaskProbs = geqBinLowerProbs.muli(ltBinUpperProbs);

            int newTotalCount = residualPlotOverall.getInt(0, j) + currBinBitMask.sumNumber().intValue();
            residualPlotOverall.putScalar(0, j, newTotalCount);

            //Counts for positive class only: values are in the current bin AND it's a positive label
            INDArray isPosLabelForBin = l.mul(currBinBitMask);

            residualPlotByLabelClass.getRow(j).addi(isPosLabelForBin.sum(0));

            int probNewTotalCount = probHistogramOverall.getInt(0, j) + currBinBitMaskProbs.sumNumber().intValue();
            probHistogramOverall.putScalar(0, j, probNewTotalCount);

            INDArray isPosLabelForBinProbs = l.mul(currBinBitMaskProbs);
            INDArray temp = isPosLabelForBinProbs.sum(0);
            probHistogramByLabelClass.getRow(j).addi(temp);
        }
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions) {
        eval(labels, networkPredictions, (INDArray) null);
    }

    @Override
    public void merge(EvaluationCalibration other) {
        if (reliabilityDiagNumBins != other.reliabilityDiagNumBins) {
            throw new UnsupportedOperationException(
                            "Cannot merge EvaluationCalibration instances with different numbers of bins");
        }

        if (other.rDiagBinPosCount == null) {
            return;
        }

        if (rDiagBinPosCount == null) {
            this.rDiagBinPosCount = other.rDiagBinPosCount;
            this.rDiagBinTotalCount = other.rDiagBinTotalCount;
            this.rDiagBinSumPredictions = other.rDiagBinSumPredictions;
        }

        this.rDiagBinPosCount.addi(other.rDiagBinPosCount);
        this.rDiagBinTotalCount.addi(other.rDiagBinTotalCount);
        this.rDiagBinSumPredictions.addi(other.rDiagBinSumPredictions);
    }

    @Override
    public void reset() {
        rDiagBinPosCount = null;
        rDiagBinTotalCount = null;
        rDiagBinSumPredictions = null;
    }

    @Override
    public String stats() {
        return "EvaluationCalibration(nBins=" + reliabilityDiagNumBins + ")";
    }

    public int numClasses() {
        if (rDiagBinTotalCount == null) {
            return -1;
        }

        // FIXME: int cast
        return (int) rDiagBinTotalCount.size(1);
    }

    /**
     * Get the reliability diagram for the specified class
     *
     * @param classIdx Index of the class to get the reliability diagram for
     */
    public ReliabilityDiagram getReliabilityDiagram(int classIdx) {

        INDArray totalCountBins = rDiagBinTotalCount.getColumn(classIdx);
        INDArray countPositiveBins = rDiagBinPosCount.getColumn(classIdx);

        double[] meanPredictionBins = rDiagBinSumPredictions.getColumn(classIdx).div(totalCountBins).data().asDouble();

        double[] fracPositives = countPositiveBins.div(totalCountBins).data().asDouble();

        if (excludeEmptyBins) {
            MatchCondition condition = new MatchCondition(totalCountBins, Conditions.equals(0));
            int numZeroBins = Nd4j.getExecutioner().exec(condition, Integer.MAX_VALUE).getInt(0);
            if (numZeroBins != 0) {
                double[] mpb = meanPredictionBins;
                double[] fp = fracPositives;

                // FIXME: int cast
                meanPredictionBins = new double[(int) (totalCountBins.length() - numZeroBins)];
                fracPositives = new double[meanPredictionBins.length];
                int j = 0;
                for (int i = 0; i < mpb.length; i++) {
                    if (totalCountBins.getDouble(i) != 0) {
                        meanPredictionBins[j] = mpb[i];
                        fracPositives[j] = fp[i];
                        j++;
                    }
                }
            }
        }
        String title = "Reliability Diagram: Class " + classIdx;
        return new ReliabilityDiagram(title, meanPredictionBins, fracPositives);
    }

    /**
     * @return The number of observed labels for each class. For N classes, be returned array is of length N, with
     * out[i] being the number of labels of class i
     */
    public int[] getLabelCountsEachClass() {
        return labelCountsEachClass == null ? null : labelCountsEachClass.data().asInt();
    }

    /**
     * @return The number of network predictions for each class. For N classes, be returned array is of length N, with
     * out[i] being the number of predicted values (max probability) for class i
     */
    public int[] getPredictionCountsEachClass() {
        return predictionCountsEachClass == null ? null : predictionCountsEachClass.data().asInt();
    }

    /**
     * Get the residual plot for all classes combined. The residual plot is defined as a histogram of<br>
     * |label_i - prob(class_i | input)| for all classes i and examples.<br>
     * In general, small residuals indicate a superior classifier to large residuals.
     *
     * @return Residual plot (histogram) - all predictions/classes
     */
    public Histogram getResidualPlotAllClasses() {
        String title = "Residual Plot - All Predictions and Classes";
        int[] counts = residualPlotOverall.data().asInt();
        return new Histogram(title, 0.0, 1.0, counts);
    }

    /**
     * Get the residual plot, only for examples of the specified class.. The residual plot is defined as a histogram of<br>
     * |label_i - prob(class_i | input)| for all and examples; for this particular method, only predictions where
     * i == labelClassIdx are included.<br>
     * In general, small residuals indicate a superior classifier to large residuals.
     *
     * @param labelClassIdx Index of the class to get the residual plot for
     * @return Residual plot (histogram) - all predictions/classes
     */
    public Histogram getResidualPlot(int labelClassIdx) {
        String title = "Residual Plot - Predictions for Label Class " + labelClassIdx;
        int[] counts = residualPlotByLabelClass.getColumn(labelClassIdx).dup().data().asInt();
        return new Histogram(title, 0.0, 1.0, counts);
    }

    /**
     * Return a probability histogram for all predictions/classes.
     *
     * @return Probability histogram
     */
    public Histogram getProbabilityHistogramAllClasses() {
        String title = "Network Probabilities Histogram - All Predictions and Classes";
        int[] counts = probHistogramOverall.data().asInt();
        return new Histogram(title, 0.0, 1.0, counts);
    }

    /**
     * Return a probability histogram of the specified label class index. That is, for label class index i,
     * a histogram of P(class_i | input) is returned, only for those examples that are labelled as class i.
     *
     * @param labelClassIdx Index of the label class to get the histogram for
     * @return Probability histogram
     */
    public Histogram getProbabilityHistogram(int labelClassIdx) {
        String title = "Network Probabilities Histogram - P(class " + labelClassIdx + ") - Data Labelled Class "
                        + labelClassIdx + " Only";
        int[] counts = probHistogramByLabelClass.getColumn(labelClassIdx).dup().data().asInt();
        return new Histogram(title, 0.0, 1.0, counts);
    }
}
