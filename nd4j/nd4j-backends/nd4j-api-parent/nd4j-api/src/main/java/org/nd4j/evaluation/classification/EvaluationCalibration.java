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

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.val;
import org.nd4j.base.Preconditions;
import org.nd4j.evaluation.BaseEvaluation;
import org.nd4j.evaluation.curves.Histogram;
import org.nd4j.evaluation.curves.ReliabilityDiagram;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.serde.jackson.shaded.NDArrayDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArraySerializer;
import org.nd4j.serde.jackson.shaded.NDArrayTextDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArrayTextSerializer;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.io.Serializable;
import java.util.List;

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

    @EqualsAndHashCode.Exclude      //Exclude axis: otherwise 2 Evaluation instances could contain identical stats and fail equality
    protected int axis = 1;

    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private INDArray rDiagBinPosCount;
    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private INDArray rDiagBinTotalCount;
    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private INDArray rDiagBinSumPredictions;

    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray labelCountsEachClass;
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray predictionCountsEachClass;

    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray residualPlotOverall;
    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private INDArray residualPlotByLabelClass;

    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
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
    public void eval(INDArray labels, INDArray predictions, INDArray mask) {

        Triple<INDArray,INDArray, INDArray> triple = BaseEvaluation.reshapeAndExtractNotMasked(labels, predictions, mask, axis);
        if(triple == null){
            //All values masked out; no-op
            return;
        }

        INDArray labels2d = triple.getFirst();
        INDArray predictions2d = triple.getSecond();
        INDArray maskArray = triple.getThird();
        Preconditions.checkState(maskArray == null, "Per-output masking for EvaluationCalibration is not supported");

        //Stats for the reliability diagram: one reliability diagram for each class
        // For each bin, we need: (a) the number of positive cases AND total cases, (b) the average probability

        val nClasses = labels2d.size(1);

        if (rDiagBinPosCount == null) {
            DataType dt = DataType.DOUBLE;
            //Initialize
            rDiagBinPosCount = Nd4j.create(DataType.LONG, reliabilityDiagNumBins, nClasses);
            rDiagBinTotalCount = Nd4j.create(DataType.LONG, reliabilityDiagNumBins, nClasses);
            rDiagBinSumPredictions = Nd4j.create(dt, reliabilityDiagNumBins, nClasses);

            labelCountsEachClass = Nd4j.create(DataType.LONG, 1, nClasses);
            predictionCountsEachClass = Nd4j.create(DataType.LONG, 1, nClasses);

            residualPlotOverall = Nd4j.create(dt, 1, histogramNumBins);
            residualPlotByLabelClass = Nd4j.create(dt, histogramNumBins, nClasses);

            probHistogramOverall = Nd4j.create(dt, 1, histogramNumBins);
            probHistogramByLabelClass = Nd4j.create(dt, histogramNumBins, nClasses);
        }


        //First: loop over classes, determine positive count and total count - for each bin
        double histogramBinSize = 1.0 / histogramNumBins;
        double reliabilityBinSize = 1.0 / reliabilityDiagNumBins;

        INDArray p = predictions2d;
        INDArray l = labels2d;

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
            INDArray geqBinLower = p.gte(j * reliabilityBinSize).castTo(predictions2d.dataType());
            INDArray ltBinUpper;
            if (j == reliabilityDiagNumBins - 1) {
                //Handle edge case
                ltBinUpper = p.lte(1.0).castTo(predictions2d.dataType());
            } else {
                ltBinUpper = p.lt((j + 1) * reliabilityBinSize).castTo(predictions2d.dataType());
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
            INDArray maskedProbs = predictions2d.mul(currBinBitMask);

            INDArray numPredictionsCurrBin = currBinBitMask.sum(0);

            rDiagBinSumPredictions.getRow(j).addi(maskedProbs.sum(0).castTo(rDiagBinSumPredictions.dataType()));
            rDiagBinPosCount.getRow(j).addi(isPosLabelForBin.sum(0).castTo(rDiagBinPosCount.dataType()));
            rDiagBinTotalCount.getRow(j).addi(numPredictionsCurrBin.castTo(rDiagBinTotalCount.dataType()));
        }


        //Second, we want histograms of:
        //(a) Distribution of label classes: label counts for each class
        //(b) Distribution of prediction classes: prediction counts for each class
        //(c) residual plots, for each class - (i) all instances, (ii) positive instances only, (iii) negative only
        //(d) Histograms of probabilities, for each class

        labelCountsEachClass.addi(labels2d.sum(0).castTo(labelCountsEachClass.dataType()));
        //For prediction counts: do an IsMax op, but we need to take masking into account...
        INDArray isPredictedClass = Nd4j.getExecutioner().exec(new IsMax(p.dup(), 1));
        if (maskArray != null) {
            LossUtil.applyMask(isPredictedClass, maskArray);
        }
        predictionCountsEachClass.addi(isPredictedClass.sum(0).castTo(predictionCountsEachClass.dataType()));



        //Residual plots: want histogram of |labels - predicted prob|

        //ND4J's histogram op: dynamically calculates the bin positions, which is not what I want here...
        INDArray labelsSubPredicted = labels2d.sub(predictions2d);
        INDArray maskedProbs = predictions2d.dup();
        Transforms.abs(labelsSubPredicted, false);

        //if masking: replace entries with < 0 to effectively remove them
        if (maskArray != null) {
            //Assume per-example masking
            INDArray newMask = maskArray.mul(-10);
            labelsSubPredicted.addiColumnVector(newMask);
            maskedProbs.addiColumnVector(newMask);
        }

        for (int j = 0; j < histogramNumBins; j++) {
            INDArray geqBinLower = labelsSubPredicted.gte(j * histogramBinSize).castTo(predictions2d.dataType());
            INDArray ltBinUpper;
            INDArray geqBinLowerProbs = maskedProbs.gte(j * histogramBinSize).castTo(predictions2d.dataType());
            INDArray ltBinUpperProbs;
            if (j == histogramNumBins - 1) {
                //Handle edge case
                ltBinUpper = labelsSubPredicted.lte(1.0).castTo(predictions2d.dataType());
                ltBinUpperProbs = maskedProbs.lte(1.0).castTo(predictions2d.dataType());
            } else {
                ltBinUpper = labelsSubPredicted.lt((j + 1) * histogramBinSize).castTo(predictions2d.dataType());
                ltBinUpperProbs = maskedProbs.lt((j + 1) * histogramBinSize).castTo(predictions2d.dataType());
            }

            INDArray currBinBitMask = geqBinLower.muli(ltBinUpper);
            INDArray currBinBitMaskProbs = geqBinLowerProbs.muli(ltBinUpperProbs);

            int newTotalCount = residualPlotOverall.getInt(0, j) + currBinBitMask.sumNumber().intValue();
            residualPlotOverall.putScalar(0, j, newTotalCount);

            //Counts for positive class only: values are in the current bin AND it's a positive label
            INDArray isPosLabelForBin = l.mul(currBinBitMask);

            residualPlotByLabelClass.getRow(j).addi(isPosLabelForBin.sum(0).castTo(residualPlotByLabelClass.dataType()));

            int probNewTotalCount = probHistogramOverall.getInt(0, j) + currBinBitMaskProbs.sumNumber().intValue();
            probHistogramOverall.putScalar(0, j, probNewTotalCount);

            INDArray isPosLabelForBinProbs = l.mul(currBinBitMaskProbs);
            INDArray temp = isPosLabelForBinProbs.sum(0);
            probHistogramByLabelClass.getRow(j).addi(temp.castTo(probHistogramByLabelClass.dataType()));
        }
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions) {
        eval(labels, networkPredictions, (INDArray) null);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray, List<? extends Serializable> recordMetaData) {
        throw new UnsupportedOperationException("Not yet implemented");
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
        Preconditions.checkState(rDiagBinPosCount != null, "Unable to get reliability diagram: no evaluation has been performed (no data)");
        INDArray totalCountBins = rDiagBinTotalCount.getColumn(classIdx);
        INDArray countPositiveBins = rDiagBinPosCount.getColumn(classIdx);

        double[] meanPredictionBins = rDiagBinSumPredictions.getColumn(classIdx).castTo(DataType.DOUBLE)
                .div(totalCountBins.castTo(DataType.DOUBLE)).data().asDouble();

        double[] fracPositives = countPositiveBins.castTo(DataType.DOUBLE).div(totalCountBins.castTo(DataType.DOUBLE)).data().asDouble();

        if (excludeEmptyBins) {
            val condition = new MatchCondition(totalCountBins, Conditions.equals(0));
            int numZeroBins = Nd4j.getExecutioner().exec(condition).getInt(0);
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
        Preconditions.checkState(rDiagBinPosCount != null, "Unable to get residual plot: no evaluation has been performed (no data)");
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
        Preconditions.checkState(rDiagBinPosCount != null, "Unable to get probability histogram: no evaluation has been performed (no data)");
        String title = "Network Probabilities Histogram - P(class " + labelClassIdx + ") - Data Labelled Class "
                        + labelClassIdx + " Only";
        int[] counts = probHistogramByLabelClass.getColumn(labelClassIdx).dup().data().asInt();
        return new Histogram(title, 0.0, 1.0, counts);
    }

    public static EvaluationCalibration fromJson(String json){
        return fromJson(json, EvaluationCalibration.class);
    }
}
