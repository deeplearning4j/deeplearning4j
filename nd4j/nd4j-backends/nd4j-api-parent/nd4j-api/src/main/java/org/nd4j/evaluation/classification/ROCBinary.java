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
import lombok.val;
import org.nd4j.base.Preconditions;
import org.nd4j.evaluation.BaseEvaluation;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.IMetric;
import org.nd4j.evaluation.classification.ROCMultiClass.Metric;
import org.nd4j.evaluation.curves.PrecisionRecallCurve;
import org.nd4j.evaluation.curves.RocCurve;
import org.nd4j.evaluation.serde.ROCArraySerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * ROC (Receiver Operating Characteristic) for multi-task binary classifiers.
 * As per {@link ROC}, ROCBinary supports both exact (thersholdSteps == 0) and thresholded; see {@link ROC} for details.
 * <p>
 * Unlike {@link ROC} (which supports a single binary label (as a single column probability, or 2 column 'softmax' probability
 * distribution), ROCBinary assumes that all outputs are independent binary variables. This also differs from
 * {@link ROCMultiClass}, which should be used for multi-class (single non-binary) cases.
 * <p>
 * ROCBinary supports per-example and per-output masking: for per-output masking, any particular output may be absent
 * (mask value 0) and hence won't be included in the calculated ROC.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class ROCBinary extends BaseEvaluation<ROCBinary> {
    public static final int DEFAULT_STATS_PRECISION = 4;

    /**
     * AUROC: Area under ROC curve<br>
     * AUPRC: Area under Precision-Recall Curve
     */
    public enum Metric implements IMetric {AUROC, AUPRC;

        @Override
        public Class<? extends IEvaluation> getEvaluationClass() {
            return ROCBinary.class;
        }

        @Override
        public boolean minimize() {
            return false;
        }
    }

    @JsonSerialize(using = ROCArraySerializer.class)
    private ROC[] underlying;

    private int thresholdSteps;
    private boolean rocRemoveRedundantPts;
    private List<String> labels;

    @EqualsAndHashCode.Exclude      //Exclude axis: otherwise 2 Evaluation instances could contain identical stats and fail equality
    protected int axis = 1;

    protected ROCBinary(int axis, int thresholdSteps, boolean rocRemoveRedundantPts, List<String> labels) {
        this.thresholdSteps = thresholdSteps;
        this.rocRemoveRedundantPts = rocRemoveRedundantPts;
        this.axis = axis;
        this.labels = labels;
    }

    public ROCBinary() {
        this(0);
    }

    /**
     * @param thresholdSteps Number of threshold steps to use for the ROC calculation. Set to 0 for exact ROC calculation
     */
    public ROCBinary(int thresholdSteps) {
        this(thresholdSteps, true);
    }

    /**
     * @param thresholdSteps Number of threshold steps to use for the ROC calculation. If set to 0: use exact calculation
     * @param rocRemoveRedundantPts Usually set to true. If true,  remove any redundant points from ROC and P-R curves
     */
    public ROCBinary(int thresholdSteps, boolean rocRemoveRedundantPts) {
        this.thresholdSteps = thresholdSteps;
        this.rocRemoveRedundantPts = rocRemoveRedundantPts;
    }

    /**
     * Set the axis for evaluation - this is the dimension along which the probability (and label independent binary classes) are present.<br>
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
    public void reset() {
        underlying = null;
    }

    @Override
    public void eval(INDArray labels, INDArray predictions, INDArray mask, List<? extends Serializable> recordMetaData) {
        Triple<INDArray,INDArray, INDArray> p = BaseEvaluation.reshapeAndExtractNotMasked(labels, predictions, mask, axis);
        INDArray labels2d = p.getFirst();
        INDArray predictions2d = p.getSecond();
        INDArray maskArray = p.getThird();

        if (underlying != null && underlying.length != labels2d.size(1)) {
            throw new IllegalStateException("Labels array does not match stored state size. Expected labels array with "
                            + "size " + underlying.length + ", got labels array with size " + labels2d.size(1));
        }

        if (labels2d.rank() == 3) {
            evalTimeSeries(labels2d, predictions2d, maskArray);
            return;
        }

        if(labels2d.dataType() != predictions2d.dataType())
            labels2d = labels2d.castTo(predictions2d.dataType());

        int n = (int) labels2d.size(1);
        if (underlying == null) {
            underlying = new ROC[n];
            for (int i = 0; i < n; i++) {
                underlying[i] = new ROC(thresholdSteps, rocRemoveRedundantPts);
            }
        }

        int[] perExampleNonMaskedIdxs = null;
        for (int i = 0; i < n; i++) {
            INDArray prob = predictions2d.getColumn(i).reshape(predictions2d.size(0), 1);
            INDArray label = labels2d.getColumn(i).reshape(labels2d.size(0), 1);
            if (maskArray != null) {
                //If mask array is present, pull out the non-masked rows only
                INDArray m;
                boolean perExampleMasking = false;
                if (maskArray.isColumnVectorOrScalar()) {
                    //Per-example masking
                    m = maskArray;
                    perExampleMasking = true;
                } else {
                    //Per-output masking
                    m = maskArray.getColumn(i);
                }
                int[] rowsToPull;

                if (perExampleNonMaskedIdxs != null) {
                    //Reuse, per-example masking
                    rowsToPull = perExampleNonMaskedIdxs;
                } else {
                    int nonMaskedCount = m.sumNumber().intValue();
                    rowsToPull = new int[nonMaskedCount];
                    val maskSize = m.size(0);
                    int used = 0;
                    for (int j = 0; j < maskSize; j++) {
                        if (m.getDouble(j) != 0.0) {
                            rowsToPull[used++] = j;
                        }
                    }
                    if (perExampleMasking) {
                        perExampleNonMaskedIdxs = rowsToPull;
                    }
                }

                //TODO Temporary workaround for: https://github.com/deeplearning4j/deeplearning4j/issues/7102
                if(prob.isView())
                    prob = prob.dup();
                if(label.isView())
                    label = label.dup();

                prob = Nd4j.pullRows(prob, 1, rowsToPull); //1: tensor along dim 1
                label = Nd4j.pullRows(label, 1, rowsToPull);
            }

            underlying[i].eval(label, prob);
        }
    }

    @Override
    public void merge(ROCBinary other) {
        if (this.underlying == null) {
            this.underlying = other.underlying;
            return;
        } else if (other.underlying == null) {
            return;
        }

        //Both have data
        if (underlying.length != other.underlying.length) {
            throw new UnsupportedOperationException("Cannot merge ROCBinary: this expects " + underlying.length
                            + "outputs, other expects " + other.underlying.length + " outputs");
        }
        for (int i = 0; i < underlying.length; i++) {
            this.underlying[i].merge(other.underlying[i]);
        }
    }

    private void assertIndex(int outputNum) {
        if (underlying == null) {
            throw new UnsupportedOperationException("ROCBinary does not have any stats: eval must be called first");
        }
        if (outputNum < 0 || outputNum >= underlying.length) {
            throw new IllegalArgumentException("Invalid input: output number must be between 0 and " + (outputNum - 1));
        }
    }

    /**
     * Returns the number of labels - (i.e., size of the prediction/labels arrays) - if known. Returns -1 otherwise
     */
    public int numLabels() {
        if (underlying == null) {
            return -1;
        }

        return underlying.length;
    }

    /**
     * Get the actual positive count (accounting for any masking) for  the specified output/column
     *
     * @param outputNum Index of the output (0 to {@link #numLabels()}-1)
     */
    public long getCountActualPositive(int outputNum) {
        assertIndex(outputNum);
        return underlying[outputNum].getCountActualPositive();
    }

    /**
     * Get the actual negative count (accounting for any masking) for  the specified output/column
     *
     * @param outputNum Index of the output (0 to {@link #numLabels()}-1)
     */
    public long getCountActualNegative(int outputNum) {
        assertIndex(outputNum);
        return underlying[outputNum].getCountActualNegative();
    }

    /**
     * Get the ROC object for the specific column
     * @param outputNum Column (output number)
     * @return The underlying ROC object for this specific column
     */
    public ROC getROC(int outputNum){
        assertIndex(outputNum);
        return underlying[outputNum];
    }

    /**
     * Get the ROC curve for the specified output
     * @param outputNum Number of the output to get the ROC curve for
     * @return ROC curve
     */
    public RocCurve getRocCurve(int outputNum) {
        assertIndex(outputNum);

        return underlying[outputNum].getRocCurve();
    }

    /**
     * Get the Precision-Recall curve for the specified output
     * @param outputNum Number of the output to get the P-R curve for
     * @return  Precision recall curve
     */
    public PrecisionRecallCurve getPrecisionRecallCurve(int outputNum) {
        assertIndex(outputNum);
        return underlying[outputNum].getPrecisionRecallCurve();
    }


    /**
     * Macro-average AUC for all outcomes
     * @return the (macro-)average AUC for all outcomes.
     */
    public double calculateAverageAuc() {
        double ret = 0.0;
        for (int i = 0; i < numLabels(); i++) {
            ret += calculateAUC(i);
        }

        return ret / (double) numLabels();
    }

    /**
     * @return the (macro-)average AUPRC (area under precision recall curve)
     */
    public double calculateAverageAUCPR(){
        double ret = 0.0;
        for (int i = 0; i < numLabels(); i++) {
            ret += calculateAUCPR(i);
        }

        return ret / (double) numLabels();
    }

    /**
     * Calculate the AUC - Area Under (ROC) Curve<br>
     * Utilizes trapezoidal integration internally
     *
     * @param outputNum Output number to calculate AUC for
     * @return AUC
     */
    public double calculateAUC(int outputNum) {
        assertIndex(outputNum);
        return underlying[outputNum].calculateAUC();
    }

    /**
     * Calculate the AUCPR - Area Under Curve - Precision Recall<br>
     * Utilizes trapezoidal integration internally
     *
     * @param outputNum Output number to calculate AUCPR for
     * @return AUCPR
     */
    public double calculateAUCPR(int outputNum) {
        assertIndex(outputNum);
        return underlying[outputNum].calculateAUCPR();
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

    @Override
    public String stats() {
        return stats(DEFAULT_STATS_PRECISION);
    }

    public String stats(int printPrecision) {
        //Calculate AUC and also print counts, for each output

        StringBuilder sb = new StringBuilder();

        int maxLabelsLength = 15;
        if (labels != null) {
            for (String s : labels) {
                maxLabelsLength = Math.max(s.length(), maxLabelsLength);
            }
        }

        String patternHeader = "%-" + (maxLabelsLength + 5) + "s%-12s%-12s%-10s%-10s";
        String header = String.format(patternHeader, "Label", "AUC", "AUPRC", "# Pos", "# Neg");

        String pattern = "%-" + (maxLabelsLength + 5) + "s" //Label
                        + "%-12." + printPrecision + "f" //AUC
                        + "%-12." + printPrecision + "f" //AUPRC
                        + "%-10d%-10d"; //Count pos, count neg

        sb.append(header);

        if (underlying != null) {
            for (int i = 0; i < underlying.length; i++) {
                double auc = calculateAUC(i);
                double auprc = calculateAUCPR(i);

                String label = (labels == null ? String.valueOf(i) : labels.get(i));

                sb.append("\n").append(String.format(pattern, label, auc, auprc, getCountActualPositive(i),
                                getCountActualNegative(i)));
            }

            if(thresholdSteps > 0){
                sb.append("\n");
                sb.append("[Note: Thresholded AUC/AUPRC calculation used with ").append(thresholdSteps)
                        .append(" steps); accuracy may reduced compared to exact mode]");
            }

        } else {
            //Empty evaluation
            sb.append("\n-- No Data --\n");
        }

        return sb.toString();
    }

    public static ROCBinary fromJson(String json){
        return fromJson(json, ROCBinary.class);
    }

    public double scoreForMetric(Metric metric, int idx){
        assertIndex(idx);
        switch (metric){
            case AUROC:
                return calculateAUC(idx);
            case AUPRC:
                return calculateAUCPR(idx);
            default:
                throw new IllegalStateException("Unknown metric: " + metric);
        }
    }

    @Override
    public double getValue(IMetric metric){
        if(metric instanceof Metric){
            if(metric == Metric.AUPRC)
                return calculateAverageAUCPR();
            else if(metric == Metric.AUROC)
                return calculateAverageAuc();
            else
                throw new IllegalStateException("Can't get value for non-binary ROC Metric " + metric);
        } else
            throw new IllegalStateException("Can't get value for non-binary ROC Metric " + metric);
    }

    @Override
    public ROCBinary newInstance() {
        return new ROCBinary(axis, thresholdSteps, rocRemoveRedundantPts, labels);
    }
}
