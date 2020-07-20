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

package org.nd4j.evaluation.regression;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.val;
import org.nd4j.evaluation.BaseEvaluation;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.IMetric;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.same.ASum;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Triple;
import org.nd4j.serde.jackson.shaded.NDArrayTextDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArrayTextSerializer;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Evaluation method for the evaluation of regression algorithms.<br>
 * Provides the following metrics, for each column:<br>
 * - MSE: mean squared error<br>
 * - MAE: mean absolute error<br>
 * - RMSE: root mean squared error<br>
 * - RSE: relative squared error<br>
 * - PC: pearson correlation coefficient<br>
 * - R^2: coefficient of determination<br>
 * <br>
 * See for example: <a href="http://www.saedsayad.com/model_evaluation_r.htm">http://www.saedsayad.com/model_evaluation_r.htm</a>
 * For classification, see {@link Evaluation}
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class RegressionEvaluation extends BaseEvaluation<RegressionEvaluation> {

    public enum Metric implements IMetric { MSE, MAE, RMSE, RSE, PC, R2;

        @Override
        public Class<? extends IEvaluation> getEvaluationClass() {
            return RegressionEvaluation.class;
        }

        /**
         * @return True if the metric should be minimized, or false if the metric should be maximized.
         * For example, MSE of 0 is best, but R^2 of 1.0 is best
         */
        @Override
        public boolean minimize(){
            if(this == R2 || this == PC){
                return false;
            }
            return true;
        }
    }

    public static final int DEFAULT_PRECISION = 5;

    @EqualsAndHashCode.Exclude      //Exclude axis: otherwise 2 Evaluation instances could contain identical stats and fail equality
    protected int axis = 1;
    private boolean initialized;
    private List<String> columnNames;
    private long precision;
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray exampleCountPerColumn; //Necessary to account for per-output masking
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray labelsSumPerColumn; //sum(actual) per column -> used to calculate mean
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray sumSquaredErrorsPerColumn; //(predicted - actual)^2
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray sumAbsErrorsPerColumn; //abs(predicted-actial)
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray currentMean;
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray currentPredictionMean;
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray sumOfProducts;
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray sumSquaredLabels;
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray sumSquaredPredicted;
    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private INDArray sumLabels;

    protected RegressionEvaluation(int axis, List<String> columnNames, long precision){
        this.axis = axis;
        this.columnNames = columnNames;
        this.precision = precision;
    }

    public RegressionEvaluation() {
        this(null, DEFAULT_PRECISION);
    }

    /** Create a regression evaluation object with the specified number of columns, and default precision
     * for the stats() method.
     * @param nColumns Number of columns
     */
    public RegressionEvaluation(long nColumns) {
        this(createDefaultColumnNames(nColumns), DEFAULT_PRECISION);
    }

    /** Create a regression evaluation object with the specified number of columns, and specified precision
     * for the stats() method.
     * @param nColumns Number of columns
     */
    public RegressionEvaluation(long nColumns, long precision) {
        this(createDefaultColumnNames(nColumns), precision);
    }

    /** Create a regression evaluation object with default precision for the stats() method
     * @param columnNames Names of the columns
     */
    public RegressionEvaluation(String... columnNames) {
        this(columnNames == null || columnNames.length == 0 ? null : Arrays.asList(columnNames), DEFAULT_PRECISION);
    }

    /** Create a regression evaluation object with default precision for the stats() method
     * @param columnNames Names of the columns
     */
    public RegressionEvaluation(List<String> columnNames) {
        this(columnNames, DEFAULT_PRECISION);
    }

    /** Create a regression evaluation object with specified precision for the stats() method
     * @param columnNames Names of the columns
     */
    public RegressionEvaluation(List<String> columnNames, long precision) {
        this.precision = precision;

        if (columnNames == null || columnNames.isEmpty()) {
            initialized = false;
        } else {
            this.columnNames = columnNames;
            initialize(columnNames.size());
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
    public void reset() {
        initialized = false;
    }

    private void initialize(int n) {
        if (columnNames == null || columnNames.size() != n) {
            columnNames = createDefaultColumnNames(n);
        }
        exampleCountPerColumn = Nd4j.zeros(DataType.DOUBLE, n);
        labelsSumPerColumn = Nd4j.zeros(DataType.DOUBLE, n);
        sumSquaredErrorsPerColumn = Nd4j.zeros(DataType.DOUBLE, n);
        sumAbsErrorsPerColumn = Nd4j.zeros(DataType.DOUBLE, n);
        currentMean = Nd4j.zeros(DataType.DOUBLE, n);

        currentPredictionMean = Nd4j.zeros(DataType.DOUBLE, n);
        sumOfProducts = Nd4j.zeros(DataType.DOUBLE, n);
        sumSquaredLabels = Nd4j.zeros(DataType.DOUBLE, n);
        sumSquaredPredicted = Nd4j.zeros(DataType.DOUBLE, n);
        sumLabels = Nd4j.zeros(DataType.DOUBLE, n);

        initialized = true;
    }

    private static List<String> createDefaultColumnNames(long nColumns) {
        List<String> list = new ArrayList<>((int) nColumns);
        for (int i = 0; i < nColumns; i++)
            list.add("col_" + i);
        return list;
    }

    @Override
    public void eval(INDArray labels, INDArray predictions) {
        eval(labels, predictions, (INDArray) null);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray, List<? extends Serializable> recordMetaData) {
        eval(labels, networkPredictions, maskArray);
    }

    @Override
    public void eval(INDArray labelsArr, INDArray predictionsArr, INDArray maskArr) {
        Triple<INDArray,INDArray, INDArray> p = BaseEvaluation.reshapeAndExtractNotMasked(labelsArr, predictionsArr, maskArr, axis);
        INDArray labels = p.getFirst();
        INDArray predictions = p.getSecond();
        INDArray maskArray = p.getThird();

        if(labels.dataType() != predictions.dataType())
            labels = labels.castTo(predictions.dataType());

        if (!initialized) {
            initialize((int) labels.size(1));
        }
        //References for the calculations is this section:
        //https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        //https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample
        //Doing online calculation of means, sum of squares, etc.

        if (columnNames.size() != labels.size(1) || columnNames.size() != predictions.size(1)) {
            throw new IllegalArgumentException(
                            "Number of the columns of labels and predictions must match specification ("
                                            + columnNames.size() + "). Got " + labels.size(1) + " and "
                                            + predictions.size(1));
        }

        if (maskArray != null) {
            //Handle per-output masking. We are assuming *binary* masks here
            labels = labels.mul(maskArray);
            predictions = predictions.mul(maskArray);
        }

        labelsSumPerColumn.addi(labels.sum(0).castTo(labelsSumPerColumn.dataType()));

        INDArray error = predictions.sub(labels);
        INDArray absErrorSum = Nd4j.getExecutioner().exec(new ASum(error, 0));
        INDArray squaredErrorSum = error.mul(error).sum(0);

        sumAbsErrorsPerColumn.addi(absErrorSum.castTo(labelsSumPerColumn.dataType()));
        sumSquaredErrorsPerColumn.addi(squaredErrorSum.castTo(labelsSumPerColumn.dataType()));

        sumOfProducts.addi(labels.mul(predictions).sum(0).castTo(labelsSumPerColumn.dataType()));

        sumSquaredLabels.addi(labels.mul(labels).sum(0).castTo(labelsSumPerColumn.dataType()));
        sumSquaredPredicted.addi(predictions.mul(predictions).sum(0).castTo(labelsSumPerColumn.dataType()));


        val nRows = labels.size(0);

        INDArray newExampleCountPerColumn;
        if (maskArray == null) {
            newExampleCountPerColumn = exampleCountPerColumn.add(nRows);
        } else {
            newExampleCountPerColumn = exampleCountPerColumn.add(maskArray.sum(0).castTo(labelsSumPerColumn.dataType()));
        }
        currentMean.muliRowVector(exampleCountPerColumn).addi(labels.sum(0).castTo(labelsSumPerColumn.dataType())).diviRowVector(newExampleCountPerColumn);
        currentPredictionMean.muliRowVector(exampleCountPerColumn).addi(predictions.sum(0).castTo(labelsSumPerColumn.dataType()))
                        .divi(newExampleCountPerColumn);
        exampleCountPerColumn = newExampleCountPerColumn;

        sumLabels.addi(labels.sum(0).castTo(labelsSumPerColumn.dataType()));
    }

    @Override
    public void merge(RegressionEvaluation other) {

        if (other.labelsSumPerColumn == null) {
            //Other RegressionEvaluation is empty -> no op
            return;

        } else if (labelsSumPerColumn == null) {
            //This RegressionEvaluation is empty -> just copy over from the other one...
            this.columnNames = other.columnNames;
            this.precision = other.precision;
            this.exampleCountPerColumn = other.exampleCountPerColumn;
            this.labelsSumPerColumn = other.labelsSumPerColumn.dup();
            this.sumSquaredErrorsPerColumn = other.sumSquaredErrorsPerColumn.dup();
            this.sumAbsErrorsPerColumn = other.sumAbsErrorsPerColumn.dup();
            this.currentMean = other.currentMean.dup();
            this.currentPredictionMean = other.currentPredictionMean.dup();
            this.sumOfProducts = other.sumOfProducts.dup();
            this.sumSquaredLabels = other.sumSquaredLabels.dup();
            this.sumSquaredPredicted = other.sumSquaredPredicted.dup();

            return;
        }

        this.labelsSumPerColumn.addi(other.labelsSumPerColumn);
        this.sumSquaredErrorsPerColumn.addi(other.sumSquaredErrorsPerColumn);
        this.sumAbsErrorsPerColumn.addi(other.sumAbsErrorsPerColumn);
        this.currentMean.muliRowVector(exampleCountPerColumn)
                        .addi(other.currentMean.mulRowVector(other.exampleCountPerColumn))
                        .diviRowVector(exampleCountPerColumn.add(other.exampleCountPerColumn));
        this.currentPredictionMean.muliRowVector(exampleCountPerColumn)
                        .addi(other.currentPredictionMean.mulRowVector(other.exampleCountPerColumn))
                        .diviRowVector(exampleCountPerColumn.add(other.exampleCountPerColumn));
        this.sumOfProducts.addi(other.sumOfProducts);
        this.sumSquaredLabels.addi(other.sumSquaredLabels);
        this.sumSquaredPredicted.addi(other.sumSquaredPredicted);

        this.exampleCountPerColumn.addi(other.exampleCountPerColumn);
    }

    public String stats() {
        if (!initialized) {
            return "RegressionEvaluation: No Data";
        } else {

            if (columnNames == null)
                columnNames = createDefaultColumnNames(numColumns());
            int maxLabelLength = 0;
            for (String s : columnNames)
                maxLabelLength = Math.max(maxLabelLength, s.length());

            int labelWidth = maxLabelLength + 5;
            long columnWidth = precision + 10;

            String resultFormat = "%-" + labelWidth + "s" +
                "%-" + columnWidth + "." + precision + "e" + //MSE
                "%-" + columnWidth + "." + precision + "e" + //MAE
                "%-" + columnWidth + "." + precision + "e" + //RMSE
                "%-" + columnWidth + "." + precision + "e" + //RSE
                "%-" + columnWidth + "." + precision + "e" + //PC
                "%-" + columnWidth + "." + precision + "e";  //R2

            //Print header:
            StringBuilder sb = new StringBuilder();
            String headerFormat = "%-" + labelWidth + "s" +
                "%-" + columnWidth + "s" + // MSE
                "%-" + columnWidth + "s" + // MAE
                "%-" + columnWidth + "s" + // RMSE
                "%-" + columnWidth + "s" + // RSE
                "%-" + columnWidth + "s" + // PC
                "%-" + columnWidth + "s";  // R2

            sb.append(String.format(headerFormat, "Column", "MSE", "MAE", "RMSE", "RSE", "PC", "R^2"));
            sb.append("\n");

            //Print results for each column:
            for (int i = 0; i < columnNames.size(); i++) {
                String name = columnNames.get(i);
                double mse = meanSquaredError(i);
                double mae = meanAbsoluteError(i);
                double rmse = rootMeanSquaredError(i);
                double rse = relativeSquaredError(i);
                double corr = pearsonCorrelation(i);
                double r2 = rSquared(i);

                sb.append(String.format(resultFormat, name, mse, mae, rmse, rse, corr, r2));
                sb.append("\n");
            }

            return sb.toString();
        }
    }

    public int numColumns() {
        if (columnNames == null) {
            if (exampleCountPerColumn == null) {
                return 0;
            }
            return (int) exampleCountPerColumn.size(1);
        }
        return columnNames.size();
    }

    public double meanSquaredError(int column) {
        //mse per column: 1/n * sum((predicted-actual)^2)
        return sumSquaredErrorsPerColumn.getDouble(column) / exampleCountPerColumn.getDouble(column);
    }

    public double meanAbsoluteError(int column) {
        //mse per column: 1/n * |predicted-actual|
        return sumAbsErrorsPerColumn.getDouble(column) / exampleCountPerColumn.getDouble(column);
    }

    public double rootMeanSquaredError(int column) {
        //rmse per column: sqrt(1/n * sum((predicted-actual)^2)
        return Math.sqrt(sumSquaredErrorsPerColumn.getDouble(column) / exampleCountPerColumn.getDouble(column));
    }

    /**
     * Legacy method for the correlation score.
     *
     * @param column Column to evaluate
     * @return Pearson Correlation for the given column
     * @see {@link #pearsonCorrelation(int)}
     * @deprecated Use {@link #pearsonCorrelation(int)} instead.
     * For the R2 score use {@link #rSquared(int)}.
     */
    @Deprecated
    public double correlationR2(int column) {
        return pearsonCorrelation(column);
    }

    /**
     * Pearson Correlation Coefficient for samples
     *
     * @param column Column to evaluate
     * @return Pearson Correlation Coefficient for column with index {@code column}
     * @see <a href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample">Wikipedia</a>
     */
    public double pearsonCorrelation(int column) {
        double sumxiyi = sumOfProducts.getDouble(column);
        double predictionMean = currentPredictionMean.getDouble(column);
        double labelMean = currentMean.getDouble(column);

        double sumSquaredLabels = this.sumSquaredLabels.getDouble(column);
        double sumSquaredPredicted = this.sumSquaredPredicted.getDouble(column);

        double exampleCount = exampleCountPerColumn.getDouble(column);
        double r = sumxiyi - exampleCount * predictionMean * labelMean;
        r /= Math.sqrt(sumSquaredLabels - exampleCount * labelMean * labelMean)
            * Math.sqrt(sumSquaredPredicted - exampleCount * predictionMean * predictionMean);

        return r;
    }

    /**
     * Coefficient of Determination (R^2 Score)
     *
     * @param column Column to evaluate
     * @return R^2 score for column with index {@code column}
     * @see <a href="https://en.wikipedia.org/wiki/Coefficient_of_determination">Wikipedia</a>
     */
    public double rSquared(int column) {
        //ss_tot = sum_i (label_i - mean(labels))^2
        //       = (sum_i label_i^2) + mean(labels) * (n * mean(labels) - 2 * sum_i label_i)
        double sumLabelSquared = sumSquaredLabels.getDouble(column);
        double meanLabel = currentMean.getDouble(column);
        double sumLabel = sumLabels.getDouble(column);
        double n = exampleCountPerColumn.getDouble(column);
        double sstot = sumLabelSquared + meanLabel * (n * meanLabel - 2 * sumLabel);
        double ssres = sumSquaredErrorsPerColumn.getDouble(column);
        return (sstot - ssres) / sstot;
    }

    public double relativeSquaredError(int column) {
        // RSE: sum(predicted-actual)^2 / sum(actual-labelsMean)^2
        // (sum(predicted^2) - 2 * sum(predicted * actual) + sum(actual ^ 2)) / (sum(actual ^ 2) - n * actualMean)
        double numerator = sumSquaredPredicted.getDouble(column) - 2 * sumOfProducts.getDouble(column)
                        + sumSquaredLabels.getDouble(column);
        double denominator = sumSquaredLabels.getDouble(column) - exampleCountPerColumn.getDouble(column)
                        * currentMean.getDouble(column) * currentMean.getDouble(column);

        if (Math.abs(denominator) > Nd4j.EPS_THRESHOLD) {
            return numerator / denominator;
        } else {
            return Double.POSITIVE_INFINITY;
        }
    }


    /**
     * Average MSE across all columns
     * @return
     */
    public double averageMeanSquaredError() {
        double ret = 0.0;
        for (int i = 0; i < numColumns(); i++) {
            ret += meanSquaredError(i);
        }

        return ret / (double) numColumns();
    }

    /**
     * Average MAE across all columns
     * @return
     */
    public double averageMeanAbsoluteError() {
        double ret = 0.0;
        for (int i = 0; i < numColumns(); i++) {
            ret += meanAbsoluteError(i);
        }

        return ret / (double) numColumns();
    }

    /**
     * Average RMSE across all columns
     * @return
     */
    public double averagerootMeanSquaredError() {
        double ret = 0.0;
        for (int i = 0; i < numColumns(); i++) {
            ret += rootMeanSquaredError(i);
        }

        return ret / (double) numColumns();
    }


    /**
     * Average RSE across all columns
     * @return
     */
    public double averagerelativeSquaredError() {
        double ret = 0.0;
        for (int i = 0; i < numColumns(); i++) {
            ret += relativeSquaredError(i);
        }

        return ret / (double) numColumns();
    }


    /**
     * Legacy method for the correlation average across all columns.
     *
     * @return Pearson Correlation averaged over all columns
     * @see {@link #averagePearsonCorrelation()}
     * @deprecated Use {@link #averagePearsonCorrelation()} instead.
     * For the R2 score use {@link #averageRSquared()}.
     */
    @Deprecated
    public double averagecorrelationR2() {
        return averagePearsonCorrelation();
    }

    /**
     * Average Pearson Correlation Coefficient across all columns
     *
     * @return Pearson Correlation Coefficient across all columns
     */
    public double averagePearsonCorrelation() {
        double ret = 0.0;
        for (int i = 0; i < numColumns(); i++) {
            ret += pearsonCorrelation(i);
        }

        return ret / (double) numColumns();
    }

    /**
     * Average R2 across all columns
     *
     * @return R2 score accross all columns
     */
    public double averageRSquared() {
        double ret = 0.0;
        for (int i = 0; i < numColumns(); i++) {
            ret += rSquared(i);
        }

        return ret / (double) numColumns();
    }

    @Override
    public double getValue(IMetric metric){
        if(metric instanceof Metric){
            return scoreForMetric((Metric) metric);
        } else
            throw new IllegalStateException("Can't get value for non-regression Metric " + metric);
    }

    public double scoreForMetric(Metric metric){
        switch (metric){
            case MSE:
                return averageMeanSquaredError();
            case MAE:
                return averageMeanAbsoluteError();
            case RMSE:
                return averagerootMeanSquaredError();
            case RSE:
                return averagerelativeSquaredError();
            case PC:
                return averagePearsonCorrelation();
            case R2:
                return averageRSquared();
            default:
                throw new IllegalStateException("Unknown metric: " + metric);
        }
    }

    public static RegressionEvaluation fromJson(String json){
        return fromJson(json, RegressionEvaluation.class);
    }

    @Override
    public RegressionEvaluation newInstance() {
        return new RegressionEvaluation(axis, columnNames, precision);
    }
}
