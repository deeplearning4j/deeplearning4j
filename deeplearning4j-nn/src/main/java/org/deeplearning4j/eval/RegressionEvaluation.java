package org.deeplearning4j.eval;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Abs;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
 * - correlation coefficient<br>
 * See for example: http://www.saedsayad.com/model_evaluation_r.htm
 * For classification, see {@link Evaluation}
 *
 * @author Alex Black
 */
public class RegressionEvaluation extends BaseEvaluation<RegressionEvaluation> {

    public static final int DEFAULT_PRECISION = 5;

    private List<String> columnNames;
    private int precision;
    private long exampleCount = 0;
    private INDArray labelsSumPerColumn; //sum(actual) per column -> used to calculate mean
    private INDArray sumSquaredErrorsPerColumn; //(predicted - actual)^2
    private INDArray sumAbsErrorsPerColumn; //abs(predicted-actial)
    private INDArray currentMean;
    private INDArray currentPredictionMean;

    private INDArray sumOfProducts;
    private INDArray sumSquaredLabels;
    private INDArray sumSquaredPredicted;

    /** Create a regression evaluation object with the specified number of columns, and default precision
     * for the stats() method.
     * @param nColumns Number of columns
     */
    public RegressionEvaluation(int nColumns) {
        this(createDefaultColumnNames(nColumns), DEFAULT_PRECISION);
    }

    /** Create a regression evaluation object with the specified number of columns, and specified precision
     * for the stats() method.
     * @param nColumns Number of columns
     */
    public RegressionEvaluation(int nColumns, int precision) {
        this(createDefaultColumnNames(nColumns), precision);
    }

    /** Create a regression evaluation object with default precision for the stats() method
     * @param columnNames Names of the columns
     */
    public RegressionEvaluation(String... columnNames) {
        this(Arrays.asList(columnNames), DEFAULT_PRECISION);
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
    public RegressionEvaluation(List<String> columnNames, int precision) {
        if (columnNames == null || columnNames.size() == 0) {
            throw new IllegalArgumentException(
                            "Column names (or integer number of columns) must be specified (got: " + columnNames + ")");
        }
        this.columnNames = columnNames;
        this.precision = precision;

        int n = columnNames.size();
        labelsSumPerColumn = Nd4j.zeros(n);
        sumSquaredErrorsPerColumn = Nd4j.zeros(n);
        sumAbsErrorsPerColumn = Nd4j.zeros(n);
        currentMean = Nd4j.zeros(n);

        currentPredictionMean = Nd4j.zeros(n);
        sumOfProducts = Nd4j.zeros(n);
        sumSquaredLabels = Nd4j.zeros(n);
        sumSquaredPredicted = Nd4j.zeros(n);
    }

    private static List<String> createDefaultColumnNames(int nColumns) {
        List<String> list = new ArrayList<>(nColumns);
        for (int i = 0; i < nColumns; i++)
            list.add("col_" + i);
        return list;
    }

    @Override
    public void eval(INDArray labels, INDArray predictions) {
        //References for the calculations is this section:
        //https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        //https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample
        //Doing online calculation of means, sum of squares, etc.

        labelsSumPerColumn.addi(labels.sum(0));

        INDArray error = predictions.sub(labels);
        INDArray absErrorSum = Nd4j.getExecutioner().execAndReturn(new Abs(error.dup())).sum(0);
        INDArray squaredErrorSum = error.mul(error).sum(0);

        sumAbsErrorsPerColumn.addi(absErrorSum);
        sumSquaredErrorsPerColumn.addi(squaredErrorSum);

        sumOfProducts.addi(labels.mul(predictions).sum(0));

        sumSquaredLabels.addi(labels.mul(labels).sum(0));
        sumSquaredPredicted.addi(predictions.mul(predictions).sum(0));

        int nRows = labels.size(0);

        currentMean.muli(exampleCount).addi(labels.sum(0)).divi(exampleCount + nRows);
        currentPredictionMean.muli(exampleCount).addi(predictions.sum(0)).divi(exampleCount + nRows);

        exampleCount += nRows;
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
            this.exampleCount = other.exampleCount;
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
        this.currentMean.muli(exampleCount).addi(other.currentMean.mul(other.exampleCount))
                        .divi(exampleCount + other.exampleCount);
        this.currentPredictionMean.muli(exampleCount).addi(other.currentPredictionMean.mul(other.exampleCount))
                        .divi(exampleCount + other.exampleCount);
        this.sumOfProducts.addi(other.sumOfProducts);
        this.sumSquaredLabels.addi(other.sumSquaredLabels);
        this.sumSquaredPredicted.addi(other.sumSquaredPredicted);

        this.exampleCount += other.exampleCount;
    }

    public String stats() {

        int maxLabelLength = 0;
        for (String s : columnNames)
            maxLabelLength = Math.max(maxLabelLength, s.length());

        int labelWidth = maxLabelLength + 5;
        int columnWidth = precision + 10;

        String format = "%-" + labelWidth + "s" + "%-" + columnWidth + "." + precision + "e" //MSE
                        + "%-" + columnWidth + "." + precision + "e" //MAE
                        + "%-" + columnWidth + "." + precision + "e" //RMSE
                        + "%-" + columnWidth + "." + precision + "e" //RSE
                        + "%-" + columnWidth + "." + precision + "e"; //R2 (correlation coefficient)



        //Print header:
        StringBuilder sb = new StringBuilder();
        String headerFormat = "%-" + labelWidth + "s" + "%-" + columnWidth + "s" + "%-" + columnWidth + "s" + "%-"
                        + columnWidth + "s" + "%-" + columnWidth + "s" + "%-" + columnWidth + "s";
        sb.append(String.format(headerFormat, "Column", "MSE", "MAE", "RMSE", "RSE", "R^2"));
        sb.append("\n");

        //Print results for each column:
        for (int i = 0; i < columnNames.size(); i++) {
            double mse = meanSquaredError(i);
            double mae = meanAbsoluteError(i);
            double rmse = rootMeanSquaredError(i);
            double rse = relativeSquaredError(i);
            double corr = correlationR2(i);

            sb.append(String.format(format, columnNames.get(i), mse, mae, rmse, rse, corr));
            sb.append("\n");
        }


        return sb.toString();
    }

    public int numColumns() {
        return columnNames.size();
    }

    public double meanSquaredError(int column) {
        //mse per column: 1/n * sum((predicted-actual)^2)
        return sumSquaredErrorsPerColumn.getDouble(column) / exampleCount;
    }

    public double meanAbsoluteError(int column) {
        //mse per column: 1/n * |predicted-actual|
        return sumAbsErrorsPerColumn.getDouble(column) / exampleCount;
    }

    public double rootMeanSquaredError(int column) {
        //rmse per column: sqrt(1/n * sum((predicted-actual)^2)
        return Math.sqrt(sumSquaredErrorsPerColumn.getDouble(column) / exampleCount);
    }

    public double correlationR2(int column) {
        //r^2 Correlation coefficient

        double sumxiyi = sumOfProducts.getDouble(column);
        double predictionMean = this.currentPredictionMean.getDouble(column);
        double labelMean = this.currentMean.getDouble(column);

        double sumSquaredLabels = this.sumSquaredLabels.getDouble(column);
        double sumSquaredPredicted = this.sumSquaredPredicted.getDouble(column);

        double r2 = sumxiyi - exampleCount * predictionMean * labelMean;
        r2 /= Math.sqrt(sumSquaredLabels - exampleCount * labelMean * labelMean)
                        * Math.sqrt(sumSquaredPredicted - exampleCount * predictionMean * predictionMean);

        return r2;
    }

    public double relativeSquaredError(int column) {
        // RSE: sum(predicted-actual)^2 / sum(actual-labelsMean)^2
        // (sum(predicted^2) - 2 * sum(predicted * actual) + sum(actual ^ 2)) / (sum(actual ^ 2) - n * actualMean)
        double numerator = sumSquaredPredicted.getDouble(column) - 2 * sumOfProducts.getDouble(column)
                        + sumSquaredLabels.getDouble(column);
        double denominator = sumSquaredLabels.getDouble(column)
                        - exampleCount * currentMean.getDouble(column) * currentMean.getDouble(column);

        if (Math.abs(denominator) > Nd4j.EPS_THRESHOLD) {
            return numerator / denominator;
        } else {
            return Double.POSITIVE_INFINITY;
        }
    }

}
