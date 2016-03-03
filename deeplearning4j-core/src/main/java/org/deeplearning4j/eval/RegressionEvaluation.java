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
public class RegressionEvaluation {

    public static final int DEFAULT_PRECISION = 5;

    private List<String> columnNames;
    private int precision;
    private int exampleCount = 0;
    private INDArray labelsSumPerColumn;    //sum(actual) per column -> used to calculate mean
    private INDArray sumSquaredErrorsPerColumn;     //(predicted - actual)^2
    private INDArray sumAbsErrorsPerColumn;         //abs(predicted-actial)
    private INDArray currentMean;
    private INDArray currentPredictionMean;
    private INDArray m2Actual;

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
        this.columnNames = columnNames;
        this.precision = precision;

        int n = columnNames.size();
        labelsSumPerColumn = Nd4j.zeros(n);
        sumSquaredErrorsPerColumn = Nd4j.zeros(n);
        sumAbsErrorsPerColumn = Nd4j.zeros(n);
        currentMean = Nd4j.zeros(n);
        m2Actual = Nd4j.zeros(n);

        currentPredictionMean = Nd4j.zeros(n);
        sumOfProducts = Nd4j.zeros(n);
        sumSquaredLabels = Nd4j.zeros(n);
        sumSquaredPredicted = Nd4j.zeros(n);
    }

    private static List<String> createDefaultColumnNames(int nColumns) {
        List<String> list = new ArrayList<>(nColumns);
        for (int i = 0; i < nColumns; i++) list.add("col_" + i);
        return list;
    }


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
        for( int i=0; i<nRows; i++ ){
            exampleCount++;

            INDArray lRow = labels.getRow(i);
            INDArray pRow = predictions.getRow(i);

            INDArray deltaPredicted = pRow.sub(currentPredictionMean);
            INDArray deltaActual = lRow.sub(currentMean);
            currentMean.addi(deltaActual.div(exampleCount));
            currentPredictionMean.addi(deltaPredicted.div(exampleCount));
            m2Actual.addi(deltaActual.muli(lRow.sub(currentMean)));
        }
    }


    /**
     * Convenience method for evaluation of time series.
     * Reshapes time series (3d) to 2d, then calls eval
     * @see #eval(INDArray, INDArray)
     */
    public void evalTimeSeries(INDArray labels, INDArray predictions) {
        //exactly as per Evaluation.evalTimeSeries
        if(labels.rank() == 2 && predictions.rank() == 2) eval(labels,predictions);
        if(labels.rank() != 3 ) throw new IllegalArgumentException("Invalid input: labels are not rank 3 (rank="+labels.rank()+")");
        if(!Arrays.equals(labels.shape(),predictions.shape())){
            throw new IllegalArgumentException("Labels and predicted have different shapes: labels="
                    + Arrays.toString(labels.shape()) + ", predictions="+Arrays.toString(predictions.shape()));
        }

        if( labels.ordering() == 'f' ) labels = Shape.toOffsetZeroCopy(labels, 'c');
        if( predictions.ordering() == 'f' ) predictions = Shape.toOffsetZeroCopy(predictions, 'c');

        //Reshape, as per RnnToFeedForwardPreProcessor:
        int[] shape = labels.shape();
        labels = labels.permute(0,2,1);	//Permute, so we get correct order after reshaping
        labels = labels.reshape(shape[0] * shape[2], shape[1]);

        predictions = predictions.permute(0, 2, 1);
        predictions = predictions.reshape(shape[0] * shape[2], shape[1]);

        eval(labels,predictions);
    }

    /**
     * Evaluate a time series, whether the output is masked usind a masking array. That is,
     * the mask array specified whether the output at a given time step is actually present, or whether it
     * is just padding.<br>
     * For example, for N examples, nOut output size, and T time series length:
     * labels and predicted will have shape [N,nOut,T], and outputMask will have shape [N,T].
     * @see #evalTimeSeries(INDArray, INDArray)
     */
    public void evalTimeSeries(INDArray labels, INDArray predictions, INDArray outputMask) {

        int totalOutputExamples = outputMask.sumNumber().intValue();
        int outSize = labels.size(1);

        INDArray labels2d = Nd4j.create(totalOutputExamples, outSize);
        INDArray predicted2d = Nd4j.create(totalOutputExamples,outSize);

        int rowCount = 0;
        for( int ex=0; ex<outputMask.size(0); ex++ ){
            for( int t=0; t<outputMask.size(1); t++ ){
                if(outputMask.getDouble(ex,t) == 0.0) continue;

                labels2d.putRow(rowCount, labels.get(NDArrayIndex.point(ex), NDArrayIndex.all(), NDArrayIndex.point(t)));
                predicted2d.putRow(rowCount, predictions.get(NDArrayIndex.point(ex), NDArrayIndex.all(), NDArrayIndex.point(t)));

                rowCount++;
            }
        }
        eval(labels2d,predicted2d);
    }


    public String stats() {

        int maxLabelLength = 0;
        for (String s : columnNames) maxLabelLength = Math.max(maxLabelLength, s.length());

        int labelWidth = maxLabelLength + 5;
        int columnWidth = precision + 10;

        String format = "%-" + labelWidth + "s"
                + "%-" + columnWidth + "." + precision + "e"    //MSE
                + "%-" + columnWidth + "." + precision + "e"    //MAE
                + "%-" + columnWidth + "." + precision + "e"    //RMSE
                + "%-" + columnWidth + "." + precision + "e"    //RSE
                + "%-" + columnWidth + "." + precision + "e";   //R2 (correlation coefficient)




        //Print header:
        StringBuilder sb = new StringBuilder();
        String headerFormat = "%-" + labelWidth + "s"
                + "%-"+columnWidth+"s"
                + "%-"+columnWidth+"s"
                + "%-"+columnWidth+"s"
                + "%-"+columnWidth+"s"
                + "%-"+columnWidth+"s";
        sb.append(String.format(headerFormat,"Column","MSE","MAE","RMSE","RSE","R^2"));
        sb.append("\n");

        //Print results for each column:
        for (int i = 0; i < columnNames.size(); i++) {
            double mse = meanSquaredError(i);
            double mae = meanAbsoluteError(i);
            double rmse = rootMeanSquaredError(i);
            double rse = relativeSquaredError(i);
            double corr = correlationR2(i);

            sb.append(String.format(format,columnNames.get(i),mse,mae,rmse,rse,corr));
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
        return Math.sqrt(sumSquaredErrorsPerColumn.getDouble(column)/exampleCount);
    }

    public double correlationR2(int column) {
        //r^2 Correlation coefficient

        double sumxiyi = sumOfProducts.getDouble(column);
        double predictionMean = this.currentPredictionMean.getDouble(column);
        double labelMean = this.currentMean.getDouble(column);

        double sumSquaredLabels = this.sumSquaredLabels.getDouble(column);
        double sumSquaredPredicted = this.sumSquaredPredicted.getDouble(column);

        double r2 = sumxiyi - exampleCount * predictionMean * labelMean;
        r2 /= Math.sqrt(sumSquaredLabels - exampleCount*labelMean*labelMean) * Math.sqrt(sumSquaredPredicted - exampleCount * predictionMean*predictionMean);

        return r2;
    }

    public double relativeSquaredError(int column){
        //RSE: sum(predicted-actual)^2 / sum(actual-labelsMean)^2
        double m2a = m2Actual.getDouble(column);
        return sumSquaredErrorsPerColumn.getDouble(column) / m2a;
    }

}
