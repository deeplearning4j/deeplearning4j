package org.deeplearning4j.evaluation;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.EvaluationCalibration;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.eval.curves.Histogram;
import org.deeplearning4j.eval.curves.PrecisionRecallCurve;
import org.deeplearning4j.eval.curves.ReliabilityDiagram;
import org.deeplearning4j.eval.curves.RocCurve;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.api.LengthUnit;
import org.deeplearning4j.ui.components.chart.ChartHistogram;
import org.deeplearning4j.ui.components.chart.ChartLine;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.deeplearning4j.ui.components.component.ComponentDiv;
import org.deeplearning4j.ui.components.component.style.StyleDiv;
import org.deeplearning4j.ui.components.table.ComponentTable;
import org.deeplearning4j.ui.components.table.style.StyleTable;
import org.deeplearning4j.ui.components.text.ComponentText;
import org.deeplearning4j.ui.components.text.style.StyleText;
import org.deeplearning4j.ui.standalone.StaticPageUtil;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Tools for evaluation and rendering {@link ROC} and {@link ROCMultiClass} results
 *
 * @author Alex Black
 */
public class EvaluationTools {

    private static final String ROC_TITLE = "ROC: TPR/Recall (y) vs. FPR (x)";
    private static final String PR_TITLE = "Precision (y) vs. Recall (x)";
    private static final String PR_THRESHOLD_TITLE = "Precision and Recall (y) vs. Classifier Threshold (x)";

    private static final double CHART_WIDTH_PX = 600.0;
    private static final double CHART_HEIGHT_PX = 400.0;

    private static final StyleChart CHART_STYLE = new StyleChart.Builder().width(CHART_WIDTH_PX, LengthUnit.Px)
                    .height(CHART_HEIGHT_PX, LengthUnit.Px).margin(LengthUnit.Px, 60, 60, 75, 10).strokeWidth(2.0)
                    .seriesColors(Color.BLUE, Color.LIGHT_GRAY).build();

    private static final StyleChart CHART_STYLE_PRECISION_RECALL =
                    new StyleChart.Builder().width(CHART_WIDTH_PX, LengthUnit.Px).height(CHART_HEIGHT_PX, LengthUnit.Px)
                                    .margin(LengthUnit.Px, 60, 60, 40, 10).strokeWidth(2.0)
                                    .seriesColors(Color.BLUE, Color.GREEN).build();

    private static final StyleTable TABLE_STYLE = new StyleTable.Builder().backgroundColor(Color.WHITE)
                    .headerColor(Color.LIGHT_GRAY).borderWidth(1).columnWidths(LengthUnit.Percent, 50, 50)
                    .width(400, LengthUnit.Px).height(200, LengthUnit.Px).build();

    private static final StyleDiv OUTER_DIV_STYLE = new StyleDiv.Builder().width(2 * CHART_WIDTH_PX, LengthUnit.Px)
                    .height(CHART_HEIGHT_PX, LengthUnit.Px).build();

    private static final StyleDiv OUTER_DIV_STYLE_WIDTH_ONLY =
                    new StyleDiv.Builder().width(2 * CHART_WIDTH_PX, LengthUnit.Px).build();

    private static final StyleDiv INNER_DIV_STYLE = new StyleDiv.Builder().width(CHART_WIDTH_PX, LengthUnit.Px)
                    .floatValue(StyleDiv.FloatValue.left).build();

    private static final StyleDiv PAD_DIV_STYLE = new StyleDiv.Builder().width(CHART_WIDTH_PX, LengthUnit.Px)
                    .height(100, LengthUnit.Px).floatValue(StyleDiv.FloatValue.left).build();

    private static final ComponentDiv PAD_DIV = new ComponentDiv(PAD_DIV_STYLE);

    private static final StyleText HEADER_TEXT_STYLE =
                    new StyleText.Builder().color(Color.BLACK).fontSize(16).underline(true).build();

    private static final StyleDiv HEADER_DIV_STYLE =
                    new StyleDiv.Builder().width(2 * CHART_WIDTH_PX - 150, LengthUnit.Px).height(30, LengthUnit.Px)
                                    .backgroundColor(Color.LIGHT_GRAY).margin(LengthUnit.Px, 5, 5, 200, 10)
                                    .floatValue(StyleDiv.FloatValue.left).build();

    private static final StyleDiv HEADER_DIV_STYLE_1400 = new StyleDiv.Builder().width(1400 - 150, LengthUnit.Px)
                    .height(30, LengthUnit.Px).backgroundColor(Color.LIGHT_GRAY).margin(LengthUnit.Px, 5, 5, 200, 10)
                    .floatValue(StyleDiv.FloatValue.left).build();

    private static final StyleDiv HEADER_DIV_PAD_STYLE = new StyleDiv.Builder().width(2 * CHART_WIDTH_PX, LengthUnit.Px)
                    .height(150, LengthUnit.Px).backgroundColor(Color.WHITE).build();

    private static final StyleDiv HEADER_DIV_TEXT_PAD_STYLE =
                    new StyleDiv.Builder().width(120, LengthUnit.Px).height(30, LengthUnit.Px)
                                    .backgroundColor(Color.LIGHT_GRAY).floatValue(StyleDiv.FloatValue.left).build();

    private static final ComponentTable INFO_TABLE = new ComponentTable.Builder(
                    new StyleTable.Builder().backgroundColor(Color.WHITE).borderWidth(0).build())
                                    .content(new String[][] {
                                                    {"Precision", "(true positives) / (true positives + false positives)"},
                                                    {"True Positive Rate (Recall)",
                                                                    "(true positives) / (data positives)"},
                                                    {"False Positive Rate", "(false positives) / (data negatives)"}})
                                    .build();

    private EvaluationTools() {}

    /**
     * Given a {@link ROC} chart, export the ROC chart and precision vs. recall charts to a stand-alone HTML file
     * @param roc  ROC to export
     * @param file File to export to
     */
    public static void exportRocChartsToHtmlFile(ROC roc, File file) throws IOException {
        String rocAsHtml = rocChartToHtml(roc);
        FileUtils.writeStringToFile(file, rocAsHtml);
    }

    /**
     * Given a {@link ROCMultiClass} chart, export the ROC chart and precision vs. recall charts to a stand-alone HTML file
     * @param roc  ROC to export
     * @param file File to export to
     */
    public static void exportRocChartsToHtmlFile(ROCMultiClass roc, File file) throws Exception {
        String rocAsHtml = rocChartToHtml(roc);
        FileUtils.writeStringToFile(file, rocAsHtml);
    }

    /**
     * Given a {@link ROC} instance, render the ROC chart and precision vs. recall charts to a stand-alone HTML file (returned as a String)
     * @param roc  ROC to render
     */
    public static String rocChartToHtml(ROC roc) {
        RocCurve rocCurve = roc.getRocCurve();

        Component c = getRocFromPoints(ROC_TITLE, rocCurve, roc.getCountActualPositive(), roc.getCountActualNegative(),
                        roc.calculateAUC(), roc.calculateAUCPR());
        Component c2 = getPRCharts(PR_TITLE, PR_THRESHOLD_TITLE, roc.getPrecisionRecallCurve());

        return StaticPageUtil.renderHTML(c, c2);
    }

    /**
     * Given a {@link ROCMultiClass} instance, render the ROC chart and precision vs. recall charts to a stand-alone HTML file (returned as a String)
     * @param rocMultiClass  ROC to render
     */
    public static String rocChartToHtml(ROCMultiClass rocMultiClass) {
        return rocChartToHtml(rocMultiClass, null);
    }

    /**
     * Given a {@link ROCMultiClass} instance and (optionally) names for each class, render the ROC chart to a stand-alone
     * HTML file (returned as a String)
     * @param rocMultiClass  ROC to render
     * @param classNames     Names of the classes. May be null
     */
    public static String rocChartToHtml(ROCMultiClass rocMultiClass, List<String> classNames) {

        int n = rocMultiClass.getNumClasses();

        List<Component> components = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            RocCurve roc = rocMultiClass.getRocCurve(i);
            String headerText = "Class " + i;
            if (classNames != null && classNames.size() > i) {
                headerText += " (" + classNames.get(i) + ")";
            }
            headerText += " vs. All";;

            Component headerDivPad = new ComponentDiv(HEADER_DIV_PAD_STYLE);
            components.add(headerDivPad);

            Component headerDivLeft = new ComponentDiv(HEADER_DIV_TEXT_PAD_STYLE);
            Component headerDiv = new ComponentDiv(HEADER_DIV_STYLE, new ComponentText(headerText, HEADER_TEXT_STYLE));
            Component c = getRocFromPoints(ROC_TITLE, roc, rocMultiClass.getCountActualPositive(i),
                            rocMultiClass.getCountActualNegative(i), rocMultiClass.calculateAUC(i),
                            rocMultiClass.calculateAUCPR(i));
            Component c2 = getPRCharts(PR_TITLE, PR_THRESHOLD_TITLE, rocMultiClass.getPrecisionRecallCurve(i));
            components.add(headerDivLeft);
            components.add(headerDiv);
            components.add(c);
            components.add(c2);
        }

        return StaticPageUtil.renderHTML(components);
    }

    /**
     * Given a {@link EvaluationCalibration} instance, export the charts to a stand-alone HTML file
     * @param ec  EvaluationCalibration instance to export HTML charts for
     * @param file File to export to
     */
    public static void exportevaluationCalibrationToHtmlFile(EvaluationCalibration ec, File file) throws IOException {
        String asHtml = evaluationCalibrationToHtml(ec);
        FileUtils.writeStringToFile(file, asHtml);
    }

    public static String evaluationCalibrationToHtml(EvaluationCalibration ec) {

        List<Component> components = new ArrayList<>();
        int nClasses = ec.numClasses();

        //Distribution of class labels + distribution of predicted classes
        Component headerDiv = new ComponentDiv(HEADER_DIV_STYLE_1400,
                        new ComponentText(
                                        "Labels and Network Prediction Class Distributions (X: Class Index. Y: Count)",
                                        HEADER_TEXT_STYLE));
        components.add(headerDiv);
        int[] labelCounts = ec.getLabelCountsEachClass();
        int[] predictedCounts = ec.getPredictionCountsEachClass();
        ChartHistogram.Builder chbLabels = new ChartHistogram.Builder("Label Class Distribution", CHART_STYLE);
        ChartHistogram.Builder chbPredictions = new ChartHistogram.Builder("Predicted Class Distribution", CHART_STYLE);
        for (int i = 0; i < nClasses; i++) {
            double lower = i - 0.5;
            double upper = i + 0.5;
            chbLabels.addBin(lower, upper, labelCounts[i]);
            chbPredictions.addBin(lower, upper, predictedCounts[i]);
        }

        ChartHistogram chL = chbLabels.build();
        ChartHistogram chP = chbPredictions.build();
        components.add(new ComponentDiv(OUTER_DIV_STYLE_WIDTH_ONLY, chL, chP));

        //Reliability diagram, for each class
        headerDiv = new ComponentDiv(HEADER_DIV_STYLE_1400, new ComponentText(
                        "Reliability Diagrams (X: Mean Predicted Value. Y: Fraction Positives)", HEADER_TEXT_STYLE));
        components.add(headerDiv);
        List<Component> sectionDiv = new ArrayList<>();
        double[] zeroOne = new double[] {0.0, 1.0};
        for (int i = 0; i < nClasses; i++) {
            ReliabilityDiagram rd = ec.getReliabilityDiagram(i);

            double[] x = rd.getMeanPredictedValueX();
            double[] y = rd.getFractionPositivesY();
            String title = rd.getTitle();

            ChartLine cl = new ChartLine.Builder(title, CHART_STYLE).addSeries("Classifier", x, y)
                            .addSeries("Ideal Classifier", zeroOne, zeroOne).build();

            sectionDiv.add(cl);
        }
        components.add(new ComponentDiv(OUTER_DIV_STYLE_WIDTH_ONLY, sectionDiv));

        //Residual plots
        headerDiv = new ComponentDiv(HEADER_DIV_STYLE_1400, new ComponentText(
                        "Network Predictions - Residual Plots - |Label(i) - P(class(i))|", HEADER_TEXT_STYLE));
        components.add(headerDiv);

        sectionDiv = new ArrayList<>();
        Histogram resPlotAll = ec.getResidualPlotAllClasses();
        sectionDiv.add(getHistogram(resPlotAll));
        for (int i = 0; i < nClasses; i++) {
            Histogram resPlotCurrent = ec.getResidualPlot(i);
            sectionDiv.add(getHistogram(resPlotCurrent));
        }
        components.add(new ComponentDiv(OUTER_DIV_STYLE_WIDTH_ONLY, sectionDiv));


        //Histogram of probabilities, overall and for each class
        headerDiv = new ComponentDiv(HEADER_DIV_STYLE_1400, new ComponentText(
                        "Network Prediction Probabilities (X: P(class). Y: Count)", HEADER_TEXT_STYLE));
        components.add(headerDiv);
        sectionDiv = new ArrayList<>();
        Histogram allProbs = ec.getProbabilityHistogramAllClasses();
        sectionDiv.add(getHistogram(allProbs));

        for (int i = 0; i < nClasses; i++) {
            Histogram classProbs = ec.getProbabilityHistogram(i);
            sectionDiv.add(getHistogram(classProbs));
        }
        components.add(new ComponentDiv(OUTER_DIV_STYLE_WIDTH_ONLY, sectionDiv));

        return StaticPageUtil.renderHTML(components);
    }

    private static Component getRocFromPoints(String title, RocCurve roc, long positiveCount, long negativeCount,
                    double auc, double aucpr) {
        double[] zeroOne = new double[] {0.0, 1.0};

        ChartLine chartLine = new ChartLine.Builder(title, CHART_STYLE).setXMin(0.0).setXMax(1.0).setYMin(0.0)
                        .setYMax(1.0).addSeries("ROC", roc.getX(), roc.getY()).addSeries("", zeroOne, zeroOne).build();

        ComponentTable ct = new ComponentTable.Builder(TABLE_STYLE).header("Field", "Value")
                        .content(new String[][] {{"AUROC: Area under ROC:", String.format("%.5f", auc)},
                                        {"AUPRC: Area under P/R:", String.format("%.5f", aucpr)},
                                        {"Total Data Positive Count", String.valueOf(positiveCount)},
                                        {"Total Data Negative Count", String.valueOf(negativeCount)}})
                        .build();

        ComponentDiv divLeft = new ComponentDiv(INNER_DIV_STYLE, PAD_DIV, ct, PAD_DIV, INFO_TABLE);
        ComponentDiv divRight = new ComponentDiv(INNER_DIV_STYLE, chartLine);

        return new ComponentDiv(OUTER_DIV_STYLE, divLeft, divRight);
    }

    private static Component getPRCharts(String precisionRecallTitle, String prThresholdTitle,
                    PrecisionRecallCurve prCurve) {

        ComponentDiv divLeft =
                        new ComponentDiv(INNER_DIV_STYLE, getPrecisionRecallCurve(precisionRecallTitle, prCurve));
        ComponentDiv divRight =
                        new ComponentDiv(INNER_DIV_STYLE, getPrecisionRecallVsThreshold(prThresholdTitle, prCurve));

        return new ComponentDiv(OUTER_DIV_STYLE, divLeft, divRight);
    }

    private static Component getPrecisionRecallCurve(String title, PrecisionRecallCurve prCurve) {
        double[] recallX = prCurve.getRecall();
        double[] precisionY = prCurve.getPrecision();

        return new ChartLine.Builder(title, CHART_STYLE).setXMin(0.0).setXMax(1.0).setYMin(0.0).setYMax(1.0)
                        .addSeries("P vs R", recallX, precisionY).build();
    }

    private static Component getPrecisionRecallVsThreshold(String title, PrecisionRecallCurve prCurve) {

        double[] recallY = prCurve.getRecall();
        double[] precisionY = prCurve.getPrecision();
        double[] thresholdX = prCurve.getThreshold();

        return new ChartLine.Builder(title, CHART_STYLE_PRECISION_RECALL).setXMin(0.0).setXMax(1.0).setYMin(0.0)
                        .setYMax(1.0).addSeries("Precision", thresholdX, precisionY)
                        .addSeries("Recall", thresholdX, recallY).showLegend(true).build();
    }

    private static Component getHistogram(Histogram histogram) {
        ChartHistogram.Builder chb = new ChartHistogram.Builder(histogram.getTitle(), CHART_STYLE);
        double[] lower = histogram.getBinLowerBounds();
        double[] upper = histogram.getBinUpperBounds();
        int[] counts = histogram.getBinCounts();
        for (int i = 0; i < counts.length; i++) {
            chb.addBin(lower[i], upper[i], counts[i]);
        }

        return chb.build();
    }
}
