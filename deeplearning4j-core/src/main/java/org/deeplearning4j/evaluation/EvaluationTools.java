package org.deeplearning4j.evaluation;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.api.LengthUnit;
import org.deeplearning4j.ui.components.chart.ChartLine;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.deeplearning4j.ui.components.component.ComponentDiv;
import org.deeplearning4j.ui.components.component.style.StyleDiv;
import org.deeplearning4j.ui.components.table.ComponentTable;
import org.deeplearning4j.ui.components.table.style.StyleTable;
import org.deeplearning4j.ui.standalone.StaticPageUtil;

import java.awt.Color;
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

    private static final StyleChart CHART_STYLE = new StyleChart.Builder()
            .width(600, LengthUnit.Px)
            .height(400, LengthUnit.Px)
            .strokeWidth(2.0)
            .seriesColors(Color.BLUE, Color.LIGHT_GRAY)
            .build();

    private static final StyleTable TABLE_STYLE = new StyleTable.Builder()
            .backgroundColor(Color.WHITE)
            .headerColor(Color.LIGHT_GRAY)
            .borderWidth(1)
            .columnWidths(LengthUnit.Percent, 50,50)
            .width(400, LengthUnit.Px)
            .height(200, LengthUnit.Px)
            .build();

    private static final StyleDiv OUTER_DIV_STYLE = new StyleDiv.Builder()
            .width(1000.0, LengthUnit.Px)
            .build();

    private static final StyleDiv INNER_DIV_STYLE = new StyleDiv.Builder()
            .width(500.0, LengthUnit.Px)
            .floatValue(StyleDiv.FloatValue.left)
            .build();

    private static final StyleDiv PAD_DIV_STYLE = new StyleDiv.Builder()
            .width(500.0, LengthUnit.Px)
            .height(100, LengthUnit.Px)
            .floatValue(StyleDiv.FloatValue.left)
            .build();

    private static final ComponentDiv PAD_DIV = new ComponentDiv(PAD_DIV_STYLE);

    private static final ComponentTable AXIS_LABEL_TABLE = new ComponentTable.Builder(
            new StyleTable.Builder().backgroundColor(Color.WHITE).borderWidth(0).build())
            .content(new String[][]{
                    {"Vertical Axis", "True Positive Rate"},
                    {"Horizontial Axis", "False Positive Rate"}})
            .build();

    private EvaluationTools() { }

    /**
     * Given a {@link ROC} chart, export the ROC chart to a stand-alone HTML file
     * @param roc  ROC to export
     * @param file File to export to
     */
    public static void exportRocChartToHtmlFile(ROC roc, File file) throws IOException {
        String rocAsHtml = rocChartToHtml(roc);
        FileUtils.writeStringToFile(file, rocAsHtml);
    }

    /**
     * Given a {@link ROCMultiClass} chart, export the ROC chart to a stand-alone HTML file
     * @param roc  ROC to export
     * @param file File to export to
     */
    public static void exportRocChartToHtmlFile(ROCMultiClass roc, File file) throws Exception {
        String rocAsHtml = rocChartToHtml(roc);
        FileUtils.writeStringToFile(file, rocAsHtml);
    }

    /**
     * Given a {@link ROC} instance, render the ROC chart to a stand-alone HTML file (returned as a String)
     * @param roc  ROC to render
     */
    public static String rocChartToHtml(ROC roc) {
        double[][] points = roc.getResultsAsArray();

        Component c = getRocFromPoints("ROC", points, roc.getCountActualPositive(), roc.getCountActualNegative(), roc.calculateAUC());

        return StaticPageUtil.renderHTML(c);
    }

    /**
     * Given a {@link ROCMultiClass} instance, render the ROC chart to a stand-alone HTML file (returned as a String)
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
        long[] actualCountPositive = rocMultiClass.getCountActualPositive();
        long[] actualCountNegative = rocMultiClass.getCountActualNegative();

        List<Component> components = new ArrayList<>(actualCountPositive.length);
        for( int i=0; i<actualCountPositive.length; i++ ){
            double[][] points = rocMultiClass.getResultsAsArray(i);
            String title = "ROC - class " + i;
            if(classNames != null && classNames.size() > i){
                title += " (" + classNames.get(i) + ")";
            }
            title += " vs. all";;
            Component c = getRocFromPoints(title, points, actualCountPositive[i], actualCountNegative[i], rocMultiClass.calculateAUC(i));
            components.add(c);
        }

        return StaticPageUtil.renderHTML(components);
    }

    private static Component getRocFromPoints(String title, double[][] points, long positiveCount, long negativeCount, double auc){
        double[] zeroOne = new double[]{0.0, 1.0};

        ChartLine chartLine = new ChartLine.Builder(title, CHART_STYLE)
                .setXMin(0.0).setXMax(1.0)
                .setYMin(0.0).setYMax(1.0)
                .addSeries("ROC", points[0], points[1])     //points[0] is false positives -> usually plotted on x axis
                .addSeries("", zeroOne, zeroOne)
                .build();

        ComponentTable ct = new ComponentTable.Builder(TABLE_STYLE)
                .header("Field", "Value")
                .content(new String[][]{
                        {"AUC", String.format("%.5f", auc)},
                        {"Count Positive", String.valueOf(positiveCount)},
                        {"Count Negative", String.valueOf(negativeCount)}})
                .build();

        ComponentDiv divLeft = new ComponentDiv(INNER_DIV_STYLE, PAD_DIV, ct, PAD_DIV, AXIS_LABEL_TABLE);
        ComponentDiv divRight = new ComponentDiv(INNER_DIV_STYLE, chartLine);

        return new ComponentDiv(OUTER_DIV_STYLE, divLeft, divRight);
    }
}
