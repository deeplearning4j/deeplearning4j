package org.deeplearning4j.spark.stats;

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.api.stats.CommonSparkTrainingStats;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.api.LengthUnit;
import org.deeplearning4j.ui.components.chart.ChartLine;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.deeplearning4j.ui.components.component.ComponentDiv;
import org.deeplearning4j.ui.components.component.style.StyleDiv;
import org.deeplearning4j.ui.components.text.ComponentText;
import org.deeplearning4j.ui.components.text.style.StyleText;
import org.deeplearning4j.ui.standalone.StaticPageUtil;

import java.awt.*;
import java.io.IOException;
import java.util.*;
import java.util.List;

/**
 * Utility methods for Spark training stats
 *
 * @author Alex Black
 */
public class StatsUtils {

    public static void exportStats(List<EventStats> list, String outputDirectory, String filename, String delimiter, SparkContext sc) throws IOException {
        String path = FilenameUtils.concat(outputDirectory, filename);
        exportStats(list, path, delimiter, sc);
    }

    public static void exportStats(List<EventStats> list, String outputPath, String delimiter, SparkContext sc) throws IOException {
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (EventStats e : list) {
            if (first) sb.append(e.getStringHeader(delimiter)).append("\n");
            sb.append(e.asString(delimiter)).append("\n");
            first = false;
        }
        SparkUtils.writeStringToFile(outputPath, sb.toString(), sc);
    }

    public static String getDurationAsString(List<EventStats> list, String delim) {
        StringBuilder sb = new StringBuilder();
        int num = list.size();
        int count = 0;
        for (EventStats e : list) {
            sb.append(e.getDurationMs());
            if (count++ < num - 1) sb.append(delim);
        }
        return sb.toString();
    }

    public static void exportStatsAsHtml(SparkTrainingStats sparkTrainingStats, String path, JavaSparkContext sc) throws Exception {
        exportStatsAsHtml(sparkTrainingStats, path, sc.sc());
    }

    public static void exportStatsAsHtml(SparkTrainingStats sparkTrainingStats, String path, SparkContext sc) throws Exception {
        Set<String> keySet = sparkTrainingStats.getKeySet();

        List<Component> components = new ArrayList<>();

        StyleChart styleChart = new StyleChart.Builder()
                .backgroundColor(Color.WHITE)
                .width(800, LengthUnit.Px)
                .height(450, LengthUnit.Px)
                .build();

        StyleText styleText = new StyleText.Builder()
                .color(Color.BLACK)
                .fontSize(20)
                .build();
        Component headerText = new ComponentText("Spark Training Analysis Statistics Plots", styleText);
        Component header = new ComponentDiv(new StyleDiv.Builder().height(40, LengthUnit.Px).width(100, LengthUnit.Percent).build(), headerText);
        components.add(header);

        for (String s : keySet) {
            List<EventStats> list = new ArrayList<>(sparkTrainingStats.getValue(s));
            Collections.sort(list, new StartTimeComparator());

            double[] x = new double[list.size()];
            double[] duration = new double[list.size()];
            for (int i = 0; i < duration.length; i++) {
                x[i] = i;
                duration[i] = list.get(i).getDurationMs();
            }

            components.add(new ChartLine.Builder(s, styleChart)
                    .addSeries("Duration", x, duration)
                    .build());
        }

        String html = StaticPageUtil.renderHTML(components);
        SparkUtils.writeStringToFile(path, html, sc);
    }


    public static class StartTimeComparator implements Comparator<EventStats> {
        @Override
        public int compare(EventStats o1, EventStats o2) {
            return Long.compare(o1.getStartTime(), o2.getStartTime());
        }
    }
}
