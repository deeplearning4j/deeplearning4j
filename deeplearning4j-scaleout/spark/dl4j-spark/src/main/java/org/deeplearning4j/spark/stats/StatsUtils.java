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
import org.deeplearning4j.ui.components.chart.ChartTimeline;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.deeplearning4j.ui.components.component.ComponentDiv;
import org.deeplearning4j.ui.components.component.style.StyleDiv;
import org.deeplearning4j.ui.components.text.ComponentText;
import org.deeplearning4j.ui.components.text.style.StyleText;
import org.deeplearning4j.ui.standalone.StaticPageUtil;
import scala.Tuple3;

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

    /**
     * Generate and export a HTML representation (including charts, etc) of the Spark training statistics<br>
     * Note: exporting is done via Spark, so the path here can be a local file, HDFS, etc.
     *
     * @param sparkTrainingStats    Stats to generate HTML page for
     * @param path                  Path to export. May be local or HDFS
     * @param sc                    Spark context
     * @throws Exception            IO errors or error generating HTML file
     */
    public static void exportStatsAsHtml(SparkTrainingStats sparkTrainingStats, String path, SparkContext sc) throws Exception {
        Set<String> keySet = sparkTrainingStats.getKeySet();

        List<Component> components = new ArrayList<>();

        StyleChart styleChart = new StyleChart.Builder()
                .backgroundColor(Color.WHITE)
                .width(800, LengthUnit.Px)
                .height(400, LengthUnit.Px)
                .build();

        StyleText styleText = new StyleText.Builder()
                .color(Color.BLACK)
                .fontSize(20)
                .build();
        Component headerText = new ComponentText("Deeplearning4j - Spark Training Analysis Statistics Plots", styleText);
        Component header = new ComponentDiv(new StyleDiv.Builder().height(40, LengthUnit.Px).width(100, LengthUnit.Percent).build(), headerText);
        components.add(header);

        Set<String> keySetInclude = new HashSet<>();
        for(String s : keySet) if(sparkTrainingStats.defaultIncludeInPlots(s)) keySetInclude.add(s);

        components.add(getTrainingStatsTimelineChart(sparkTrainingStats, keySetInclude));

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


    private static Component getTrainingStatsTimelineChart(SparkTrainingStats stats, Set<String> includeSet) {
        Set<Tuple3<String, String, Long>> uniqueTuples = new HashSet<>();
        Set<String> machineIDs = new HashSet<>();
        Set<String> jvmIDs = new HashSet<>();

        Map<String, String> machineShortNames = new HashMap<>();
        Map<String, String> jvmShortNames = new HashMap<>();

        for (String s : includeSet) {
            List<EventStats> list = stats.getValue(s);
            for (EventStats e : list) {
                machineIDs.add(e.getMachineID());
                jvmIDs.add(e.getJvmID());
                uniqueTuples.add(new Tuple3<>(e.getMachineID(), e.getJvmID(), e.getThreadID()));
            }
        }
        int count = 0;
        for (String s : machineIDs) {
            machineShortNames.put(s, "PC " + count++);
        }
        count = 0;
        for (String s : jvmIDs) {
            jvmShortNames.put(s, "JVM " + count++);
        }

        int nLanes = uniqueTuples.size();
        List<Tuple3<String, String, Long>> outputOrder = new ArrayList<>(uniqueTuples);
        Collections.sort(outputOrder, new TupleComparator());

        Color[] colors = getRandomColors(includeSet.size(), 12345);
        Map<String, Color> colorMap = new HashMap<>();
        count = 0;
        for (String s : includeSet) {
            colorMap.put(s, colors[count++]);
        }

        List<List<ChartTimeline.TimelineEntry>> entriesByLane = new ArrayList<>();
        for (int i = 0; i < nLanes; i++) entriesByLane.add(new ArrayList<ChartTimeline.TimelineEntry>());
        for (String s : includeSet) {

            List<EventStats> list = stats.getValue(s);
            for (EventStats e : list) {
                if (e.getDurationMs() == 0) continue;

                long start = e.getStartTime();
                long end = start + e.getDurationMs();


                Tuple3<String, String, Long> tuple = new Tuple3<>(e.getMachineID(), e.getJvmID(), e.getThreadID());

                int idx = outputOrder.indexOf(tuple);
                Color c = colorMap.get(s);
                ChartTimeline.TimelineEntry entry = new ChartTimeline.TimelineEntry(stats.getShortNameForKey(s), start, end, c);
                entriesByLane.get(idx).add(entry);
            }
        }

        //Sort each lane by start time:
        for (List<ChartTimeline.TimelineEntry> l : entriesByLane) {
            Collections.sort(l, new Comparator<ChartTimeline.TimelineEntry>() {
                @Override
                public int compare(ChartTimeline.TimelineEntry o1, ChartTimeline.TimelineEntry o2) {
                    return Long.compare(o1.getStartTimeMs(), o2.getStartTimeMs());
                }
            });
        }

        StyleChart sc = new StyleChart.Builder()
                .width(1280, LengthUnit.Px)
                .height(60 * nLanes + 320, LengthUnit.Px)
                .margin(LengthUnit.Px, 60, 10, 200, 10) //top, bottom, left, right
                .build();

        ChartTimeline.Builder b = new ChartTimeline.Builder("Timeline: Training Activities", sc);
        int i = 0;
        for (List<ChartTimeline.TimelineEntry> l : entriesByLane) {
            Tuple3<String, String, Long> t3 = outputOrder.get(i);
            String name = machineShortNames.get(t3._1()) + ", " + jvmShortNames.get(t3._2()) + ", Thread " + t3._3();
            b.addLane(name, l);
            i++;
        }

        return b.build();
    }

    private static class TupleComparator implements Comparator<Tuple3<String, String, Long>> {
        @Override
        public int compare(Tuple3<String, String, Long> o1, Tuple3<String, String, Long> o2) {
            if (o1._1().equals(o2._1())) {
                //Equal machine IDs, so sort on JVM ids
                if (o1._2().equals(o2._2())) {
                    //Equal machine AND JVM IDs, so sort on thread ID
                    return Long.compare(o1._3(), o2._3());
                } else {
                    return o1._2().compareTo(o2._2());
                }
            } else {
                return o1._1().compareTo(o2._1());
            }
        }
    }

    private static Color[] getRandomColors(int nColors, long seed) {
        Random r = new Random(seed);
        Color[] c = new Color[nColors];
        for (int i = 0; i < nColors; i++) {
            c[i] = Color.getHSBColor(r.nextFloat(), 0.5f, 0.75f);   //random hue; fixed saturation + variance to (hopefully) ensure readability of labels
        }
        return c;
    }
}
