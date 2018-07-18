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

package org.deeplearning4j.spark.stats;

import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.api.LengthUnit;
import org.deeplearning4j.ui.components.chart.ChartHistogram;
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
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.*;
import java.util.List;

/**
 * Utility methods for Spark training stats
 *
 * @author Alex Black
 */
public class StatsUtils {

    public static final long DEFAULT_MAX_TIMELINE_SIZE_MS = 20 * 60 * 1000; //20 minutes

    private StatsUtils() {}

    public static void exportStats(List<EventStats> list, String outputDirectory, String filename, String delimiter,
                    SparkContext sc) throws IOException {
        String path = FilenameUtils.concat(outputDirectory, filename);
        exportStats(list, path, delimiter, sc);
    }

    public static void exportStats(List<EventStats> list, String outputPath, String delimiter, SparkContext sc)
                    throws IOException {
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (EventStats e : list) {
            if (first)
                sb.append(e.getStringHeader(delimiter)).append("\n");
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
            if (count++ < num - 1)
                sb.append(delim);
        }
        return sb.toString();
    }

    public static void exportStatsAsHtml(SparkTrainingStats sparkTrainingStats, String path, JavaSparkContext sc)
                    throws Exception {
        exportStatsAsHtml(sparkTrainingStats, path, sc.sc());
    }

    /**
     * Generate and export a HTML representation (including charts, etc) of the Spark training statistics<br>
     * Note: exporting is done via Spark, so the path here can be a local file, HDFS, etc.
     *
     * @param sparkTrainingStats Stats to generate HTML page for
     * @param path               Path to export. May be local or HDFS
     * @param sc                 Spark context
     * @throws Exception IO errors or error generating HTML file
     */
    public static void exportStatsAsHtml(SparkTrainingStats sparkTrainingStats, String path, SparkContext sc)
                    throws Exception {
        exportStatsAsHtml(sparkTrainingStats, DEFAULT_MAX_TIMELINE_SIZE_MS, path, sc);
    }

    /**
     * Generate and export a HTML representation (including charts, etc) of the Spark training statistics<br>
     * Note: exporting is done via Spark, so the path here can be a local file, HDFS, etc.
     *
     * @param sparkTrainingStats Stats to generate HTML page for
     * @param path               Path to export. May be local or HDFS
     * @param maxTimelineSizeMs  maximum amount of activity to show in a single timeline plot (multiple plots will be used if training exceeds this amount of time)
     * @param sc                 Spark context
     * @throws Exception IO errors or error generating HTML file
     */
    public static void exportStatsAsHtml(SparkTrainingStats sparkTrainingStats, long maxTimelineSizeMs, String path,
                    SparkContext sc) throws Exception {
        FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
        try (BufferedOutputStream bos = new BufferedOutputStream(fileSystem.create(new Path(path)))) {
            exportStatsAsHTML(sparkTrainingStats, maxTimelineSizeMs, bos);
        }
    }

    /**
     * Generate and export a HTML representation (including charts, etc) of the Spark training statistics<br>
     * This overload is for writing to an output stream
     *
     * @param sparkTrainingStats Stats to generate HTML page for
     * @throws Exception IO errors or error generating HTML file
     */
    public static void exportStatsAsHTML(SparkTrainingStats sparkTrainingStats, OutputStream outputStream)
                    throws Exception {
        exportStatsAsHTML(sparkTrainingStats, DEFAULT_MAX_TIMELINE_SIZE_MS, outputStream);
    }

    /**
     * Generate and export a HTML representation (including charts, etc) of the Spark training statistics<br>
     * This overload is for writing to an output stream
     *
     * @param sparkTrainingStats Stats to generate HTML page for
     * @param maxTimelineSizeMs  maximum amount of activity to show in a single timeline plot (multiple plots will be used if training exceeds this amount of time)
     * @throws Exception IO errors or error generating HTML file
     */
    public static void exportStatsAsHTML(SparkTrainingStats sparkTrainingStats, long maxTimelineSizeMs,
                    OutputStream outputStream) throws Exception {
        Set<String> keySet = sparkTrainingStats.getKeySet();

        List<Component> components = new ArrayList<>();

        StyleChart styleChart = new StyleChart.Builder().backgroundColor(Color.WHITE).width(700, LengthUnit.Px)
                        .height(400, LengthUnit.Px).build();

        StyleText styleText = new StyleText.Builder().color(Color.BLACK).fontSize(20).build();
        Component headerText = new ComponentText("Deeplearning4j - Spark Training Analysis", styleText);
        Component header = new ComponentDiv(
                        new StyleDiv.Builder().height(40, LengthUnit.Px).width(100, LengthUnit.Percent).build(),
                        headerText);
        components.add(header);

        Set<String> keySetInclude = new HashSet<>();
        for (String s : keySet)
            if (sparkTrainingStats.defaultIncludeInPlots(s))
                keySetInclude.add(s);

        Collections.addAll(components,
                        getTrainingStatsTimelineChart(sparkTrainingStats, keySetInclude, maxTimelineSizeMs));

        for (String s : keySet) {
            List<EventStats> list = new ArrayList<>(sparkTrainingStats.getValue(s));
            Collections.sort(list, new StartTimeComparator());

            double[] x = new double[list.size()];
            double[] duration = new double[list.size()];
            double minDur = Double.MAX_VALUE;
            double maxDur = -Double.MAX_VALUE;
            for (int i = 0; i < duration.length; i++) {
                x[i] = i;
                duration[i] = list.get(i).getDurationMs();
                minDur = Math.min(minDur, duration[i]);
                maxDur = Math.max(maxDur, duration[i]);
            }

            Component line = new ChartLine.Builder(s, styleChart).addSeries("Duration", x, duration)
                            .setYMin(minDur == maxDur ? minDur - 1 : null).setYMax(minDur == maxDur ? minDur + 1 : null)
                            .build();

            //Also build a histogram...
            Component hist = null;
            if (minDur != maxDur && !list.isEmpty())
                hist = getHistogram(duration, 20, s, styleChart);

            Component[] temp;
            if (hist != null) {
                temp = new Component[] {line, hist};
            } else {
                temp = new Component[] {line};
            }

            components.add(new ComponentDiv(new StyleDiv.Builder().width(100, LengthUnit.Percent).build(), temp));


            //TODO this is really ugly
            if (!list.isEmpty() && (list.get(0) instanceof ExampleCountEventStats
                            || list.get(0) instanceof PartitionCountEventStats)) {
                boolean exCount = list.get(0) instanceof ExampleCountEventStats;

                double[] y = new double[list.size()];
                double miny = Double.MAX_VALUE;
                double maxy = -Double.MAX_VALUE;
                for (int i = 0; i < y.length; i++) {
                    y[i] = (exCount ? ((ExampleCountEventStats) list.get(i)).getTotalExampleCount()
                                    : ((PartitionCountEventStats) list.get(i)).getNumPartitions());
                    miny = Math.min(miny, y[i]);
                    maxy = Math.max(maxy, y[i]);
                }

                String title = s + " / " + (exCount ? "Number of Examples" : "Number of Partitions");
                Component line2 = new ChartLine.Builder(title, styleChart)
                                .addSeries((exCount ? "Examples" : "Partitions"), x, y)
                                .setYMin(miny == maxy ? miny - 1 : null).setYMax(miny == maxy ? miny + 1 : null)
                                .build();


                //Also build a histogram...
                Component hist2 = null;
                if (miny != maxy)
                    hist2 = getHistogram(y, 20, title, styleChart);

                Component[] temp2;
                if (hist2 != null) {
                    temp2 = new Component[] {line2, hist2};
                } else {
                    temp2 = new Component[] {line2};
                }

                components.add(new ComponentDiv(new StyleDiv.Builder().width(100, LengthUnit.Percent).build(), temp2));
            }
        }

        String html = StaticPageUtil.renderHTML(components);
        outputStream.write(html.getBytes("UTF-8"));
    }


    public static class StartTimeComparator implements Comparator<EventStats> {
        @Override
        public int compare(EventStats o1, EventStats o2) {
            return Long.compare(o1.getStartTime(), o2.getStartTime());
        }
    }


    private static Component[] getTrainingStatsTimelineChart(SparkTrainingStats stats, Set<String> includeSet,
                    long maxDurationMs) {
        Set<Tuple3<String, String, Long>> uniqueTuples = new HashSet<>();
        Set<String> machineIDs = new HashSet<>();
        Set<String> jvmIDs = new HashSet<>();

        Map<String, String> machineShortNames = new HashMap<>();
        Map<String, String> jvmShortNames = new HashMap<>();

        long earliestStart = Long.MAX_VALUE;
        long latestEnd = Long.MIN_VALUE;
        for (String s : includeSet) {
            List<EventStats> list = stats.getValue(s);
            for (EventStats e : list) {
                machineIDs.add(e.getMachineID());
                jvmIDs.add(e.getJvmID());
                uniqueTuples.add(new Tuple3<String, String, Long>(e.getMachineID(), e.getJvmID(), e.getThreadID()));
                earliestStart = Math.min(earliestStart, e.getStartTime());
                latestEnd = Math.max(latestEnd, e.getStartTime() + e.getDurationMs());
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

        Color[] colors = getColors(includeSet.size());
        Map<String, Color> colorMap = new HashMap<>();
        count = 0;
        for (String s : includeSet) {
            colorMap.put(s, colors[count++]);
        }

        //Create key for charts:
        List<Component> tempList = new ArrayList<>();
        for (String s : includeSet) {
            String key = stats.getShortNameForKey(s) + " - " + s;

            tempList.add(new ComponentDiv(
                            new StyleDiv.Builder().backgroundColor(colorMap.get(s)).width(33.3, LengthUnit.Percent)
                                            .height(25, LengthUnit.Px).floatValue(StyleDiv.FloatValue.left).build(),
                            new ComponentText(key, new StyleText.Builder().fontSize(11).build())));
        }
        Component key = new ComponentDiv(new StyleDiv.Builder().width(100, LengthUnit.Percent).build(), tempList);

        //How many charts?
        int nCharts = (int) ((latestEnd - earliestStart) / maxDurationMs);
        if (nCharts < 1)
            nCharts = 1;
        long[] chartStartTimes = new long[nCharts];
        long[] chartEndTimes = new long[nCharts];
        for (int i = 0; i < nCharts; i++) {
            chartStartTimes[i] = earliestStart + i * maxDurationMs;
            chartEndTimes[i] = earliestStart + (i + 1) * maxDurationMs;
        }


        List<List<List<ChartTimeline.TimelineEntry>>> entriesByLane = new ArrayList<>();
        for (int c = 0; c < nCharts; c++) {
            entriesByLane.add(new ArrayList<List<ChartTimeline.TimelineEntry>>());
            for (int i = 0; i < nLanes; i++) {
                entriesByLane.get(c).add(new ArrayList<ChartTimeline.TimelineEntry>());
            }
        }

        for (String s : includeSet) {

            List<EventStats> list = stats.getValue(s);
            for (EventStats e : list) {
                if (e.getDurationMs() == 0)
                    continue;

                long start = e.getStartTime();
                long end = start + e.getDurationMs();

                int chartIdx = -1;
                for (int j = 0; j < nCharts; j++) {
                    if (start >= chartStartTimes[j] && start < chartEndTimes[j]) {
                        chartIdx = j;
                    }
                }
                if (chartIdx == -1)
                    chartIdx = nCharts - 1;


                Tuple3<String, String, Long> tuple = new Tuple3<>(e.getMachineID(), e.getJvmID(), e.getThreadID());

                int idx = outputOrder.indexOf(tuple);
                Color c = colorMap.get(s);
                //                ChartTimeline.TimelineEntry entry = new ChartTimeline.TimelineEntry(null, start, end, c);
                ChartTimeline.TimelineEntry entry =
                                new ChartTimeline.TimelineEntry(stats.getShortNameForKey(s), start, end, c);
                entriesByLane.get(chartIdx).get(idx).add(entry);
            }
        }

        //Sort each lane by start time:
        for (int i = 0; i < nCharts; i++) {
            for (List<ChartTimeline.TimelineEntry> l : entriesByLane.get(i)) {
                Collections.sort(l, new Comparator<ChartTimeline.TimelineEntry>() {
                    @Override
                    public int compare(ChartTimeline.TimelineEntry o1, ChartTimeline.TimelineEntry o2) {
                        return Long.compare(o1.getStartTimeMs(), o2.getStartTimeMs());
                    }
                });
            }
        }

        StyleChart sc = new StyleChart.Builder().width(1280, LengthUnit.Px)
                        .height(35 * nLanes + (60 + 20 + 25), LengthUnit.Px).margin(LengthUnit.Px, 60, 20, 200, 10) //top, bottom, left, right
                        .build();

        List<Component> list = new ArrayList<>(nCharts);
        for (int j = 0; j < nCharts; j++) {
            ChartTimeline.Builder b = new ChartTimeline.Builder("Timeline: Training Activities", sc);
            int i = 0;
            for (List<ChartTimeline.TimelineEntry> l : entriesByLane.get(j)) {
                Tuple3<String, String, Long> t3 = outputOrder.get(i);
                String name = machineShortNames.get(t3._1()) + ", " + jvmShortNames.get(t3._2()) + ", Thread "
                                + t3._3();
                b.addLane(name, l);
                i++;
            }
            list.add(b.build());
        }

        list.add(key);

        return list.toArray(new Component[list.size()]);
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

    private static Color[] getColors(int nColors) {
        Color[] c = new Color[nColors];
        double step;
        if (nColors <= 1)
            step = 1.0;
        else
            step = 1.0 / (nColors + 1);
        for (int i = 0; i < nColors; i++) {
            //            c[i] = Color.getHSBColor((float) step * i, 0.4f, 0.75f);   //step hue; fixed saturation + variance to (hopefully) ensure readability of labels
            if (i % 2 == 0)
                c[i] = Color.getHSBColor((float) step * i, 0.4f, 0.75f); //step hue; fixed saturation + variance to (hopefully) ensure readability of labels
            else
                c[i] = Color.getHSBColor((float) step * i, 1.0f, 1.0f); //step hue; fixed saturation + variance to (hopefully) ensure readability of labels
        }
        return c;
    }

    private static Component getHistogram(double[] data, int nBins, String title, StyleChart styleChart) {
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        for (double d : data) {
            min = Math.min(min, d);
            max = Math.max(max, d);
        }

        if (min == max)
            return null;
        double[] bins = new double[nBins + 1];
        int[] counts = new int[nBins];
        double step = (max - min) / nBins;
        for (int i = 0; i < bins.length; i++)
            bins[i] = min + i * step;

        for (double d : data) {
            for (int i = 0; i < bins.length - 1; i++) {
                if (d >= bins[i] && d < bins[i + 1]) {
                    counts[i]++;
                    break;
                }
            }
            if (d == bins[bins.length - 1])
                counts[counts.length - 1]++;
        }

        ChartHistogram.Builder b = new ChartHistogram.Builder(title, styleChart);
        for (int i = 0; i < bins.length - 1; i++) {
            b.addBin(bins[i], bins[i + 1], counts[i]);
        }

        return b.build();
    }
}
