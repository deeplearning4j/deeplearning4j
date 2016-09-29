package org.deeplearning4j.optimize.listeners.stats;

import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.stats.temp.HistogramBin;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.math.BigDecimal;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Alex on 28/09/2016.
 */
public class StatsListener implements IterationListener {

    private enum StatType {Mean, Stdev, MeanMagnitude}

    private final StatsListenerReceiver receiver;
    private int iterCount = 0;

    private long initTime;
    private long lastReportTime = -1;
    private int lastReportIteration = -1;
    private int examplesSinceLastReport = 0;
    private int minibatchesSinceLastReport = 0;

    private long totalExamples = 0;
    private long totalMinibatches = 0;

    private List<GarbageCollectorMXBean> gcBeans;
    private Map<String,Pair<Long,Long>> gcStatsAtLastReport;

    public StatsListener(StatsListenerReceiver receiver) {
        this.receiver = receiver;
    }

    @Override
    public boolean invoked() {
        return false;
    }

    @Override
    public void invoke() {

    }

    @Override
    public void iterationDone(Model model, int iteration) {
        StatsListenerConfiguration config = receiver.getCurrentConfiguration();

        long currentTime = getTime();
        if (iterCount == 0) {
            initTime = currentTime;
        }

        if (config.collectPerformanceStats()) {
            updateExamplesMinibatchesCounts(model);
        }

        if (config.reportingFrequency() > 1 && (iterCount == 0 || iterCount % config.reportingFrequency() != 0)) {
            iterCount++;
            return;
        }

        StatsReport report = receiver.newStatsReport();
        report.reportTime(currentTime);

        long deltaReportTime = currentTime - lastReportTime;

        //--- Performance and System Stats ---

        if (config.collectPerformanceStats()) {
            //Stats to collect: total runtime, total examples, total minibatches, iterations/second, examples/second
            double examplesPerSecond;
            double minibatchesPerSecond;
            if (iterCount == 0) {
                //Not possible to work out perf/second: first iteration...
                examplesPerSecond = 0.0;
                minibatchesPerSecond = 0.0;
            } else {
                long deltaTimeMS = currentTime - lastReportTime;
                examplesPerSecond = 1000.0 * examplesSinceLastReport / deltaTimeMS;
                minibatchesPerSecond = 1000.0 * minibatchesSinceLastReport / deltaTimeMS;
            }
            long totalRuntimeMS = currentTime - initTime;
            report.reportPerformance(totalRuntimeMS, totalExamples, totalMinibatches, examplesPerSecond, minibatchesPerSecond);

            examplesSinceLastReport = 0;
            minibatchesSinceLastReport = 0;
        }

        if (config.collectMemoryStats()) {
            Runtime runtime = Runtime.getRuntime();
            long jvmTotal = runtime.totalMemory();
            long jvmMax = runtime.maxMemory();

            //Off-heap memory
            long offheapTotal = Pointer.totalBytes();
            long offheapMax = Pointer.maxBytes();

            //TODO: GPU...

            report.reportMemoryUse(jvmTotal, jvmMax, offheapTotal, offheapMax, null, null);
        }

        if(config.collectGarbageCollectionStats()){
            if(lastReportIteration == -1 || gcBeans == null){
                //Haven't reported GC stats before...
                gcBeans = ManagementFactory.getGarbageCollectorMXBeans();
                gcStatsAtLastReport = new HashMap<>();
                for( GarbageCollectorMXBean bean : gcBeans ){
                    long count = bean.getCollectionCount();
                    long timeMs = bean.getCollectionTime();
                    gcStatsAtLastReport.put(bean.getName(), new Pair<>(count,timeMs));
                }
            } else {
                for( GarbageCollectorMXBean bean : gcBeans ){
                    long count = bean.getCollectionCount();
                    long timeMs = bean.getCollectionTime();
                    Pair<Long,Long> lastStats = gcStatsAtLastReport.get(bean.getName());
                    long deltaCount = count - lastStats.getFirst();
                    long deltaGCTime = timeMs - lastStats.getSecond();

                    lastStats.setFirst(count);
                    lastStats.setSecond(timeMs);
                    report.reportGarbageCollection(bean.getName(), deltaReportTime, deltaCount, deltaGCTime);
                }
            }
        }

        //--- General ---

        if (config.collectScore()) {
            report.reportScore(model.score());
        }

        if (config.collectLearningRates()) {
            //TODO - how to get this?
        }


        //--- Histograms ---

        if (config.collectHistograms(StatsType.Parameters)) {
            Map<String, Pair<INDArray, int[]>> paramHistograms = getHistograms(model.paramTable(), config.numHistogramBins(StatsType.Parameters));
            report.reportHistograms(StatsType.Parameters, paramHistograms);
        }

        if (config.collectHistograms(StatsType.Updates)) {
            Map<String, Pair<INDArray, int[]>> updateHistograms = getHistograms(model.gradient().gradientForVariable(), config.numHistogramBins(StatsType.Updates));
            report.reportHistograms(StatsType.Updates, updateHistograms);
        }

        if (config.collectHistograms(StatsType.Activations)) {
            Map<String, INDArray> activations = getActivationArraysMap(model);
            Map<String, Pair<INDArray, int[]>> activationHistograms = getHistograms(activations, config.numHistogramBins(StatsType.Activations));
            report.reportHistograms(StatsType.Activations, activationHistograms);
        }


        //--- Summary Stats: Mean, Variance, Mean Magnitudes ---

        if (config.collectMean(StatsType.Parameters)) {
            Map<String, Double> meanParams = calculateSummaryStats(model.paramTable(), StatType.Mean);
            report.reportMean(StatsType.Parameters, meanParams);
        }

        if (config.collectMean(StatsType.Updates)) {
            Map<String, Double> meanUpdates = calculateSummaryStats(model.gradient().gradientForVariable(), StatType.Mean);
            report.reportMean(StatsType.Updates, meanUpdates);
        }

        if (config.collectMean(StatsType.Activations)) {
            Map<String, INDArray> activations = getActivationArraysMap(model);
            Map<String, Double> meanActivations = calculateSummaryStats(activations, StatType.Mean);
            report.reportMean(StatsType.Activations, meanActivations);
        }


        if (config.collectStdev(StatsType.Parameters)) {
            Map<String, Double> stdevParams = calculateSummaryStats(model.paramTable(), StatType.Stdev);
            report.reportStdev(StatsType.Parameters, stdevParams);
        }

        if (config.collectStdev(StatsType.Updates)) {
            Map<String, Double> stdevUpdates = calculateSummaryStats(model.gradient().gradientForVariable(), StatType.Stdev);
            report.reportStdev(StatsType.Updates, stdevUpdates);
        }

        if (config.collectStdev(StatsType.Activations)) {
            Map<String, INDArray> activations = getActivationArraysMap(model);
            Map<String, Double> stdevActivations = calculateSummaryStats(activations, StatType.Stdev);
            report.reportStdev(StatsType.Activations, stdevActivations);
        }


        if (config.collectMeanMagnitudes(StatsType.Parameters)) {
            Map<String, Double> meanMagParams = calculateSummaryStats(model.paramTable(), StatType.MeanMagnitude);
            report.reportMeanMagnitudes(StatsType.Parameters, meanMagParams);
        }

        if (config.collectMeanMagnitudes(StatsType.Updates)) {
            Map<String, Double> meanMagUpdates = calculateSummaryStats(model.gradient().gradientForVariable(), StatType.MeanMagnitude);
            report.reportMeanMagnitudes(StatsType.Updates, meanMagUpdates);
        }

        if (config.collectMeanMagnitudes(StatsType.Activations)) {
            Map<String, INDArray> activations = getActivationArraysMap(model);
            Map<String, Double> meanMagActivations = calculateSummaryStats(activations, StatType.MeanMagnitude);
            report.reportMeanMagnitudes(StatsType.Activations, meanMagActivations);
        }


        long endTime = getTime();
        report.reportStatsCollectionDurationMS(endTime-currentTime);    //Amount of time required to alculate all histograms, means etc.
        lastReportTime = currentTime;
        lastReportIteration = iterCount;
        receiver.postResult(report);
        iterCount++;
    }

    private long getTime() {
        //Abstraction to allow NTP to be plugged in later...
        return System.currentTimeMillis();
    }

    private void updateExamplesMinibatchesCounts(Model model) {
        int examplesThisMinibatch = 0;
        if (model instanceof MultiLayerNetwork) {
            examplesThisMinibatch = ((MultiLayerNetwork) model).getInput().size(0);
        } else if (model instanceof ComputationGraph) {
            examplesThisMinibatch = ((ComputationGraph) model).getInput(0).size(0);
        } else if (model instanceof Layer) {
            examplesThisMinibatch = ((Layer) model).getInputMiniBatchSize();
        }
        examplesSinceLastReport += examplesThisMinibatch;
        totalExamples += examplesThisMinibatch;
        minibatchesSinceLastReport++;
        totalMinibatches++;
    }

    private static Map<String, Double> calculateSummaryStats(Map<String, INDArray> source, StatType statType) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (Map.Entry<String, INDArray> entry : source.entrySet()) {
            String name = entry.getKey();
            double value;
            switch (statType) {
                case Mean:
                    value = entry.getValue().meanNumber().doubleValue();
                    break;
                case Stdev:
                    value = entry.getValue().stdNumber().doubleValue();
                    break;
                case MeanMagnitude:
                    value = entry.getValue().norm1Number().doubleValue() / entry.getValue().length();
                    break;
                default:
                    throw new RuntimeException();   //Should never happen
            }
            out.put(name, value);
        }
        return out;
    }

    private static Map<String, Pair<INDArray, int[]>> getHistograms(Map<String, INDArray> map, int nBins) {
        //TODO This is temporary approach...
        Map<String, Pair<INDArray, int[]>> out = new LinkedHashMap<>();

        for (Map.Entry<String, INDArray> entry : map.entrySet()) {
            HistogramBin histogram = new HistogramBin.Builder(entry.getValue().dup())
                    .setBinCount(nBins)
                    .setRounding(6)
                    .build();
            INDArray bins = histogram.getBins();
            int[] count = new int[nBins];
            int i = 0;
            for (Map.Entry<BigDecimal, AtomicInteger> e : histogram.getData().entrySet()) {
                count[i] = e.getValue().get();
            }

            out.put(entry.getKey(), new Pair<>(bins, count));
        }
        return out;
    }

    private static Map<String, INDArray> getActivationArraysMap(Model model) {
        Map<String, INDArray> map = new LinkedHashMap<>();
        if (model instanceof MultiLayerNetwork) {
            MultiLayerNetwork net = (MultiLayerNetwork) model;

            Layer[] layers = net.getLayers();
            //Activations for layer i are stored as input to layer i+1
            //TODO handle output activations...
            //Also: complication here - things like batch norm...
            for (int i = 1; i < layers.length; i++) {
                String name = String.valueOf(i - 1);
                map.put(name, layers[i].input());
            }

        } else {
            //Compgraph is more complex: output from one layer might go to multiple other layers/vertices, etc.
            throw new UnsupportedOperationException("Not yet implemented");
        }

        return map;
    }
}
