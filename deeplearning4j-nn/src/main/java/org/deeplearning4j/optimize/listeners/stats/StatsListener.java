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

import java.math.BigDecimal;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Alex on 28/09/2016.
 */
public class StatsListener implements IterationListener {

    private enum StatType {Mean, Stdev, MeanMagnitude};

    private final StatsListenerReceiver receiver;
    private int iterCount = 0;

    private long initTime;
    private long lastReportTime = -1;
    private int lastReportIteration = -1;
    private int examplesSinceLastReport = 0;
    private int minibatchesSinceLastReport = 0;

    public StatsListener(StatsListenerReceiver receiver){
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
        if(iterCount == 0){
            initTime = currentTime;
        }

        if(config.collectPerformanceStats()){
            updateExamplesMinibatchesCounts(model);
        }

        if(config.reportingFrequency() > 1 && (iterCount == 0 || iterCount % config.reportingFrequency() != 0) ){
            iterCount++;
            return;
        }

        StatsReport report = receiver.newStatsReport();
        report.reportTime(currentTime);

        //--- Performance and System Stats ---

        if(config.collectPerformanceStats()){
            //Stats to collect: iterations/second, examples/second, total time,
            if(iterCount > 0){ //Not possible to work out perf/second: first iteration...
                long deltaTimeMS = currentTime - lastReportTime;
                double examplesPerSecond = 1000.0 * examplesSinceLastReport / deltaTimeMS;
                double minibatchesPerSecond = 1000.0 * minibatchesSinceLastReport / deltaTimeMS;
                report.reportPerformance(examplesPerSecond, minibatchesPerSecond);
            }
        }

        if(config.collectMemoryStats()){
            Runtime runtime = Runtime.getRuntime();
            long jvmTotal = runtime.totalMemory();
            long jvmMax = runtime.maxMemory();

            //TODO is this safe in general, for all backends?
            long offheapTotal = Pointer.totalBytes();
            long offheapMax = Pointer.maxBytes();

            //TODO: GPU...

            report.reportMemoryUse(jvmTotal,jvmMax,offheapTotal,offheapMax,null,null);
        }

        //--- General ---

        if(config.collectScore()){
            report.reportScore(model.score());
        }

        if(config.collectLearningRates()){
            //TODO
        }


        //--- Histograms ---

        if(config.collectHistogramParameters()){
            Map<String,Pair<INDArray,int[]>> paramHistograms = getHistograms(model.paramTable(), config.numHistogramBins());
            report.reportHistogramParameter(paramHistograms);
        }

        if(config.collectHistogramUpdates()){
            Map<String,Pair<INDArray,int[]>> updateHistograms = getHistograms(model.gradient().gradientForVariable(), config.numHistogramBins());
            report.reportHistogramParameter(updateHistograms);
        }

        if(config.collectHistogramActivations()){

        }


        //--- Summary Stats: Mean, Variance, Mean Magnitudes ---

        if(config.collectMean(StatsType.Parameters)){
            Map<String,Double> meanParams = calculateSummaryStats(model.paramTable(), StatType.MeanMagnitude);
            report.reportMean(StatsType.Parameters, meanParams);
        }

        if(config.collectMean(StatsType.Updates)){
            Map<String,Double> meanUpdates = calculateSummaryStats(model.gradient().gradientForVariable(), StatType.MeanMagnitude);
            report.reportMean(StatsType.Updates,meanUpdates);
        }

        if(config.collectMean(StatsType.Activations)){

        }

        if(config.collectStdev(StatsType.Parameters)){

        }

        if(config.collectStdev(StatsType.Updates)){

        }

        if(config.collectStdev(StatsType.Activations)){

        }


        if(config.collectMeanMagnitudes(StatsType.Parameters)){
            Map<String,Double> meanMagParams = calculateSummaryStats(model.paramTable(), StatType.MeanMagnitude);
            report.reportMeanMagnitudes(StatsType.Parameters, meanMagParams);
        }

        if(config.collectMeanMagnitudes(StatsType.Updates)){
            Map<String,Double> meanMagUpdates = calculateSummaryStats(model.gradient().gradientForVariable(), StatType.MeanMagnitude);
            report.reportMeanMagnitudes(StatsType.Updates, meanMagUpdates);
        }

        if(config.collectMeanMagnitudes(StatsType.Activations)){
            //TODO
        }


        iterCount++;
        lastReportTime = currentTime;

        receiver.postResult(report);
    }

    private long getTime(){
        //Abstraction to allow NTP to be plugged in later...
        return System.currentTimeMillis();
    }

    private void updateExamplesMinibatchesCounts(Model model){
        if(model instanceof MultiLayerNetwork){
            examplesSinceLastReport += ((MultiLayerNetwork) model).getInput().size(0);
        } else if(model instanceof ComputationGraph){
            examplesSinceLastReport += ((ComputationGraph) model).getInput(0).size(0);
        } else if(model instanceof Layer){
            examplesSinceLastReport += ((Layer) model).getInputMiniBatchSize();
        }
        minibatchesSinceLastReport++;
    }

    private static Map<String,Double> calculateSummaryStats(Map<String,INDArray> source, StatType statType){
        Map<String,Double> out = new LinkedHashMap<>();
        for(Map.Entry<String,INDArray> entry : source.entrySet()) {
            String name = entry.getKey();
            double value;
            switch (statType){
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

    private static Map<String,Pair<INDArray,int[]>> getHistograms(Map<String,INDArray> map, int nBins){
        //TODO This is temporary approach...
        Map<String,Pair<INDArray,int[]>> out = new LinkedHashMap<>();

        for(Map.Entry<String,INDArray> entry : map.entrySet()){
            HistogramBin histogram = new HistogramBin.Builder(entry.getValue().dup())
                    .setBinCount(nBins)
                    .setRounding(6)
                    .build();
            INDArray bins = histogram.getBins();
            int[] count = new int[nBins];
            int i = 0;
            for( Map.Entry<BigDecimal,AtomicInteger> e : histogram.getData().entrySet()){
                count[i] = e.getValue().get();
            }

            out.put(entry.getKey(), new Pair<>(bins, count));
        }
        return out;
    }
}
