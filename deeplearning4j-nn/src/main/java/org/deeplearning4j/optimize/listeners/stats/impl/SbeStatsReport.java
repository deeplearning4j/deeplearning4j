package org.deeplearning4j.optimize.listeners.stats.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.optimize.listeners.stats.StatsType;
import org.deeplearning4j.optimize.listeners.stats.api.Histogram;
import org.deeplearning4j.optimize.listeners.stats.api.StatsReport;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Alex on 01/10/2016.
 */
public class SbeStatsReport implements StatsReport {

    private int iterationCount;
    private long reportTime;
    private long statsCollectionDurationMs;
    private double currentScore;

    private long jvmCurrentBytes;
    private long jvmMaxBytes;
    private long offHeapCurrentBytes;
    private long offHeapMaxBytes;
    private long[] deviceCurrentBytes;
    private long[] deviceMaxBytes;

    private long totalRuntimeMs;
    private long totalExamples;
    private long totalMinibatches;
    private double examplesPerSecond;
    private double minibatchesPerSecond;

    private List<GCStats> gcStats;

    private Map<StatsType,Map<String,Histogram>> histograms;
    private Map<StatsType,Map<String,Double>> meanValues;
    private Map<StatsType,Map<String,Double>> stdevValues;
    private Map<StatsType,Map<String,Double>> meanMagnitudeValues;

    @Override
    public void reportIterationCount(int iterationCount) {
        this.iterationCount = iterationCount;
    }

    @Override
    public void reportTime(long reportTime) {
        this.reportTime = reportTime;
    }

    @Override
    public long getTime() {
        return reportTime;
    }

    @Override
    public void reportStatsCollectionDurationMS(long statsCollectionDurationMS) {
        this.statsCollectionDurationMs = statsCollectionDurationMS;
    }

    @Override
    public void reportScore(double currentScore) {
        this.currentScore = currentScore;
    }

    @Override
    public void reportMemoryUse(long jvmCurrentBytes, long jvmMaxBytes, long offHeapCurrentBytes, long offHeapMaxBytes,
                                long[] deviceCurrentBytes, long[] deviceMaxBytes) {
        this.jvmCurrentBytes = jvmCurrentBytes;
        this.jvmMaxBytes = jvmMaxBytes;
        this.offHeapCurrentBytes = offHeapCurrentBytes;
        this.offHeapMaxBytes = offHeapMaxBytes;
        this.deviceCurrentBytes = deviceCurrentBytes;
        this.deviceMaxBytes = deviceMaxBytes;
    }

    @Override
    public void reportPerformance(long totalRuntimeMs, long totalExamples, long totalMinibatches, double examplesPerSecond,
                                  double minibatchesPerSecond) {
        this.totalRuntimeMs = totalRuntimeMs;
        this.totalExamples = totalExamples;
        this.totalMinibatches = totalMinibatches;
        this.examplesPerSecond = examplesPerSecond;
        this.minibatchesPerSecond = minibatchesPerSecond;
    }

    @Override
    public void reportGarbageCollection(String gcName, int deltaReportTimeMs, int deltaGCCount, int deltaGCTime) {
        if(gcStats == null) gcStats = new ArrayList<>();
        gcStats.add(new GCStats(gcName, deltaReportTimeMs, deltaGCCount, deltaGCTime));
    }

    @Override
    public void reportHistograms(StatsType statsType, Map<String, Histogram> histogram) {
        if(this.histograms == null) this.histograms = new HashMap<>();
        this.histograms.put(statsType, histogram);
    }

    @Override
    public void reportMean(StatsType statsType, Map<String, Double> mean) {
        if(this.meanValues == null) this.meanValues = new HashMap<>();
        this.meanValues.put(statsType, mean);
    }

    @Override
    public void reportStdev(StatsType statsType, Map<String, Double> stdev) {
        if(this.stdevValues == null) this.stdevValues = new HashMap<>();
        this.stdevValues.put(statsType,stdev);
    }

    @Override
    public void reportMeanMagnitudes(StatsType statsType, Map<String, Double> meanMagnitudes) {
        if(this.meanMagnitudeValues == null) this.meanMagnitudeValues = new HashMap<>();
        this.meanMagnitudeValues.put(statsType,meanMagnitudes);
    }

    @Override
    public byte[] toByteArray() {
        
    }

    @Override
    public void fromByteArray(byte[] bytes){



    }

    @AllArgsConstructor @Data
    private static class GCStats {
        private String gcName;
        private int deltaReportTime;
        private int deltaGCCount;
        private int deltaGCTime;
    }
}
