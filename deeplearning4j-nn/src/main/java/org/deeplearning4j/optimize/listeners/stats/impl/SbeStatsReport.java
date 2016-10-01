package org.deeplearning4j.optimize.listeners.stats.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.agrona.MutableDirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.deeplearning4j.optimize.listeners.stats.StatsType;
import org.deeplearning4j.optimize.listeners.stats.api.Histogram;
import org.deeplearning4j.optimize.listeners.stats.api.StatsReport;
import org.deeplearning4j.optimize.listeners.stats.sbe.MemoryType;
import org.deeplearning4j.optimize.listeners.stats.sbe.MessageHeaderEncoder;
import org.deeplearning4j.optimize.listeners.stats.sbe.UpdateEncoder;

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

    private boolean scorePresent;
    private boolean memoryUsePresent;
    private boolean performanceStatsPresent;

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
        this.scorePresent = true;
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
        this.memoryUsePresent = true;
    }

    @Override
    public void reportPerformance(long totalRuntimeMs, long totalExamples, long totalMinibatches, double examplesPerSecond,
                                  double minibatchesPerSecond) {
        this.totalRuntimeMs = totalRuntimeMs;
        this.totalExamples = totalExamples;
        this.totalMinibatches = totalMinibatches;
        this.examplesPerSecond = examplesPerSecond;
        this.minibatchesPerSecond = minibatchesPerSecond;
        this.performanceStatsPresent = true;
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

        MessageHeaderEncoder enc = new MessageHeaderEncoder();
        UpdateEncoder ue = new UpdateEncoder();

        //First: determine buffer size.
        //(a) Header: 8 bytes (4x uint16 = 8 bytes)
        //(b) Fixed length entries length (sie.BlockLength())
        //(c) Group 1: Memory use.
        //(d) Group 2: Performance stats
        //(e) Group 3: GC stats
        //(f) Group 4: Per parameter performance stats
        //Here: no variable length String fields...

        int bufferSize = 8 + ue.sbeBlockLength();

        //TODO need full length calc here...


        byte[] bytes = new byte[bufferSize];
        MutableDirectBuffer buffer = new UnsafeBuffer(bytes);
        enc.wrap(buffer, 0)
                .blockLength(ue.sbeBlockLength())
                .templateId(ue.sbeTemplateId())
                .schemaId(ue.sbeSchemaId())
                .version(ue.sbeSchemaVersion());

        int offset = enc.encodedLength();   //Expect 8 bytes

        //Fixed length fields: always encoded
        ue.time(reportTime)
                .deltaTime(0)   //TODO
                .fieldsPresent()
                .score(scorePresent)
                .memoryUse(memoryUsePresent)
                .performance(performanceStatsPresent)
                .garbageCollection(gcStats != null && !gcStats.isEmpty())
                .histogramParameters(histograms != null && histograms.containsKey(StatsType.Parameters))
                .histogramUpdates(histograms != null && histograms.containsKey(StatsType.Updates))
                .histogramActivations(histograms != null && histograms.containsKey(StatsType.Activations))
                .meanParameters(meanValues != null && meanValues.containsKey(StatsType.Parameters))
                .meanUpdates(meanValues != null && meanValues.containsKey(StatsType.Updates))
                .meanActivations(meanValues != null && meanValues.containsKey(StatsType.Activations))
                .meanMagnitudeParameters(meanMagnitudeValues != null && meanMagnitudeValues.containsKey(StatsType.Parameters))
                .meanMagnitudeUpdates(meanMagnitudeValues != null && meanMagnitudeValues.containsKey(StatsType.Updates))
                .meanMagnitudeActivations(meanMagnitudeValues != null && meanMagnitudeValues.containsKey(StatsType.Activations));

        ue.statsCollectionDuration(statsCollectionDurationMs);

        int memoryUseCount;
        if(!memoryUsePresent){
            memoryUseCount = 0;
        } else {
            memoryUseCount = 4 + (deviceCurrentBytes == null ? 0 : deviceCurrentBytes.length)
                    + (deviceMaxBytes == null ? 0 : deviceMaxBytes.length);
        }
        UpdateEncoder.MemoryUseEncoder mue = ue.memoryUseCount(memoryUseCount);
        if(memoryUsePresent){
            mue.next().memoryType(MemoryType.JvmCurrent).memoryBytes(jvmCurrentBytes)
                    .next().memoryType(MemoryType.JvmMax).memoryBytes(jvmMaxBytes)
                    .next().memoryType(MemoryType.OffHeapCurrent).memoryBytes(offHeapCurrentBytes)
                    .next().memoryType(MemoryType.OffHeapMax).memoryBytes(offHeapMaxBytes);
            if(deviceCurrentBytes != null){
                for( int i=0; i<deviceCurrentBytes.length; i++ ){
                    mue.next().memoryType(MemoryType.DeviceCurrent).memoryBytes(deviceCurrentBytes[i]);
                }
            }
            if(deviceMaxBytes != null){
                for( int i=0; i<deviceMaxBytes.length; i++ ){
                    mue.next().memoryType(MemoryType.DeviceMax).memoryBytes(deviceMaxBytes[i]);
                }
            }
        }

        UpdateEncoder.PerformanceEncoder pe = ue.performanceCount(performanceStatsPresent ? 1 : 0);
        if(performanceStatsPresent){
            pe.next().totalRuntimeMs(totalRuntimeMs)
                    .totalExamples(totalExamples)
                    .totalMinibatches(totalMinibatches)
                    .examplesPerSecond((float)examplesPerSecond)
                    .minibatchesPerSecond((float)minibatchesPerSecond);
        }

        UpdateEncoder.GcStatsEncoder gce = ue.gcStatsCount(gcStats == null && gcStats.size() > 0 ? 0 : gcStats.size());
        if(gcStats != null && gcStats.size() > 0 ){
            for(GCStats g : gcStats){
                gce.next().deltaGCCount(g.deltaGCCount)
                        .deltaGCTimeMs(g.deltaGCTime)
                        .gcName(g.getGcName());
            }
        }

        //TODO: need to handle parameters order...











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
