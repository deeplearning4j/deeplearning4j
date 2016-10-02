package org.deeplearning4j.optimize.listeners.stats.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.agrona.MutableDirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.deeplearning4j.optimize.listeners.stats.api.StatsType;
import org.deeplearning4j.optimize.listeners.stats.api.Histogram;
import org.deeplearning4j.optimize.listeners.stats.api.StatsReport;
import org.deeplearning4j.optimize.listeners.stats.api.SummaryType;
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

        //--------------------------------------------------------------------------------------------------------------
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

        //Memory use group length...
        int memoryUseCount;
        if(!memoryUsePresent){
            memoryUseCount = 0;
        } else {
            memoryUseCount = 4 + (deviceCurrentBytes == null ? 0 : deviceCurrentBytes.length)
                    + (deviceMaxBytes == null ? 0 : deviceMaxBytes.length);
        }

        bufferSize += 4 + 5*memoryUseCount;    //Group header: 4 bytes (always present); Each entry in group - 1x MemoryType (uint8) + 1x int64 -> 5 bytes

        //Performance group length
        bufferSize += 4 + (memoryUsePresent ? 32 : 0); //Group header: 4 bytes (always present); Only 1 group: 3xint64 + 2xfloat = 32 bytes

        //GC stats group length
        bufferSize += 4;    //Group header: always present
        List<byte[]> gcStatsLabelBytes;
        if(gcStats != null && gcStats.size() > 0){
            gcStatsLabelBytes = new ArrayList<>();
            for( int i=0; i<gcStats.size(); i++ ){
                GCStats stats = gcStats.get(i);
                bufferSize += 8;    //Fixed per group entry: 2x int32 -> 8 bytes
                byte[] nameAsBytes = SbeUtil.toBytes(true, stats.gcName);
                bufferSize += nameAsBytes.length;
                gcStatsLabelBytes.add(nameAsBytes);
            }
        }

        //Per parameter stats group length
        bufferSize += 4;    //Group header: always present
        //TODO: need to handle parameters and order properly...
        List<String> params = new ArrayList<>();
        if(histograms != null && histograms.size() > 0){
            params = new ArrayList<>(histograms.get(histograms.keySet().iterator().next()).keySet());
        }
        int nParams = params.size();
        bufferSize += nParams * 10;  //Each parameter entry: has a param ID -> uint16 -> 2 bytes PLUS headers for 2 nested groups: 2*4 = 8 each -> 10 bytes
        for(String s : params){
            //For each parameter: MAY also have a number of summary stats (mean, stdev etc), and histograms (both as nested groups)
            int summaryStatsCount = 0;
            for(StatsType statsType : StatsType.values() ){ //Parameters, updates, activations
                for(SummaryType summaryType : SummaryType.values()){        //Mean, stdev, MM
                    Map<String,Double> map = mapForTypes(statsType, summaryType);
                    if(map == null) continue;
                    if(map.containsKey(s)) summaryStatsCount++;
                }
            }
            //Each summary stat value: StatsType (uint8), SummaryType (uint8), value (double) -> 10 bytes
            bufferSize += summaryStatsCount * 10;

            //Histograms for this parameter
            int nHistograms = histograms.size();    //0, 1 or 2 for each parameter
            //For each histogram: StatsType (uint8) + 2x double + int32 -> 21 bytes PLUS counts group header (4 bytes) -> 25 bytes
            bufferSize += 25 * nHistograms;
            //PLUS, the number of count values, given by nBins...
            int nBinCountEntries = 0;
            for(StatsType statsType : StatsType.values() ){
                if(!histograms.containsKey(statsType)) continue;
                Map<String,Histogram> map = histograms.get(statsType);
                if(map.containsKey(s)){ //If it doesn't: assume 0 count...
                    nBinCountEntries += map.get(s).getNBins();
                }
            }
            bufferSize += 4 * nBinCountEntries; //Each entry: uint32 -> 4 bytes
        }

        //End buffer size calculation

        //--------------------------------------------------------------------------------------------------------------

        //Start encoding

        byte[] bytes = new byte[bufferSize];
        MutableDirectBuffer buffer = new UnsafeBuffer(bytes);
        enc.wrap(buffer, 0)
                .blockLength(ue.sbeBlockLength())
                .templateId(ue.sbeTemplateId())
                .schemaId(ue.sbeSchemaId())
                .version(ue.sbeSchemaVersion());

        int offset = enc.encodedLength();   //Expect 8 bytes
        ue.wrap(buffer, offset);

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

        // +++++ Per Parameter Stats +++++
        int nSummaryStats = 0;
        if(meanValues != null) nSummaryStats += meanValues.size();      //0 to 3 values: parameters, updates, activations
        if(stdevValues != null) nSummaryStats += stdevValues.size();
        if(meanMagnitudeValues != null) nSummaryStats += meanMagnitudeValues.size();

        int nHistograms = (histograms == null ? 0 : histograms.size());
        UpdateEncoder.PerParameterStatsEncoder ppe = ue.perParameterStatsCount(nParams);

        int paramId = 0;
        for(String s : params){
            ppe = ppe.next();
            ppe.paramID(paramId++);
            UpdateEncoder.PerParameterStatsEncoder.SummaryStatEncoder sse = ppe.summaryStatCount(nSummaryStats);

            //Summary stats
            for(StatsType statsType : StatsType.values() ){ //Parameters, updates, activations
                for(SummaryType summaryType : SummaryType.values()){        //Mean, stdev, MM
                    Map<String,Double> map = mapForTypes(statsType, summaryType);
                    if(map == null) continue;
                    appendOrDefault(sse, s, statsType, summaryType, map, Double.NaN);
                }
            }

            //Histograms
            UpdateEncoder.PerParameterStatsEncoder.HistogramsEncoder sshe = ppe.histogramsCount(nHistograms);
            if(nHistograms > 0) {
                for (StatsType statsType : StatsType.values()) {
                    Map<String,Histogram> map = histograms.get(statsType);
                    if(map == null) continue;
                    Histogram h = map.get(s);   //Histogram for StatsType for this parameter
                    double min;
                    double max;
                    int nBins;
                    int[] binCounts;
                    if(h == null){
                        min = 0.0;
                        max = 0.0;
                        nBins = 0;
                        binCounts = null;
                    } else {
                        min = h.getMin();
                        max = h.getMax();
                        nBins = h.getNBins();
                        binCounts = h.getBinCounts();
                    }

                    sshe = sshe.next().minValue(min).maxValue(max).nBins(nBins);
                    UpdateEncoder.PerParameterStatsEncoder.HistogramsEncoder.HistogramCountsEncoder histCountsEncoder = sshe.histogramCountsCount(nBins);
                    for( int i=0; i<nBins; i++ ){
                        int count = (binCounts == null || binCounts.length <= i ? 0 : binCounts[i]);
                        histCountsEncoder.next().binCount(count);
                    }
                }
            }
        }

        offset += ue.encodedLength();
        if(offset != bytes.length){
            throw new RuntimeException();
        }

        return bytes;
    }

    private Map<String,Double> mapForTypes(StatsType statsType, SummaryType summaryType){
        switch (summaryType){
            case Mean:
                if(meanValues == null) return null;
                return meanValues.get(statsType);
            case Stdev:
                if(stdevValues == null) return null;
                return stdevValues.get(statsType);
            case MeanMagnitudes:
                if(meanMagnitudeValues == null) return null;
                return meanMagnitudeValues.get(statsType);
        }
        return null;
    }

    private static void appendOrDefault(UpdateEncoder.PerParameterStatsEncoder.SummaryStatEncoder sse, String param,
                                        StatsType statsType, SummaryType summaryType,
                                        Map<String,Double> map, double defaultValue){
        Double d = map.get(param);
        if(d == null) d = defaultValue;

        org.deeplearning4j.optimize.listeners.stats.sbe.StatsType st;
        switch (statsType){
            case Parameters:
                st = org.deeplearning4j.optimize.listeners.stats.sbe.StatsType.Parameters;
                break;
            case Updates:
                st = org.deeplearning4j.optimize.listeners.stats.sbe.StatsType.Updates;
                break;
            case Activations:
                st = org.deeplearning4j.optimize.listeners.stats.sbe.StatsType.Activations;
                break;
            default:
                throw new RuntimeException("Unknown stats type: " + statsType);
        }
        org.deeplearning4j.optimize.listeners.stats.sbe.SummaryType summaryT;
        switch(summaryType){
            case Mean:
                summaryT = org.deeplearning4j.optimize.listeners.stats.sbe.SummaryType.Mean;
                break;
            case Stdev:
                summaryT = org.deeplearning4j.optimize.listeners.stats.sbe.SummaryType.Stdev;
                break;
            case MeanMagnitudes:
                summaryT = org.deeplearning4j.optimize.listeners.stats.sbe.SummaryType.MeanMagnitude;
                break;
            default:
                throw new RuntimeException("Unknown summary type: " + summaryType);
        }

        sse.next().statType(st)
                .summaryType(summaryT)
                .value(d);
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
