package org.deeplearning4j.ui.stats.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.agrona.DirectBuffer;
import org.agrona.MutableDirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.ui.stats.api.Histogram;
import org.deeplearning4j.ui.stats.api.StatsReport;
import org.deeplearning4j.ui.stats.api.StatsType;
import org.deeplearning4j.ui.stats.api.SummaryType;
import org.deeplearning4j.ui.stats.sbe.*;
import org.deeplearning4j.ui.storage.AgronaPersistable;
import org.nd4j.linalg.primitives.Pair;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;

/**
 * An implementation of {@link StatsReport} using Simple Binary Encoding (SBE)
 *
 * @author Alex Black
 */
@EqualsAndHashCode
@ToString
@Data
public class SbeStatsReport implements StatsReport, AgronaPersistable {
    private String sessionID;
    private String typeID;
    private String workerID;
    private long timeStamp;

    private int iterationCount;
    private int statsCollectionDurationMs;
    private double score;

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

    private Map<String, Double> learningRatesByParam;
    private Map<StatsType, Map<String, Histogram>> histograms;
    private Map<StatsType, Map<String, Double>> meanValues;
    private Map<StatsType, Map<String, Double>> stdevValues;
    private Map<StatsType, Map<String, Double>> meanMagnitudeValues;

    private String metaDataClassName;
    //Store in serialized form; deserialize iff required. Might save us some class not found (or, version) errors, if
    // metadata is saved but is never used
    private List<byte[]> dataSetMetaData;

    private boolean scorePresent;
    private boolean memoryUsePresent;
    private boolean performanceStatsPresent;

    public SbeStatsReport() {
        //No-Arg constructor only for deserialization
    }

    @Override
    public void reportIDs(String sessionID, String typeID, String workerID, long timeStamp) {
        this.sessionID = sessionID;
        this.typeID = typeID;
        this.workerID = workerID;
        this.timeStamp = timeStamp;
    }

    @Override
    public void reportIterationCount(int iterationCount) {
        this.iterationCount = iterationCount;
    }


    @Override
    public void reportStatsCollectionDurationMS(int statsCollectionDurationMS) {
        this.statsCollectionDurationMs = statsCollectionDurationMS;
    }

    @Override
    public void reportScore(double currentScore) {
        this.score = currentScore;
        this.scorePresent = true;
    }

    @Override
    public void reportLearningRates(Map<String, Double> learningRatesByParam) {
        this.learningRatesByParam = learningRatesByParam;
    }

    @Override
    public Map<String, Double> getLearningRates() {
        return this.learningRatesByParam;
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
    public void reportPerformance(long totalRuntimeMs, long totalExamples, long totalMinibatches,
                    double examplesPerSecond, double minibatchesPerSecond) {
        this.totalRuntimeMs = totalRuntimeMs;
        this.totalExamples = totalExamples;
        this.totalMinibatches = totalMinibatches;
        this.examplesPerSecond = examplesPerSecond;
        this.minibatchesPerSecond = minibatchesPerSecond;
        this.performanceStatsPresent = true;
    }

    @Override
    public void reportGarbageCollection(String gcName, int deltaGCCount, int deltaGCTime) {
        if (gcStats == null)
            gcStats = new ArrayList<>();
        gcStats.add(new GCStats(gcName, deltaGCCount, deltaGCTime));
    }

    @Override
    public List<Pair<String, int[]>> getGarbageCollectionStats() {
        if (gcStats == null)
            return null;
        List<Pair<String, int[]>> temp = new ArrayList<>();
        for (GCStats g : gcStats) {
            temp.add(new Pair<>(g.gcName, new int[] {g.getDeltaGCCount(), g.getDeltaGCTime()}));
        }
        return temp;
    }

    @Override
    public void reportHistograms(StatsType statsType, Map<String, Histogram> histogram) {
        if (this.histograms == null)
            this.histograms = new HashMap<>();
        this.histograms.put(statsType, histogram);
    }

    @Override
    public Map<String, Histogram> getHistograms(StatsType statsType) {
        if (histograms == null)
            return null;
        return histograms.get(statsType);
    }

    @Override
    public void reportMean(StatsType statsType, Map<String, Double> mean) {
        if (this.meanValues == null)
            this.meanValues = new HashMap<>();
        this.meanValues.put(statsType, mean);
    }

    @Override
    public Map<String, Double> getMean(StatsType statsType) {
        if (this.meanValues == null)
            return null;
        return meanValues.get(statsType);
    }

    @Override
    public void reportStdev(StatsType statsType, Map<String, Double> stdev) {
        if (this.stdevValues == null)
            this.stdevValues = new HashMap<>();
        this.stdevValues.put(statsType, stdev);
    }

    @Override
    public Map<String, Double> getStdev(StatsType statsType) {
        if (this.stdevValues == null)
            return null;
        return stdevValues.get(statsType);
    }

    @Override
    public void reportMeanMagnitudes(StatsType statsType, Map<String, Double> meanMagnitudes) {
        if (this.meanMagnitudeValues == null)
            this.meanMagnitudeValues = new HashMap<>();
        this.meanMagnitudeValues.put(statsType, meanMagnitudes);
    }

    @Override
    public void reportDataSetMetaData(List<Serializable> dataSetMetaData, Class<?> metaDataClass) {
        reportDataSetMetaData(dataSetMetaData, (metaDataClass == null ? null : metaDataClass.getName()));
    }

    @Override
    public void reportDataSetMetaData(List<Serializable> dataSetMetaData, String metaDataClass) {
        if (dataSetMetaData != null) {
            this.dataSetMetaData = new ArrayList<>();
            for (Serializable s : dataSetMetaData) {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
                    oos.writeObject(s);
                    oos.flush();
                    oos.close();
                } catch (IOException e) {
                    throw new RuntimeException("Unexpected IOException from ByteArrayOutputStream", e);
                }
                byte[] b = baos.toByteArray();
                this.dataSetMetaData.add(b);
            }
        } else {
            this.dataSetMetaData = null;
        }
        this.metaDataClassName = metaDataClass;
    }

    @Override
    public Map<String, Double> getMeanMagnitudes(StatsType statsType) {
        if (this.meanMagnitudeValues == null)
            return null;
        return this.meanMagnitudeValues.get(statsType);
    }

    @Override
    public List<Serializable> getDataSetMetaData() {
        if (dataSetMetaData == null || dataSetMetaData.isEmpty())
            return null;

        List<Serializable> l = new ArrayList<>();
        for (byte[] b : dataSetMetaData) {
            try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(b))) {
                l.add((Serializable) ois.readObject());
            } catch (IOException | ClassNotFoundException e) {
                throw new RuntimeException(e);
            }
        }
        return l;
    }

    @Override
    public String getDataSetMetaDataClassName() {
        return metaDataClassName;
    }

    @Override
    public boolean hasScore() {
        return scorePresent;
    }

    @Override
    public boolean hasLearningRates() {
        return learningRatesByParam != null;
    }

    @Override
    public boolean hasMemoryUse() {
        return memoryUsePresent;
    }

    @Override
    public boolean hasPerformance() {
        return performanceStatsPresent;
    }

    @Override
    public boolean hasGarbageCollection() {
        return gcStats != null && !gcStats.isEmpty();
    }

    @Override
    public boolean hasHistograms(StatsType statsType) {
        if (histograms == null)
            return false;
        return histograms.containsKey(statsType);
    }

    @Override
    public boolean hasSummaryStats(StatsType statsType, SummaryType summaryType) {
        switch (summaryType) {
            case Mean:
                return meanValues != null && meanValues.containsKey(statsType);
            case Stdev:
                return stdevValues != null && stdevValues.containsKey(statsType);
            case MeanMagnitudes:
                return meanMagnitudeValues != null && meanMagnitudeValues.containsKey(statsType);
        }
        return false;
    }

    @Override
    public boolean hasDataSetMetaData() {
        return dataSetMetaData != null || metaDataClassName != null;
    }

    private Map<String, Double> mapForTypes(StatsType statsType, SummaryType summaryType) {
        switch (summaryType) {
            case Mean:
                if (meanValues == null)
                    return null;
                return meanValues.get(statsType);
            case Stdev:
                if (stdevValues == null)
                    return null;
                return stdevValues.get(statsType);
            case MeanMagnitudes:
                if (meanMagnitudeValues == null)
                    return null;
                return meanMagnitudeValues.get(statsType);
        }
        return null;
    }

    private static void appendOrDefault(UpdateEncoder.PerParameterStatsEncoder.SummaryStatEncoder sse, String param,
                    StatsType statsType, SummaryType summaryType, Map<String, Double> map, double defaultValue) {
        Double d = map.get(param);
        if (d == null)
            d = defaultValue;

        org.deeplearning4j.ui.stats.sbe.StatsType st;
        switch (statsType) {
            case Parameters:
                st = org.deeplearning4j.ui.stats.sbe.StatsType.Parameters;
                break;
            case Gradients:
                st = org.deeplearning4j.ui.stats.sbe.StatsType.Gradients;
                break;
            case Updates:
                st = org.deeplearning4j.ui.stats.sbe.StatsType.Updates;
                break;
            case Activations:
                st = org.deeplearning4j.ui.stats.sbe.StatsType.Activations;
                break;
            default:
                throw new RuntimeException("Unknown stats type: " + statsType);
        }
        org.deeplearning4j.ui.stats.sbe.SummaryType summaryT;
        switch (summaryType) {
            case Mean:
                summaryT = org.deeplearning4j.ui.stats.sbe.SummaryType.Mean;
                break;
            case Stdev:
                summaryT = org.deeplearning4j.ui.stats.sbe.SummaryType.Stdev;
                break;
            case MeanMagnitudes:
                summaryT = org.deeplearning4j.ui.stats.sbe.SummaryType.MeanMagnitude;
                break;
            default:
                throw new RuntimeException("Unknown summary type: " + summaryType);
        }
        sse.next().statType(st).summaryType(summaryT).value(d);
    }

    private static StatsType translate(org.deeplearning4j.ui.stats.sbe.StatsType statsType) {
        switch (statsType) {
            case Parameters:
                return StatsType.Parameters;
            case Gradients:
                return StatsType.Gradients;
            case Updates:
                return StatsType.Updates;
            case Activations:
                return StatsType.Activations;
            default:
                throw new RuntimeException("Unknown stats type: " + statsType);
        }
    }

    private static org.deeplearning4j.ui.stats.sbe.StatsType translate(StatsType statsType) {
        switch (statsType) {
            case Parameters:
                return org.deeplearning4j.ui.stats.sbe.StatsType.Parameters;
            case Gradients:
                return org.deeplearning4j.ui.stats.sbe.StatsType.Gradients;
            case Updates:
                return org.deeplearning4j.ui.stats.sbe.StatsType.Updates;
            case Activations:
                return org.deeplearning4j.ui.stats.sbe.StatsType.Activations;
            default:
                throw new RuntimeException("Unknown stats type: " + statsType);
        }
    }

    private static SummaryType translate(org.deeplearning4j.ui.stats.sbe.SummaryType summaryType) {
        switch (summaryType) {
            case Mean:
                return SummaryType.Mean;
            case Stdev:
                return SummaryType.Stdev;
            case MeanMagnitude:
                return SummaryType.MeanMagnitudes;
            default:
                throw new RuntimeException("Unknown summary type: " + summaryType);
        }
    }

    @Override
    public String getSessionID() {
        return sessionID;
    }

    @Override
    public String getTypeID() {
        return typeID;
    }

    @Override
    public String getWorkerID() {
        return workerID;
    }

    @Override
    public long getTimeStamp() {
        return timeStamp;
    }


    //================ Ser/de methods =================

    @Override
    public int encodingLengthBytes() {
        //TODO convert Strings to byte[] only once

        //First: determine buffer size.
        //(a) Header: 8 bytes (4x uint16 = 8 bytes)
        //(b) Fixed length entries length (sie.BlockLength())
        //(c) Group 1: Memory use.
        //(d) Group 2: Performance stats
        //(e) Group 3: GC stats
        //(f) Group 4: param names (variable length strings)
        //(g) Group 5: layer names (variable length strings)
        //(g) Group 6: Per parameter performance stats
        //Variable length String fields: 4 - session/type/worker IDs and metadata -> 4*4=16 bytes header, plus content

        UpdateEncoder ue = new UpdateEncoder();
        int bufferSize = 8 + ue.sbeBlockLength() + 16;

        //Memory use group length...
        int memoryUseCount;
        if (!memoryUsePresent) {
            memoryUseCount = 0;
        } else {
            memoryUseCount = 4 + (deviceCurrentBytes == null ? 0 : deviceCurrentBytes.length)
                            + (deviceMaxBytes == null ? 0 : deviceMaxBytes.length);
        }
        bufferSize += 4 + 9 * memoryUseCount; //Group header: 4 bytes (always present); Each entry in group - 1x MemoryType (uint8) + 1x int64 -> 1+8 = 9 bytes

        //Performance group length
        bufferSize += 4 + (performanceStatsPresent ? 32 : 0); //Group header: 4 bytes (always present); Only 1 group: 3xint64 + 2xfloat = 32 bytes

        //GC stats group length
        bufferSize += 4; //Group header: always present
        List<byte[]> gcStatsLabelBytes = null;
        if (gcStats != null && !gcStats.isEmpty()) {
            gcStatsLabelBytes = new ArrayList<>();
            for (int i = 0; i < gcStats.size(); i++) {
                GCStats stats = gcStats.get(i);
                bufferSize += 12; //Fixed per group entry: 2x int32 -> 8 bytes PLUS the header for the variable length GC name: another 4 bytes
                byte[] nameAsBytes = SbeUtil.toBytes(true, stats.gcName);
                bufferSize += nameAsBytes.length;
                gcStatsLabelBytes.add(nameAsBytes);
            }
        }

        //Param names group
        bufferSize += 4; //Header; always present
        List<String> paramNames = getParamNames();
        for (String s : paramNames) {
            bufferSize += 4; //header for each entry
            bufferSize += SbeUtil.toBytes(true, s).length; //Content
        }

        //Layer names group
        bufferSize += 4; //Header; always present
        List<String> layerNames = getlayerNames();
        for (String s : layerNames) {
            bufferSize += 4;
            bufferSize += SbeUtil.toBytes(true, s).length; //Content
        }

        //Per parameter and per layer (activations) stats group length
        bufferSize += 4; //Per parameter/layer stats group header: always present
        int nEntries = paramNames.size() + layerNames.size();
        bufferSize += nEntries * 12; //Each parameter/layer entry: has learning rate -> float -> 4 bytes PLUS headers for 2 nested groups: 2*4 = 8 each -> 12 bytes total
        bufferSize += entrySize(paramNames, StatsType.Parameters, StatsType.Gradients, StatsType.Updates);
        bufferSize += entrySize(layerNames, StatsType.Activations);

        //Metadata group:
        bufferSize += 4; //Metadata group header: always present
        if (dataSetMetaData != null && !dataSetMetaData.isEmpty()) {
            for (byte[] b : dataSetMetaData) {
                bufferSize += 4 + b.length; //4 bytes header + content
            }
        }

        //Session/worker IDs
        byte[] bSessionID = SbeUtil.toBytes(true, sessionID);
        byte[] bTypeID = SbeUtil.toBytes(true, typeID);
        byte[] bWorkerID = SbeUtil.toBytes(true, workerID);
        bufferSize += bSessionID.length + bTypeID.length + bWorkerID.length;

        //Metadata class name:
        byte[] metaDataClassNameBytes = SbeUtil.toBytes(true, metaDataClassName);
        bufferSize += metaDataClassNameBytes.length;

        return bufferSize;
    }

    private int entrySize(List<String> entryNames, StatsType... statsTypes) {
        int bufferSize = 0;
        for (String s : entryNames) {
            //For each parameter: MAY also have a number of summary stats (mean, stdev etc), and histograms (both as nested groups)
            int summaryStatsCount = 0;
            for (StatsType statsType : statsTypes) { //Parameters, Gradients, updates, activations
                for (SummaryType summaryType : SummaryType.values()) { //Mean, stdev, MM
                    Map<String, Double> map = mapForTypes(statsType, summaryType);
                    if (map == null)
                        continue;
                    summaryStatsCount++;
                }
            }
            //Each summary stat value: StatsType (uint8), SummaryType (uint8), value (double) -> 1+1+8 = 10 bytes
            bufferSize += summaryStatsCount * 10;

            //Histograms for this parameter
            int nHistogramsThisParam = 0;
            if (histograms != null && histograms.size() > 0) {
                for (Map<String, Histogram> map : histograms.values()) {
                    if (map != null && map.containsKey(s))
                        nHistogramsThisParam++;
                }
            }
            //For each histogram: StatsType (uint8) + 2x double + int32 -> 1 + 2*8 + 4 = 21 bytes PLUS counts group header (4 bytes) -> 25 bytes fixed per histogram
            bufferSize += 25 * nHistogramsThisParam;
            //PLUS, the number of count values, given by nBins...
            int nBinCountEntries = 0;
            for (StatsType statsType : statsTypes) {
                if (histograms == null || !histograms.containsKey(statsType))
                    continue;
                Map<String, Histogram> map = histograms.get(statsType);
                if (map != null && map.containsKey(s)) { //If it doesn't: assume 0 count...
                    nBinCountEntries += map.get(s).getNBins();
                }
            }
            bufferSize += 4 * nBinCountEntries; //Each entry: uint32 -> 4 bytes
        }
        return bufferSize;
    }

    private List<String> getParamNames() {
        Set<String> paramNames = new LinkedHashSet<>();
        if (learningRatesByParam != null)
            paramNames.addAll(learningRatesByParam.keySet());
        if (histograms != null) {
            addToSet(paramNames, histograms.get(StatsType.Parameters));
            addToSet(paramNames, histograms.get(StatsType.Gradients));
            addToSet(paramNames, histograms.get(StatsType.Updates));
        }
        if (meanValues != null) {
            addToSet(paramNames, meanValues.get(StatsType.Parameters));
            addToSet(paramNames, meanValues.get(StatsType.Gradients));
            addToSet(paramNames, meanValues.get(StatsType.Updates));
        }
        if (stdevValues != null) {
            addToSet(paramNames, stdevValues.get(StatsType.Parameters));
            addToSet(paramNames, stdevValues.get(StatsType.Gradients));
            addToSet(paramNames, stdevValues.get(StatsType.Updates));
        }
        if (meanMagnitudeValues != null) {
            addToSet(paramNames, meanMagnitudeValues.get(StatsType.Parameters));
            addToSet(paramNames, meanMagnitudeValues.get(StatsType.Gradients));
            addToSet(paramNames, meanMagnitudeValues.get(StatsType.Updates));
        }
        return new ArrayList<>(paramNames);
    }

    private List<String> getlayerNames() {
        Set<String> layerNames = new LinkedHashSet<>();
        if (histograms != null) {
            addToSet(layerNames, histograms.get(StatsType.Activations));
        }
        if (meanValues != null) {
            addToSet(layerNames, meanValues.get(StatsType.Activations));
        }
        if (stdevValues != null) {
            addToSet(layerNames, stdevValues.get(StatsType.Activations));
        }
        if (meanMagnitudeValues != null) {
            addToSet(layerNames, meanMagnitudeValues.get(StatsType.Activations));
        }
        return new ArrayList<>(layerNames);
    }

    private void addToSet(Set<String> set, Map<String, ?> map) {
        if (map == null)
            return;
        set.addAll(map.keySet());
    }

    @Override
    public byte[] encode() {
        byte[] bytes = new byte[encodingLengthBytes()];
        MutableDirectBuffer buffer = new UnsafeBuffer(bytes);
        encode(buffer);
        return bytes;
    }

    @Override
    public void encode(ByteBuffer buffer) {
        encode(new UnsafeBuffer(buffer));
    }

    @Override
    public void encode(MutableDirectBuffer buffer) {
        MessageHeaderEncoder enc = new MessageHeaderEncoder();
        UpdateEncoder ue = new UpdateEncoder();

        enc.wrap(buffer, 0).blockLength(ue.sbeBlockLength()).templateId(ue.sbeTemplateId()).schemaId(ue.sbeSchemaId())
                        .version(ue.sbeSchemaVersion());

        int offset = enc.encodedLength(); //Expect 8 bytes
        ue.wrap(buffer, offset);

        //Fixed length fields: always encoded
        ue.time(timeStamp).deltaTime(0) //TODO
                        .iterationCount(iterationCount).fieldsPresent().score(scorePresent).memoryUse(memoryUsePresent)
                        .performance(performanceStatsPresent).garbageCollection(gcStats != null && !gcStats.isEmpty())
                        .histogramParameters(histograms != null && histograms.containsKey(StatsType.Parameters))
                        .histogramActivations(histograms != null && histograms.containsKey(StatsType.Gradients))
                        .histogramUpdates(histograms != null && histograms.containsKey(StatsType.Updates))
                        .histogramActivations(histograms != null && histograms.containsKey(StatsType.Activations))
                        .meanParameters(meanValues != null && meanValues.containsKey(StatsType.Parameters))
                        .meanGradients(meanValues != null && meanValues.containsKey(StatsType.Gradients))
                        .meanUpdates(meanValues != null && meanValues.containsKey(StatsType.Updates))
                        .meanActivations(meanValues != null && meanValues.containsKey(StatsType.Activations))
                        .meanMagnitudeParameters(meanMagnitudeValues != null
                                        && meanMagnitudeValues.containsKey(StatsType.Parameters))
                        .meanMagnitudeGradients(meanMagnitudeValues != null
                                        && meanMagnitudeValues.containsKey(StatsType.Gradients))
                        .meanMagnitudeUpdates(meanMagnitudeValues != null
                                        && meanMagnitudeValues.containsKey(StatsType.Updates))
                        .meanMagnitudeActivations(meanMagnitudeValues != null
                                        && meanMagnitudeValues.containsKey(StatsType.Activations))
                        .learningRatesPresent(learningRatesByParam != null)
                        .dataSetMetaDataPresent(hasDataSetMetaData());

        ue.statsCollectionDuration(statsCollectionDurationMs).score(score);

        int memoryUseCount;
        if (!memoryUsePresent) {
            memoryUseCount = 0;
        } else {
            memoryUseCount = 4 + (deviceCurrentBytes == null ? 0 : deviceCurrentBytes.length)
                            + (deviceMaxBytes == null ? 0 : deviceMaxBytes.length);
        }

        UpdateEncoder.MemoryUseEncoder mue = ue.memoryUseCount(memoryUseCount);
        if (memoryUsePresent) {
            mue.next().memoryType(MemoryType.JvmCurrent).memoryBytes(jvmCurrentBytes).next()
                            .memoryType(MemoryType.JvmMax).memoryBytes(jvmMaxBytes).next()
                            .memoryType(MemoryType.OffHeapCurrent).memoryBytes(offHeapCurrentBytes).next()
                            .memoryType(MemoryType.OffHeapMax).memoryBytes(offHeapMaxBytes);
            if (deviceCurrentBytes != null) {
                for (int i = 0; i < deviceCurrentBytes.length; i++) {
                    mue.next().memoryType(MemoryType.DeviceCurrent).memoryBytes(deviceCurrentBytes[i]);
                }
            }
            if (deviceMaxBytes != null) {
                for (int i = 0; i < deviceMaxBytes.length; i++) {
                    mue.next().memoryType(MemoryType.DeviceMax).memoryBytes(deviceMaxBytes[i]);
                }
            }
        }

        UpdateEncoder.PerformanceEncoder pe = ue.performanceCount(performanceStatsPresent ? 1 : 0);
        if (performanceStatsPresent) {
            pe.next().totalRuntimeMs(totalRuntimeMs).totalExamples(totalExamples).totalMinibatches(totalMinibatches)
                            .examplesPerSecond((float) examplesPerSecond)
                            .minibatchesPerSecond((float) minibatchesPerSecond);
        }

        UpdateEncoder.GcStatsEncoder gce = ue.gcStatsCount(gcStats == null || gcStats.isEmpty() ? 0 : gcStats.size());
        List<byte[]> gcStatsLabelBytes = null;
        if (gcStats != null && !gcStats.isEmpty()) {
            gcStatsLabelBytes = new ArrayList<>();
            for (GCStats stats : gcStats) {
                byte[] nameAsBytes = SbeUtil.toBytes(true, stats.gcName);
                gcStatsLabelBytes.add(nameAsBytes);
            }
        }
        if (gcStats != null && !gcStats.isEmpty()) {
            int i = 0;
            for (GCStats g : gcStats) {
                byte[] gcLabelBytes = gcStatsLabelBytes.get(i++);
                gce.next().deltaGCCount(g.deltaGCCount).deltaGCTimeMs(g.deltaGCTime).putGcName(gcLabelBytes, 0,
                                gcLabelBytes.length);
            }
        }

        //Param names
        List<String> paramNames = getParamNames();
        UpdateEncoder.ParamNamesEncoder pne = ue.paramNamesCount(paramNames.size());
        for (String s : paramNames) {
            pne.next().paramName(s);
        }

        //Layer names
        List<String> layerNames = getlayerNames();
        UpdateEncoder.LayerNamesEncoder lne = ue.layerNamesCount(layerNames.size());
        for (String s : layerNames) {
            lne.next().layerName(s);
        }

        // +++++ Per Parameter Stats +++++
        UpdateEncoder.PerParameterStatsEncoder ppe = ue.perParameterStatsCount(paramNames.size() + layerNames.size());
        StatsType[] st = new StatsType[] {StatsType.Parameters, StatsType.Gradients, StatsType.Updates};
        for (String s : paramNames) {
            ppe = ppe.next();
            float lr = 0.0f;
            if (learningRatesByParam != null && learningRatesByParam.containsKey(s)) {
                lr = learningRatesByParam.get(s).floatValue();
            }
            ppe.learningRate(lr);

            int summaryStatsCount = 0;
            for (StatsType statsType : st) { //Parameters, updates
                for (SummaryType summaryType : SummaryType.values()) { //Mean, stdev, MM
                    Map<String, Double> map = mapForTypes(statsType, summaryType);
                    if (map == null || map.size() == 0)
                        continue;
                    summaryStatsCount++;
                }
            }

            UpdateEncoder.PerParameterStatsEncoder.SummaryStatEncoder sse = ppe.summaryStatCount(summaryStatsCount);

            //Summary stats
            for (StatsType statsType : st) { //Parameters, updates
                for (SummaryType summaryType : SummaryType.values()) { //Mean, stdev, MM
                    Map<String, Double> map = mapForTypes(statsType, summaryType);
                    if (map == null || map.size() == 0)
                        continue;
                    appendOrDefault(sse, s, statsType, summaryType, map, Double.NaN);
                }
            }

            int nHistogramsThisParam = 0;
            if (histograms != null && histograms.size() > 0) {
                for (StatsType statsType : st) { //Parameters, updates
                    Map<String, Histogram> map = histograms.get(statsType);
                    if (map == null)
                        continue;
                    if (map.containsKey(s))
                        nHistogramsThisParam++;
                }
            }



            //Histograms
            UpdateEncoder.PerParameterStatsEncoder.HistogramsEncoder sshe = ppe.histogramsCount(nHistogramsThisParam);
            if (nHistogramsThisParam > 0) {
                for (StatsType statsType : st) {
                    Map<String, Histogram> map = histograms.get(statsType);
                    if (map == null || !map.containsKey(s))
                        continue;
                    Histogram h = map.get(s); //Histogram for StatsType for this parameter
                    double min;
                    double max;
                    int nBins;
                    int[] binCounts;
                    if (h == null) {
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

                    sshe = sshe.next().statType(translate(statsType)).minValue(min).maxValue(max).nBins(nBins);
                    UpdateEncoder.PerParameterStatsEncoder.HistogramsEncoder.HistogramCountsEncoder histCountsEncoder =
                                    sshe.histogramCountsCount(nBins);
                    for (int i = 0; i < nBins; i++) {
                        int count = (binCounts == null || binCounts.length <= i ? 0 : binCounts[i]);
                        histCountsEncoder.next().binCount(count);
                    }
                }
            }
        }

        for (String s : layerNames) {
            ppe = ppe.next();
            ppe.learningRate(0.0f); //Not applicable

            int summaryStatsCount = 0;
            for (SummaryType summaryType : SummaryType.values()) { //Mean, stdev, MM
                Map<String, Double> map = mapForTypes(StatsType.Activations, summaryType);
                if (map == null || map.size() == 0)
                    continue;
                if (map.containsKey(s))
                    summaryStatsCount++;
            }

            UpdateEncoder.PerParameterStatsEncoder.SummaryStatEncoder sse = ppe.summaryStatCount(summaryStatsCount);

            //Summary stats
            for (SummaryType summaryType : SummaryType.values()) { //Mean, stdev, MM
                Map<String, Double> map = mapForTypes(StatsType.Activations, summaryType);
                if (map == null || map.size() == 0)
                    continue;
                appendOrDefault(sse, s, StatsType.Activations, summaryType, map, Double.NaN);
            }

            int nHistogramsThisLayer = 0;
            if (histograms != null && histograms.size() > 0) {
                for (Map<String, Histogram> map : histograms.values()) {
                    if (map != null && map.containsKey(s))
                        nHistogramsThisLayer++;
                }
            }

            //Histograms
            UpdateEncoder.PerParameterStatsEncoder.HistogramsEncoder sshe = ppe.histogramsCount(nHistogramsThisLayer);
            if (nHistogramsThisLayer > 0) {
                Map<String, Histogram> map = histograms.get(StatsType.Activations);
                if (map == null || !map.containsKey(s))
                    continue;
                Histogram h = map.get(s); //Histogram for StatsType for this parameter
                double min;
                double max;
                int nBins;
                int[] binCounts;
                if (h == null) {
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

                sshe = sshe.next().statType(translate(StatsType.Activations)).minValue(min).maxValue(max).nBins(nBins);
                UpdateEncoder.PerParameterStatsEncoder.HistogramsEncoder.HistogramCountsEncoder histCountsEncoder =
                                sshe.histogramCountsCount(nBins);
                for (int i = 0; i < nBins; i++) {
                    int count = (binCounts == null || binCounts.length <= i ? 0 : binCounts[i]);
                    histCountsEncoder.next().binCount(count);
                }
            }
        }

        // +++ DataSet MetaData +++
        UpdateEncoder.DataSetMetaDataBytesEncoder metaEnc =
                        ue.dataSetMetaDataBytesCount(dataSetMetaData != null ? dataSetMetaData.size() : 0);
        if (dataSetMetaData != null && !dataSetMetaData.isEmpty()) {
            for (byte[] b : dataSetMetaData) {
                metaEnc = metaEnc.next();
                UpdateEncoder.DataSetMetaDataBytesEncoder.MetaDataBytesEncoder mdbe =
                                metaEnc.metaDataBytesCount(b.length);
                for (byte bb : b) {
                    mdbe.next().bytes(bb);
                }
            }
        }

        //Session/worker IDs
        byte[] bSessionID = SbeUtil.toBytes(true, sessionID);
        byte[] bTypeID = SbeUtil.toBytes(true, typeID);
        byte[] bWorkerID = SbeUtil.toBytes(true, workerID);
        ue.putSessionID(bSessionID, 0, bSessionID.length);
        ue.putTypeID(bTypeID, 0, bTypeID.length);
        ue.putWorkerID(bWorkerID, 0, bWorkerID.length);

        //Class name for DataSet metadata
        byte[] metaDataClassNameBytes = SbeUtil.toBytes(true, metaDataClassName);
        ue.putDataSetMetaDataClassName(metaDataClassNameBytes, 0, metaDataClassNameBytes.length);
    }

    @Override
    public void encode(OutputStream outputStream) throws IOException {
        //TODO there may be more efficient way of doing this
        outputStream.write(encode());
    }

    @Override
    public void decode(byte[] decode) {
        MutableDirectBuffer buffer = new UnsafeBuffer(decode);
        decode(buffer);
    }

    @Override
    public void decode(ByteBuffer buffer) {
        decode(new UnsafeBuffer(buffer));
    }

    @Override
    public void decode(DirectBuffer buffer) {
        //TODO we could do this more efficiently, with buffer re-use, etc.
        MessageHeaderDecoder dec = new MessageHeaderDecoder();
        UpdateDecoder ud = new UpdateDecoder();
        dec.wrap(buffer, 0);

        final int blockLength = dec.blockLength();
        final int version = dec.version();

        int headerLength = dec.encodedLength();
        //TODO: in general, we'd check the header, version, schema etc.

        ud.wrap(buffer, headerLength, blockLength, version);

        //TODO iteration count
        timeStamp = ud.time();
        long deltaTime = ud.deltaTime(); //TODO
        iterationCount = ud.iterationCount();

        UpdateFieldsPresentDecoder fpd = ud.fieldsPresent();
        scorePresent = fpd.score();
        memoryUsePresent = fpd.memoryUse();
        performanceStatsPresent = fpd.performance();
        boolean gc = fpd.garbageCollection();
        boolean histogramParameters = fpd.histogramParameters();
        boolean histogramUpdates = fpd.histogramUpdates();
        boolean histogramActivations = fpd.histogramActivations();
        boolean meanParameters = fpd.meanParameters();
        boolean meanUpdates = fpd.meanUpdates();
        boolean meanActivations = fpd.meanActivations();
        boolean meanMagParams = fpd.meanMagnitudeParameters();
        boolean meanMagUpdates = fpd.meanMagnitudeUpdates();
        boolean meanMagAct = fpd.meanMagnitudeActivations();
        boolean learningRatesPresent = fpd.learningRatesPresent();
        boolean metaDataPresent = fpd.dataSetMetaDataPresent();

        statsCollectionDurationMs = ud.statsCollectionDuration();
        score = ud.score();

        //First group: memory use
        UpdateDecoder.MemoryUseDecoder mud = ud.memoryUse();
        List<Long> dcMem = null; //TODO avoid
        List<Long> dmMem = null;
        for (UpdateDecoder.MemoryUseDecoder m : mud) {
            MemoryType type = m.memoryType();
            long memBytes = m.memoryBytes();
            switch (type) {
                case JvmCurrent:
                    jvmCurrentBytes = memBytes;
                    break;
                case JvmMax:
                    jvmMaxBytes = memBytes;
                    break;
                case OffHeapCurrent:
                    offHeapCurrentBytes = memBytes;
                    break;
                case OffHeapMax:
                    offHeapMaxBytes = memBytes;
                    break;
                case DeviceCurrent:
                    if (dcMem == null)
                        dcMem = new ArrayList<>();
                    dcMem.add(memBytes);
                    break;
                case DeviceMax:
                    if (dmMem == null)
                        dmMem = new ArrayList<>();
                    dmMem.add(memBytes);
                    break;
                case NULL_VAL:
                    break;
            }
        }
        if (dcMem != null) {
            long[] a = new long[dcMem.size()];
            int i = 0;
            for (Long l : dcMem) {
                a[i++] = l;
            }
            deviceCurrentBytes = a;
        }
        if (dmMem != null) {
            long[] a = new long[dmMem.size()];
            int i = 0;
            for (Long l : dmMem) {
                a[i++] = l;
            }
            deviceMaxBytes = a;
        }

        //Second group: performance stats (0 or 1 entries only)
        for (UpdateDecoder.PerformanceDecoder pd : ud.performance()) {
            totalRuntimeMs = pd.totalRuntimeMs();
            totalExamples = pd.totalExamples();
            totalMinibatches = pd.totalMinibatches();
            examplesPerSecond = pd.examplesPerSecond();
            minibatchesPerSecond = pd.minibatchesPerSecond();
        }

        //Third group: GC stats
        for (UpdateDecoder.GcStatsDecoder gcsd : ud.gcStats()) {
            if (gcStats == null)
                gcStats = new ArrayList<>();
            int deltaGCCount = gcsd.deltaGCCount();
            int deltaGCTimeMs = gcsd.deltaGCTimeMs();
            String gcName = gcsd.gcName();
            GCStats s = new GCStats(gcName, deltaGCCount, deltaGCTimeMs); //TODO delta time...
            gcStats.add(s);
        }

        //Fourth group: param names
        UpdateDecoder.ParamNamesDecoder pnd = ud.paramNames();
        int nParams = pnd.count();
        List<String> paramNames = null;
        if (nParams > 0) {
            paramNames = new ArrayList<>(nParams);
        }
        for (UpdateDecoder.ParamNamesDecoder pndec : pnd) {
            paramNames.add(pndec.paramName());
        }

        //Fifth group: layer names
        UpdateDecoder.LayerNamesDecoder lnd = ud.layerNames();
        int nLayers = lnd.count();
        List<String> layerNames = null;
        if (nLayers > 0) {
            layerNames = new ArrayList<>(nLayers);
        }
        for (UpdateDecoder.LayerNamesDecoder l : lnd) {
            layerNames.add(l.layerName());
        }


        //Sixth group: Per parameter stats (and histograms, etc) AND per layer stats
        int entryNum = 0;
        for (UpdateDecoder.PerParameterStatsDecoder ppsd : ud.perParameterStats()) {
            boolean isParam = entryNum < nParams;
            String name = (isParam ? paramNames.get(entryNum) : layerNames.get(entryNum - nParams));
            entryNum++;

            float lr = ppsd.learningRate();

            if (learningRatesPresent && isParam) {
                if (learningRatesByParam == null)
                    learningRatesByParam = new HashMap<>();
                learningRatesByParam.put(name, (double) lr);
            }

            //Summary stats (mean/stdev/mean magnitude)
            for (UpdateDecoder.PerParameterStatsDecoder.SummaryStatDecoder ssd : ppsd.summaryStat()) {
                StatsType st = translate(ssd.statType());
                SummaryType summaryType = translate(ssd.summaryType());
                double value = ssd.value();

                switch (summaryType) {
                    case Mean:
                        if (meanValues == null)
                            meanValues = new HashMap<>();
                        Map<String, Double> map = meanValues.get(st);
                        if (map == null) {
                            map = new HashMap<>();
                            meanValues.put(st, map);
                        }
                        map.put(name, value);
                        break;
                    case Stdev:
                        if (stdevValues == null)
                            stdevValues = new HashMap<>();
                        Map<String, Double> map2 = stdevValues.get(st);
                        if (map2 == null) {
                            map2 = new HashMap<>();
                            stdevValues.put(st, map2);
                        }
                        map2.put(name, value);
                        break;
                    case MeanMagnitudes:
                        if (meanMagnitudeValues == null)
                            meanMagnitudeValues = new HashMap<>();
                        Map<String, Double> map3 = meanMagnitudeValues.get(st);
                        if (map3 == null) {
                            map3 = new HashMap<>();
                            meanMagnitudeValues.put(st, map3);
                        }
                        map3.put(name, value);
                        break;
                }
            }

            //Histograms
            for (UpdateDecoder.PerParameterStatsDecoder.HistogramsDecoder hd : ppsd.histograms()) {
                StatsType st = translate(hd.statType());
                double min = hd.minValue();
                double max = hd.maxValue();
                int nBins = hd.nBins();
                int[] binCounts = new int[nBins];
                int i = 0;
                for (UpdateDecoder.PerParameterStatsDecoder.HistogramsDecoder.HistogramCountsDecoder hcd : hd
                                .histogramCounts()) {
                    binCounts[i++] = (int) hcd.binCount();
                }

                Histogram h = new Histogram(min, max, nBins, binCounts);
                if (histograms == null)
                    histograms = new HashMap<>();
                Map<String, Histogram> map = histograms.get(st);
                if (map == null) {
                    map = new HashMap<>();
                    histograms.put(st, map);
                }
                map.put(name, h);
            }
        }

        //Final group: DataSet metadata
        for (UpdateDecoder.DataSetMetaDataBytesDecoder metaDec : ud.dataSetMetaDataBytes()) {
            if (this.dataSetMetaData == null)
                this.dataSetMetaData = new ArrayList<>();
            UpdateDecoder.DataSetMetaDataBytesDecoder.MetaDataBytesDecoder mdbd = metaDec.metaDataBytes();
            int length = mdbd.count();
            byte[] b = new byte[length];
            int i = 0;
            for (UpdateDecoder.DataSetMetaDataBytesDecoder.MetaDataBytesDecoder mdbd2 : mdbd) {
                b[i++] = mdbd2.bytes();
            }
            this.dataSetMetaData.add(b);
        }

        //IDs
        this.sessionID = ud.sessionID();
        this.typeID = ud.typeID();
        this.workerID = ud.workerID();

        //Variable length: DataSet metadata class name
        this.metaDataClassName = ud.dataSetMetaDataClassName();
        if (!metaDataPresent) {
            this.metaDataClassName = null;
        }
    }

    @Override
    public void decode(InputStream inputStream) throws IOException {
        byte[] bytes = IOUtils.toByteArray(inputStream);
        decode(bytes);
    }


    @AllArgsConstructor
    @Data
    private static class GCStats implements Serializable {
        private String gcName;
        private int deltaGCCount;
        private int deltaGCTime;
    }
}
