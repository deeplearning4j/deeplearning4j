package org.deeplearning4j.ui.stats.impl.java;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.ui.stats.api.Histogram;
import org.deeplearning4j.ui.stats.api.StatsReport;
import org.deeplearning4j.ui.stats.api.StatsType;
import org.deeplearning4j.ui.stats.api.SummaryType;
import org.nd4j.linalg.primitives.Pair;

import java.io.*;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Alex on 14/12/2016.
 */
@EqualsAndHashCode
@ToString
@Data
public class JavaStatsReport implements StatsReport {

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

    public JavaStatsReport() {
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
    public Map<String, Double> getLearningRates() {
        return this.learningRatesByParam;
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

    @AllArgsConstructor
    @Data
    private static class GCStats implements Serializable {
        private String gcName;
        private int deltaGCCount;
        private int deltaGCTime;
    }

    @Override
    public int encodingLengthBytes() {
        //TODO - presumably a more efficient way to do this
        byte[] encoded = encode();
        return encoded.length;
    }

    @Override
    public byte[] encode() {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(this);
        } catch (IOException e) {
            throw new RuntimeException(e); //Should never happen
        }
        return baos.toByteArray();
    }

    @Override
    public void encode(ByteBuffer buffer) {
        buffer.put(encode());
    }

    @Override
    public void encode(OutputStream outputStream) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(outputStream)) {
            oos.writeObject(this);
        }
    }

    @Override
    public void decode(byte[] decode) {
        JavaStatsReport r;
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(decode))) {
            r = (JavaStatsReport) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e); //Should never happen
        }

        Field[] fields = JavaStatsReport.class.getDeclaredFields();
        for (Field f : fields) {
            f.setAccessible(true);
            try {
                f.set(this, f.get(r));
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e); //Should never happen
            }
        }
    }

    @Override
    public void decode(ByteBuffer buffer) {
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        decode(bytes);
    }

    @Override
    public void decode(InputStream inputStream) throws IOException {
        decode(IOUtils.toByteArray(inputStream));
    }
}
