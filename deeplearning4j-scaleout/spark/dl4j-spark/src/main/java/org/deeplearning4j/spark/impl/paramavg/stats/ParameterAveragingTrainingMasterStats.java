package org.deeplearning4j.spark.impl.paramavg.stats;

import lombok.Data;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkContext;
import org.deeplearning4j.spark.api.stats.CommonSparkTrainingStats;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.stats.BaseEventStats;
import org.deeplearning4j.spark.stats.EventStats;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.deeplearning4j.spark.time.TimeSource;
import org.deeplearning4j.spark.time.TimeSourceProvider;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.IOException;
import java.util.*;

/**
 * Statistics colected by a {@link org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster}
 *
 * @author Alex Black
 */
@Data
public class ParameterAveragingTrainingMasterStats implements SparkTrainingStats {

    public static final String DEFAULT_DELIMITER = CommonSparkTrainingStats.DEFAULT_DELIMITER;
    public static final String FILENAME_BROADCAST_CREATE = "parameterAveragingMasterBroadcastCreateTimesMs.txt";
    public static final String FILENAME_FIT_TIME = "parameterAveragingMasterFitTimesMs.txt";
    public static final String FILENAME_SPLIT_TIME = "parameterAveragingMasterSplitTimesMs.txt";
    public static final String FILENAME_AGGREGATE_TIME = "parameterAveragingMasterAggregateTimesMs.txt";
    public static final String FILENAME_PROCESS_PARAMS_TIME = "parameterAveragingMasterProcessParamsUpdaterTimesMs.txt";
    public static final String FILENAME_REPARTITION_STATS = "parameterAveragingMasterRepartitionTimesMs.txt";

    private static Set<String> columnNames = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(
                    "ParameterAveragingMasterBroadcastCreateTimesMs",
                    "ParameterAveragingMasterFitTimesMs",
                    "ParameterAveragingMasterSplitTimesMs",
                    "ParameterAveragingMasterAggregateTimesMs",
                    "ParameterAveragingMasterProcessParamsUpdaterTimesMs",
                    "ParameterAveragingMasterRepartitionTimesMs"
            )));

    private SparkTrainingStats workerStats;
    private List<EventStats> parameterAveragingMasterBroadcastCreateTimesMs;
    private List<EventStats> parameterAveragingMasterFitTimesMs;
    private List<EventStats> parameterAveragingMasterSplitTimesMs;
    private List<EventStats> paramaterAveragingMasterAggregateTimesMs;
    private List<EventStats> parameterAveragingMasterProcessParamsUpdaterTimesMs;
    private List<EventStats> parameterAveragingMasterRepartitionTimesMs;


    public ParameterAveragingTrainingMasterStats(SparkTrainingStats workerStats, List<EventStats> parameterAveragingMasterBroadcastCreateTimeMs,
                                                 List<EventStats> parameterAveragingMasterFitTimeMs, List<EventStats> parameterAveragingMasterSplitTimeMs,
                                                 List<EventStats> parameterAveragingMasterAggregateTimesMs, List<EventStats> parameterAveragingMasterProcessParamsUpdaterTimesMs,
                                                 List<EventStats> parameterAveragingMasterRepartitionTimesMs) {
        this.workerStats = workerStats;
        this.parameterAveragingMasterBroadcastCreateTimesMs = parameterAveragingMasterBroadcastCreateTimeMs;
        this.parameterAveragingMasterFitTimesMs = parameterAveragingMasterFitTimeMs;
        this.parameterAveragingMasterSplitTimesMs = parameterAveragingMasterSplitTimeMs;
        this.paramaterAveragingMasterAggregateTimesMs = parameterAveragingMasterAggregateTimesMs;
        this.parameterAveragingMasterProcessParamsUpdaterTimesMs = parameterAveragingMasterProcessParamsUpdaterTimesMs;
        this.parameterAveragingMasterRepartitionTimesMs = parameterAveragingMasterRepartitionTimesMs;
    }


    @Override
    public Set<String> getKeySet() {
        Set<String> out = new LinkedHashSet<>(columnNames);
        if (workerStats != null) out.addAll(workerStats.getKeySet());
        return out;
    }

    @Override
    public List<EventStats> getValue(String key) {
        switch (key) {
            case "ParameterAveragingMasterBroadcastCreateTimesMs":
                return parameterAveragingMasterBroadcastCreateTimesMs;
            case "ParameterAveragingMasterFitTimesMs":
                return parameterAveragingMasterFitTimesMs;
            case "ParameterAveragingMasterSplitTimesMs":
                return parameterAveragingMasterSplitTimesMs;
            case "ParameterAveragingMasterAggregateTimesMs":
                return paramaterAveragingMasterAggregateTimesMs;
            case "ParameterAveragingMasterProcessParamsUpdaterTimesMs":
                return parameterAveragingMasterProcessParamsUpdaterTimesMs;
            case "ParameterAveragingMasterRepartitionTimesMs":
                return parameterAveragingMasterRepartitionTimesMs;
            default:
                if (workerStats != null) return workerStats.getValue(key);
                throw new IllegalArgumentException("Unknown key: \"" + key + "\"");
        }
    }

    @Override
    public String getShortNameForKey(String key) {
        switch (key) {
            case "ParameterAveragingMasterBroadcastCreateTimesMs":
                return "CreateBroadcast";
            case "ParameterAveragingMasterFitTimesMs":
                return "Fit";
            case "ParameterAveragingMasterSplitTimesMs":
                return "Split";
            case "ParameterAveragingMasterAggregateTimesMs":
                return "Aggregate";
            case "ParameterAveragingMasterProcessParamsUpdaterTimesMs":
                return "ProcessParams";
            case "ParameterAveragingMasterRepartitionTimesMs":
                return "Repartition";
            default:
                if (workerStats != null) return workerStats.getShortNameForKey(key);
                throw new IllegalArgumentException("Unknown key: \"" + key + "\"");
        }
    }

    @Override
    public boolean defaultIncludeInPlots(String key) {
        switch (key) {
            case "ParameterAveragingMasterBroadcastCreateTimesMs":
                return true;
            case "ParameterAveragingMasterFitTimesMs":
                return false;
            case "ParameterAveragingMasterSplitTimesMs":
                return true;
            case "ParameterAveragingMasterAggregateTimesMs":
                return true;
            case "ParameterAveragingMasterProcessParamsUpdaterTimesMs":
                return true;
            case "ParameterAveragingMasterRepartitionTimesMs":
                return true;
            default:
                if (workerStats != null) return workerStats.defaultIncludeInPlots(key);
                return false;
        }
    }

    @Override
    public void addOtherTrainingStats(SparkTrainingStats other) {
        if (!(other instanceof ParameterAveragingTrainingMasterStats))
            throw new IllegalArgumentException("Expected ParameterAveragingTrainingMasterStats, got " + (other != null ? other.getClass() : null));

        ParameterAveragingTrainingMasterStats o = (ParameterAveragingTrainingMasterStats) other;

        if (workerStats != null) {
            if (o.workerStats != null) workerStats.addOtherTrainingStats(o.workerStats);
        } else {
            if (o.workerStats != null) workerStats = o.workerStats;
        }

        this.parameterAveragingMasterBroadcastCreateTimesMs.addAll(o.parameterAveragingMasterBroadcastCreateTimesMs);
        this.parameterAveragingMasterFitTimesMs.addAll(o.parameterAveragingMasterFitTimesMs);
        if (parameterAveragingMasterRepartitionTimesMs == null) {
            if (o.parameterAveragingMasterRepartitionTimesMs != null)
                parameterAveragingMasterRepartitionTimesMs = o.parameterAveragingMasterRepartitionTimesMs;
        } else {
            if (o.parameterAveragingMasterRepartitionTimesMs != null)
                parameterAveragingMasterRepartitionTimesMs.addAll(o.parameterAveragingMasterRepartitionTimesMs);
        }
    }

    @Override
    public SparkTrainingStats getNestedTrainingStats() {
        return workerStats;
    }

    @Override
    public String statsAsString() {
        StringBuilder sb = new StringBuilder();
        String f = SparkTrainingStats.DEFAULT_PRINT_FORMAT;

        sb.append(String.format(f, "ParameterAveragingMasterBroadcastCreateTimesMs"));
        if (parameterAveragingMasterBroadcastCreateTimesMs == null) sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterBroadcastCreateTimesMs, ",")).append("\n");

        sb.append(String.format(f, "ParameterAveragingMasterFitTimesMs"));
        if (parameterAveragingMasterFitTimesMs == null) sb.append("-\n");
        else sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterFitTimesMs, ",")).append("\n");

        sb.append(String.format(f, "ParameterAveragingMasterSplitTimesMs"));
        if (parameterAveragingMasterSplitTimesMs == null) sb.append("-\n");
        else sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterSplitTimesMs, ",")).append("\n");

        sb.append(String.format(f, "ParameterAveragingMasterAggregateTimesMs"));
        if (paramaterAveragingMasterAggregateTimesMs == null) sb.append("-\n");
        else sb.append(StatsUtils.getDurationAsString(paramaterAveragingMasterAggregateTimesMs, ",")).append("\n");

        sb.append(String.format(f, "ParameterAveragingMasterProcessParamsUpdaterTimesMs"));
        if (parameterAveragingMasterProcessParamsUpdaterTimesMs == null) sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterProcessParamsUpdaterTimesMs, ",")).append("\n");


        sb.append(String.format(f, "ParameterAveragingMasterRepartitionTimesMs"));
        if (parameterAveragingMasterRepartitionTimesMs == null) sb.append("-\n");
        else sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterRepartitionTimesMs, ",")).append("\n");

        if (workerStats != null) sb.append(workerStats.statsAsString());

        return sb.toString();
    }

    @Override
    public void exportStatFiles(String outputPath, SparkContext sc) throws IOException {
        String d = DEFAULT_DELIMITER;

        //broadcast create time:
        String broadcastTimePath = FilenameUtils.concat(outputPath, FILENAME_BROADCAST_CREATE);
        StatsUtils.exportStats(parameterAveragingMasterBroadcastCreateTimesMs, broadcastTimePath, d, sc);

        //Fit time:
        String fitTimePath = FilenameUtils.concat(outputPath, FILENAME_FIT_TIME);
        StatsUtils.exportStats(parameterAveragingMasterFitTimesMs, fitTimePath, d, sc);

        //Split time:
        String splitTimePath = FilenameUtils.concat(outputPath, FILENAME_SPLIT_TIME);
        StatsUtils.exportStats(parameterAveragingMasterSplitTimesMs, splitTimePath, d, sc);

        //Aggregate time:
        String aggregatePath = FilenameUtils.concat(outputPath, FILENAME_AGGREGATE_TIME);
        StatsUtils.exportStats(paramaterAveragingMasterAggregateTimesMs, aggregatePath, d, sc);

        //broadcast create time:
        String processParamsPath = FilenameUtils.concat(outputPath, FILENAME_PROCESS_PARAMS_TIME);
        StatsUtils.exportStats(parameterAveragingMasterProcessParamsUpdaterTimesMs, processParamsPath, d, sc);

        //Repartition
        if (parameterAveragingMasterRepartitionTimesMs != null) {
            String repartitionPath = FilenameUtils.concat(outputPath, FILENAME_REPARTITION_STATS);
            StatsUtils.exportStats(parameterAveragingMasterRepartitionTimesMs, repartitionPath, d, sc);
        }

        if (workerStats != null) workerStats.exportStatFiles(outputPath, sc);
    }

    public static class ParameterAveragingTrainingMasterStatsHelper {

        private long lastBroadcastStartTime;
        private long lastRepartitionStartTime;
        private long lastFitStartTime;
        private long lastSplitStartTime;
        private long lastAggregateStartTime;
        private long lastProcessParamsUpdaterStartTime;

        private SparkTrainingStats workerStats;

        //TODO use fast int collection here (to avoid boxing cost)
        private List<EventStats> broadcastTimes = new ArrayList<>();
        private List<EventStats> repartitionTimes = new ArrayList<>();
        private List<EventStats> fitTimes = new ArrayList<>();
        private List<EventStats> splitTimes = new ArrayList<>();
        private List<EventStats> aggregateTimes = new ArrayList<>();
        private List<EventStats> processParamsUpdaterTimes = new ArrayList<>();

        private final TimeSource timeSource = TimeSourceProvider.getInstance();

        public void logBroadcastStart() {
            this.lastBroadcastStartTime = timeSource.currentTimeMillis();
        }

        public void logBroadcastEnd() {
            long now = timeSource.currentTimeMillis();

            broadcastTimes.add(new BaseEventStats(lastBroadcastStartTime, now - lastBroadcastStartTime));
        }

        public void logRepartitionStart() {
            lastRepartitionStartTime = timeSource.currentTimeMillis();
        }

        public void logRepartitionEnd() {
            long now = timeSource.currentTimeMillis();
            repartitionTimes.add(new BaseEventStats(lastRepartitionStartTime, now - lastRepartitionStartTime));
        }

        public void logFitStart() {
            lastFitStartTime = timeSource.currentTimeMillis();
        }

        public void logFitEnd() {
            long now = timeSource.currentTimeMillis();
            fitTimes.add(new BaseEventStats(lastFitStartTime, now - lastFitStartTime));
        }

        public void logSplitStart() {
            lastSplitStartTime = timeSource.currentTimeMillis();
        }

        public void logSplitEnd() {
            long now = timeSource.currentTimeMillis();
            splitTimes.add(new BaseEventStats(lastSplitStartTime, now - lastSplitStartTime));
        }

        public void logAggregateStartTime() {
            lastAggregateStartTime = timeSource.currentTimeMillis();
        }

        public void logAggregationEndTime() {
            long now = timeSource.currentTimeMillis();
            aggregateTimes.add(new BaseEventStats(lastAggregateStartTime, now - lastAggregateStartTime));
        }

        public void logProcessParamsUpdaterStart() {
            lastProcessParamsUpdaterStartTime = timeSource.currentTimeMillis();
        }

        public void logProcessParamsUpdaterEnd() {
            long now = timeSource.currentTimeMillis();
            processParamsUpdaterTimes.add(new BaseEventStats(lastProcessParamsUpdaterStartTime, now - lastProcessParamsUpdaterStartTime));
        }

        public void addWorkerStats(SparkTrainingStats workerStats) {
            if (this.workerStats == null) this.workerStats = workerStats;
            else if (workerStats != null) this.workerStats.addOtherTrainingStats(workerStats);
        }

        public ParameterAveragingTrainingMasterStats build() {
            return new ParameterAveragingTrainingMasterStats(workerStats, broadcastTimes, fitTimes, splitTimes, aggregateTimes,
                    processParamsUpdaterTimes, repartitionTimes);
        }

    }

}
