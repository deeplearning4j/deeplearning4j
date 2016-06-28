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
 * Statistics colected by {@link org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingWorker} instances
 *
 * @author Alex Black
 */
@Data
public class ParameterAveragingTrainingWorkerStats implements SparkTrainingStats {

    public static final String DEFAULT_DELIMITER = CommonSparkTrainingStats.DEFAULT_DELIMITER;
    public static final String FILENAME_BROADCAST_GET_STATS = "parameterAveragingWorkerBroadcastGetValueTimeMs.txt";
    public static final String FILENAME_INIT_STATS = "parameterAveragingWorkerInitTimeMs.txt";
    public static final String FILENAME_FIT_STATS = "parameterAveragingWorkerFitTimesMs.txt";

    private List<EventStats> parameterAveragingWorkerBroadcastGetValueTimeMs;
    private List<EventStats> parameterAveragingWorkerInitTimeMs;
    private List<EventStats> parameterAveragingWorkerFitTimesMs;

    private static Set<String> columnNames = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(
                    "ParameterAveragingWorkerBroadcastGetValueTimeMs",
                    "ParameterAveragingWorkerInitTimeMs",
                    "ParameterAveragingWorkerFitTimesMs"
            )));

    public ParameterAveragingTrainingWorkerStats(List<EventStats> parameterAveragingWorkerBroadcastGetValueTimeMs, List<EventStats> parameterAveragingWorkerInitTimeMs,
                                                 List<EventStats> parameterAveragingWorkerFitTimesMs){
        this.parameterAveragingWorkerBroadcastGetValueTimeMs = parameterAveragingWorkerBroadcastGetValueTimeMs;
        this.parameterAveragingWorkerInitTimeMs = parameterAveragingWorkerInitTimeMs;
        this.parameterAveragingWorkerFitTimesMs = parameterAveragingWorkerFitTimesMs;
    }

    @Override
    public Set<String> getKeySet() {
        return columnNames;
    }

    @Override
    public List<EventStats> getValue(String key) {
        switch(key){
            case "ParameterAveragingWorkerBroadcastGetValueTimeMs":
                return parameterAveragingWorkerBroadcastGetValueTimeMs;
            case "ParameterAveragingWorkerInitTimeMs":
                return parameterAveragingWorkerInitTimeMs;
            case "ParameterAveragingWorkerFitTimesMs":
                return parameterAveragingWorkerFitTimesMs;
            default:
                throw new IllegalArgumentException("Unknown key: \"" + key + "\"");
        }
    }

    @Override
    public String getShortNameForKey(String key){
        switch(key){
            case "ParameterAveragingWorkerBroadcastGetValueTimeMs":
                return "BroadcastGetVal";
            case "ParameterAveragingWorkerInitTimeMs":
                return "ModelInit";
            case "ParameterAveragingWorkerFitTimesMs":
                return "Fit";
            default:
                throw new IllegalArgumentException("Unknown key: \"" + key + "\"");
        }
    }

    @Override
    public boolean defaultIncludeInPlots(String key){
        switch(key){
            case "ParameterAveragingWorkerBroadcastGetValueTimeMs":
                return true;
            case "ParameterAveragingWorkerInitTimeMs":
                return true;
            case "ParameterAveragingWorkerFitTimesMs":
                return true;
            default:
                throw new IllegalArgumentException("Unknown key: \"" + key + "\"");
        }
    }

    @Override
    public void addOtherTrainingStats(SparkTrainingStats other) {
        if(!(other instanceof ParameterAveragingTrainingWorkerStats)) throw new IllegalArgumentException("Cannot merge ParameterAveragingTrainingWorkerStats with " + (other != null ? other.getClass() : null));

        ParameterAveragingTrainingWorkerStats o = (ParameterAveragingTrainingWorkerStats)other;

        this.parameterAveragingWorkerBroadcastGetValueTimeMs.addAll(o.parameterAveragingWorkerBroadcastGetValueTimeMs);
        this.parameterAveragingWorkerInitTimeMs.addAll(o.parameterAveragingWorkerInitTimeMs);
        this.parameterAveragingWorkerFitTimesMs.addAll(o.parameterAveragingWorkerFitTimesMs);
    }

    @Override
    public SparkTrainingStats getNestedTrainingStats(){
        return null;
    }

    @Override
    public String statsAsString() {
        StringBuilder sb = new StringBuilder();
        String f = SparkTrainingStats.DEFAULT_PRINT_FORMAT;

        sb.append(String.format(f,"ParameterAveragingWorkerBroadcastGetValueTimeMs"));
        if(parameterAveragingWorkerBroadcastGetValueTimeMs == null ) sb.append("-\n");
        else sb.append(StatsUtils.getDurationAsString(parameterAveragingWorkerBroadcastGetValueTimeMs,",")).append("\n");

        sb.append(String.format(f,"ParameterAveragingWorkerInitTimeMs"));
        if(parameterAveragingWorkerInitTimeMs == null ) sb.append("-\n");
        else sb.append(StatsUtils.getDurationAsString(parameterAveragingWorkerInitTimeMs,",")).append("\n");

        sb.append(String.format(f,"ParameterAveragingWorkerFitTimesMs"));
        if(parameterAveragingWorkerFitTimesMs == null ) sb.append("-\n");
        else sb.append(StatsUtils.getDurationAsString(parameterAveragingWorkerFitTimesMs,",")).append("\n");

        return sb.toString();
    }

    @Override
    public void exportStatFiles(String outputPath, SparkContext sc) throws IOException {
        String d = DEFAULT_DELIMITER;

        //Broadcast get time:
        StatsUtils.exportStats(parameterAveragingWorkerBroadcastGetValueTimeMs,outputPath, FILENAME_BROADCAST_GET_STATS, d, sc);

        //Network init time:
        StatsUtils.exportStats(parameterAveragingWorkerInitTimeMs,outputPath, FILENAME_INIT_STATS, d, sc);

        //Network fit time:
        StatsUtils.exportStats(parameterAveragingWorkerFitTimesMs,outputPath, FILENAME_FIT_STATS, d, sc);
    }

    public static class ParameterAveragingTrainingWorkerStatsHelper {
        private long broadcastStartTime;
        private long broadcastEndTime;
        private long initEndTime;
        private long lastFitStartTime;
        //TODO replace with fast int collection (no boxing)
        private List<EventStats> fitTimes = new ArrayList<>();

        private final TimeSource timeSource = TimeSourceProvider.getInstance();


        public void logBroadcastGetValueStart(){
            broadcastStartTime = timeSource.currentTimeMillis();
        }

        public void logBroadcastGetValueEnd(){
            broadcastEndTime = timeSource.currentTimeMillis();
        }

        public void logInitEnd(){
            initEndTime = timeSource.currentTimeMillis();
        }

        public void logFitStart(){
            lastFitStartTime = timeSource.currentTimeMillis();
        }

        public void logFitEnd(){
            long now = timeSource.currentTimeMillis();
            fitTimes.add(new BaseEventStats(lastFitStartTime, now - lastFitStartTime));
        }

        public ParameterAveragingTrainingWorkerStats build(){
            //Using ArrayList not Collections.singletonList() etc so we can add to them later (during merging)
            List<EventStats> bList = new ArrayList<>();
            bList.add(new BaseEventStats(broadcastStartTime,broadcastEndTime-broadcastStartTime));
            List<EventStats> initList = new ArrayList<>();
            initList.add(new BaseEventStats(broadcastEndTime,initEndTime-broadcastEndTime));    //Init starts at same time that broadcast ends

            return new ParameterAveragingTrainingWorkerStats(bList, initList, fitTimes);

        }
    }
}
