package org.deeplearning4j.spark.impl.paramavg.stats;

import lombok.Data;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

/**
 * Created by Alex on 17/06/2016.
 */
@Data
public class ParameterAveragingTrainingMasterStats implements SparkTrainingStats {

    private static Set<String> columnNames = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(
                    "ParameterAveragingMasterBroadcastCreateTimesMs",
                    "ParameterAveragingMasterFitTimesMs",
                    "ParameterAveragingMasterSplitTimesMs",
                    "ParameterAveragingMasterAggregateTimesMs",
                    "ParameterAveragingMasterProcessParamsUpdaterTimesMs"
            )));

    private SparkTrainingStats workerStats;
    private int[] parameterAveragingMasterBroadcastCreateTimesMs;
    private int[] parameterAveragingMasterFitTimesMs;
    private int[] parameterAveragingMasterSplitTimesMs;
    private int[] paramaterAveragingMasterAggregateTimesMs;
    private int[] parameterAveragingMasterProcessParamsUpdaterTimesMs;


    public ParameterAveragingTrainingMasterStats(SparkTrainingStats workerStats, int[] parameterAveragingMasterBroadcastCreateTimeMs,
                                                 int[] parameterAveragingMasterFitTimeMs, int[] parameterAveragingMasterSplitTimeMs,
                                                 int[] parameterAveragingMasterAggregateTimesMs, int[] parameterAveragingMasterProcessParamsUpdaterTimesMs){
        this.workerStats = workerStats;
        this.parameterAveragingMasterBroadcastCreateTimesMs = parameterAveragingMasterBroadcastCreateTimeMs;
        this.parameterAveragingMasterFitTimesMs = parameterAveragingMasterFitTimeMs;
        this.parameterAveragingMasterSplitTimesMs = parameterAveragingMasterSplitTimeMs;
        this.paramaterAveragingMasterAggregateTimesMs = parameterAveragingMasterAggregateTimesMs;
        this.parameterAveragingMasterProcessParamsUpdaterTimesMs = parameterAveragingMasterProcessParamsUpdaterTimesMs;
    }


    @Override
    public Set<String> getKeySet() {
        Set<String> out = new LinkedHashSet<>(columnNames);
        if(workerStats != null) out.addAll(workerStats.getKeySet());
        return out;
    }

    @Override
    public Object getValue(String key) {
        switch(key){
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
            default:
                if(workerStats != null) return workerStats.getValue(key);
                throw new IllegalArgumentException("Unknown key: \"" + key + "\"");
        }
    }

    @Override
    public void addOtherTrainingStats(SparkTrainingStats other) {
        if(!(other instanceof ParameterAveragingTrainingMasterStats)) throw new IllegalArgumentException("Expected ParameterAveragingTrainingMasterStats, got " + (other != null ? other.getClass() : null));

        ParameterAveragingTrainingMasterStats o = (ParameterAveragingTrainingMasterStats) other;

        if(workerStats != null){
            if(o.workerStats != null ) workerStats.addOtherTrainingStats(o.workerStats);
        } else {
            if(o.workerStats != null) workerStats = o.workerStats;
        }

        this.parameterAveragingMasterBroadcastCreateTimesMs = ArrayUtil.combine(parameterAveragingMasterBroadcastCreateTimesMs, o.parameterAveragingMasterBroadcastCreateTimesMs);
        this.parameterAveragingMasterFitTimesMs = ArrayUtil.combine(parameterAveragingMasterFitTimesMs, o.parameterAveragingMasterFitTimesMs);
    }

    @Override
    public SparkTrainingStats getNestedTrainingStats(){
        return workerStats;
    }

    @Override
    public String statsAsString() {
        StringBuilder sb = new StringBuilder();
        String f = SparkTrainingStats.DEFAULT_PRINT_FORMAT;

        sb.append(String.format(f,"ParameterAveragingMasterBroadcastCreateTimesMs"));
        if(parameterAveragingMasterBroadcastCreateTimesMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(parameterAveragingMasterBroadcastCreateTimesMs)).append("\n");

        sb.append(String.format(f,"ParameterAveragingMasterFitTimesMs"));
        if(parameterAveragingMasterFitTimesMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(parameterAveragingMasterFitTimesMs)).append("\n");

        sb.append(String.format(f,"ParameterAveragingMasterSplitTimesMs"));
        if(parameterAveragingMasterSplitTimesMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(parameterAveragingMasterSplitTimesMs)).append("\n");

        sb.append(String.format(f,"ParameterAveragingMasterAggregateTimesMs"));
        if(paramaterAveragingMasterAggregateTimesMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(paramaterAveragingMasterAggregateTimesMs)).append("\n");

        sb.append(String.format(f,"ParameterAveragingMasterProcessParamsUpdaterTimesMs"));
        if(parameterAveragingMasterProcessParamsUpdaterTimesMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(parameterAveragingMasterProcessParamsUpdaterTimesMs)).append("\n");


        if(workerStats != null) sb.append(workerStats.statsAsString());

        return sb.toString();
    }

    public static class parameterAveragingTrainingMasterStatsHelper {

        private long lastBroadcastStartTime;
        private long lastFitStartTime;
        private long lastSplitStartTime;
        private long lastAggregateStartTime;
        private long lastProcessParamsUpdaterStartTime;

        private SparkTrainingStats workerStats;

        //TODO use fast int collection here (to avoid boxing cost)
        private List<Integer> broadcastTimes = new ArrayList<>();
        private List<Integer> fitTimes = new ArrayList<>();
        private List<Integer> splitTimes = new ArrayList<>();
        private List<Integer> aggregateTimes = new ArrayList<>();
        private List<Integer> processParamsUpdaterTimes = new ArrayList<>();

        public void logBroadcastStart(){
            this.lastBroadcastStartTime = System.currentTimeMillis();
        }

        public void logBroadcastEnd(){
            long now = System.currentTimeMillis();
            broadcastTimes.add((int)(now - lastBroadcastStartTime));
        }

        public void logFitStart(){
            lastFitStartTime = System.currentTimeMillis();
        }

        public void logFitEnd(){
            long now = System.currentTimeMillis();
            fitTimes.add((int)(now - lastFitStartTime));
        }

        public void logSplitStart(){
            lastSplitStartTime = System.currentTimeMillis();
        }

        public void logSplitEnd(){
            long now = System.currentTimeMillis();
            splitTimes.add((int)(now - lastSplitStartTime));
        }

        public void logAggregateStartTime(){
            lastAggregateStartTime = System.currentTimeMillis();
        }

        public void logAggregationEndTime(){
            long now = System.currentTimeMillis();
            aggregateTimes.add((int)(now - lastAggregateStartTime));
        }

        public void logProcessParamsUpdaterStart(){
            lastProcessParamsUpdaterStartTime = System.currentTimeMillis();
        }

        public void logProcessParamsUpdaterEnd(){
            long now = System.currentTimeMillis();
            processParamsUpdaterTimes.add((int)(now - lastProcessParamsUpdaterStartTime));
        }

        public void addWorkerStats(SparkTrainingStats workerStats){
            if(this.workerStats == null) this.workerStats = workerStats;
            else if(workerStats != null) this.workerStats.addOtherTrainingStats(workerStats);
        }

        public ParameterAveragingTrainingMasterStats build(){
            int[] bcast = new int[broadcastTimes.size()];
            for( int i=0; i<bcast.length; i++ ) bcast[i] = broadcastTimes.get(i);
            int[] fit = new int[fitTimes.size()];
            for( int i=0; i<fit.length; i++ ) fit[i] = fitTimes.get(i);
            int[] split = new int[splitTimes.size()];
            for( int i=0; i<split.length; i++ ) split[i] = splitTimes.get(i);
            int[] agg = new int[aggregateTimes.size()];
            for( int i=0; i<agg.length; i++ ) agg[i] = aggregateTimes.get(i);
            int[] proc = new int[processParamsUpdaterTimes.size()];
            for( int i=0; i<proc.length; i++ ) proc[i] = processParamsUpdaterTimes.get(i);

            return new ParameterAveragingTrainingMasterStats(workerStats,bcast,fit,split,agg, proc);
        }

    }

}
