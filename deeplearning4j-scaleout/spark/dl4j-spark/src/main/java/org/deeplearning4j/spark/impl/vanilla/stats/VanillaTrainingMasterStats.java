package org.deeplearning4j.spark.impl.vanilla.stats;

import lombok.Data;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

/**
 * Created by Alex on 17/06/2016.
 */
@Data
public class VanillaTrainingMasterStats implements SparkTrainingStats {

    private static Set<String> columnNames = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(
                    "VanillaMasterBroadcastCreateTimesMs",
                    "VanillaMasterFitTimesMs",
                    "VanillaMasterSplitTimesMs",
                    "VanillaMasterAggregateTimesMs",
                    "VanillaMasterProcessParamsUpdaterTimesMs"
            )));

    private SparkTrainingStats workerStats;
    private int[] vanillaMasterBroadcastCreateTimesMs;
    private int[] vanillaMasterFitTimesMs;
    private int[] vanillaMasterSplitTimesMs;
    private int[] vanillaMasterAggregateTimesMs;
    private int[] vanillaMasterProcessParamsUpdaterTimesMs;


    public VanillaTrainingMasterStats(SparkTrainingStats workerStats, int[] vanillaMasterBroadcastCreateTimeMs,
                                      int[] vanillaMasterFitTimeMs, int[] vanillaMasterSplitTimeMs,
                                      int[] vanillaMasterAggregateTimesMs, int[] vanillaMasterProcessParamsUpdaterTimesMs){
        this.workerStats = workerStats;
        this.vanillaMasterBroadcastCreateTimesMs = vanillaMasterBroadcastCreateTimeMs;
        this.vanillaMasterFitTimesMs = vanillaMasterFitTimeMs;
        this.vanillaMasterSplitTimesMs = vanillaMasterSplitTimeMs;
        this.vanillaMasterAggregateTimesMs = vanillaMasterAggregateTimesMs;
        this.vanillaMasterProcessParamsUpdaterTimesMs = vanillaMasterProcessParamsUpdaterTimesMs;
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
            case "VanillaMasterBroadcastCreateTimesMs":
                return vanillaMasterBroadcastCreateTimesMs;
            case "VanillaMasterFitTimesMs":
                return vanillaMasterFitTimesMs;
            case "VanillaMasterSplitTimesMs":
                return vanillaMasterSplitTimesMs;
            case "VanillaMasterAggregateTimesMs":
                return vanillaMasterAggregateTimesMs;
            case "VanillaMasterProcessParamsUpdaterTimesMs":
                return vanillaMasterProcessParamsUpdaterTimesMs;
            default:
                if(workerStats != null) return workerStats.getValue(key);
                throw new IllegalArgumentException("Unknown key: \"" + key + "\"");
        }
    }

    @Override
    public void addOtherTrainingStats(SparkTrainingStats other) {
        if(!(other instanceof VanillaTrainingMasterStats)) throw new IllegalArgumentException("Expected VanillaTrainingMasterStats, got " + (other != null ? other.getClass() : null));

        VanillaTrainingMasterStats o = (VanillaTrainingMasterStats) other;

        if(workerStats != null){
            if(o.workerStats != null ) workerStats.addOtherTrainingStats(o.workerStats);
        } else {
            if(o.workerStats != null) workerStats = o.workerStats;
        }

        this.vanillaMasterBroadcastCreateTimesMs = ArrayUtil.combine(vanillaMasterBroadcastCreateTimesMs, o.vanillaMasterBroadcastCreateTimesMs);
        this.vanillaMasterFitTimesMs = ArrayUtil.combine(vanillaMasterFitTimesMs, o.vanillaMasterFitTimesMs);
    }

    @Override
    public SparkTrainingStats getNestedTrainingStats(){
        return workerStats;
    }

    @Override
    public String statsAsString() {
        StringBuilder sb = new StringBuilder();
        String f = SparkTrainingStats.DEFAULT_PRINT_FORMAT;

        sb.append(String.format(f,"VanillaMasterBroadcastCreateTimesMs"));
        if(vanillaMasterBroadcastCreateTimesMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(vanillaMasterBroadcastCreateTimesMs)).append("\n");

        sb.append(String.format(f,"VanillaMasterFitTimesMs"));
        if(vanillaMasterFitTimesMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(vanillaMasterFitTimesMs)).append("\n");

        sb.append(String.format(f,"VanillaMasterSplitTimesMs"));
        if(vanillaMasterSplitTimesMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(vanillaMasterSplitTimesMs)).append("\n");

        sb.append(String.format(f,"VanillaMasterAggregateTimesMs"));
        if(vanillaMasterAggregateTimesMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(vanillaMasterAggregateTimesMs)).append("\n");

        sb.append(String.format(f,"VanillaMasterProcessParamsUpdaterTimesMs"));
        if(vanillaMasterProcessParamsUpdaterTimesMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(vanillaMasterProcessParamsUpdaterTimesMs)).append("\n");


        if(workerStats != null) sb.append(workerStats.statsAsString());

        return sb.toString();
    }

    public static class VanillaTrainingMasterStatsHelper {

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

        public VanillaTrainingMasterStats build(){
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

            return new VanillaTrainingMasterStats(workerStats,bcast,fit,split,agg, proc);
        }

    }

}
