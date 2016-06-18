package org.deeplearning4j.spark.impl.paramavg.stats;

import lombok.Data;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

/**
 * Created by Alex on 17/06/2016.
 */
@Data
public class ParameterAveragingTrainingWorkerStats implements SparkTrainingStats {

    private int[] parameterAveragingWorkerBroadcastGetValueTimeMs;
    private int[] parameterAveragingWorkerInitTimeMs;
    private int[] parameterAveragingWorkerFitTimesMs;

    private static Set<String> columnNames = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(
                    "ParameterAveragingWorkerBroadcastGetValueTimeMs",
                    "ParameterAveragingWorkerInitTimeMs",
                    "ParameterAveragingWorkerFitTimesMs"
            )));

    public ParameterAveragingTrainingWorkerStats(int parameterAveragingWorkerBroadcastGetValueTimeMs, int parameterAveragingWorkerInitTimeMs,
                                                 int[] parameterAveragingWorkerFitTimesMs){
        this.parameterAveragingWorkerBroadcastGetValueTimeMs = new int[]{parameterAveragingWorkerBroadcastGetValueTimeMs};
        this.parameterAveragingWorkerInitTimeMs = new int[]{parameterAveragingWorkerInitTimeMs};
        this.parameterAveragingWorkerFitTimesMs = parameterAveragingWorkerFitTimesMs;
    }

    @Override
    public Set<String> getKeySet() {
        return columnNames;
    }

    @Override
    public Object getValue(String key) {
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
    public void addOtherTrainingStats(SparkTrainingStats other) {
        if(!(other instanceof ParameterAveragingTrainingWorkerStats)) throw new IllegalArgumentException("Cannot merge ParameterAveragingTrainingWorkerStats with " + (other != null ? other.getClass() : null));

        ParameterAveragingTrainingWorkerStats o = (ParameterAveragingTrainingWorkerStats)other;

        this.parameterAveragingWorkerBroadcastGetValueTimeMs = ArrayUtil.combine(parameterAveragingWorkerBroadcastGetValueTimeMs,o.parameterAveragingWorkerBroadcastGetValueTimeMs);
        this.parameterAveragingWorkerInitTimeMs = ArrayUtil.combine(parameterAveragingWorkerInitTimeMs, o.parameterAveragingWorkerInitTimeMs);
        this.parameterAveragingWorkerFitTimesMs = ArrayUtil.combine(parameterAveragingWorkerFitTimesMs, o.parameterAveragingWorkerFitTimesMs);
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
        else sb.append(Arrays.toString(parameterAveragingWorkerBroadcastGetValueTimeMs)).append("\n");

        sb.append(String.format(f,"ParameterAveragingWorkerInitTimeMs"));
        if(parameterAveragingWorkerInitTimeMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(parameterAveragingWorkerInitTimeMs)).append("\n");

        sb.append(String.format(f,"ParameterAveragingWorkerFitTimesMs"));
        if(parameterAveragingWorkerFitTimesMs == null ) sb.append("-\n");
        else sb.append(Arrays.toString(parameterAveragingWorkerFitTimesMs)).append("\n");

        return sb.toString();
    }

    public static class ParameterAveragingTrainingWorkerStatsHelper {
        private long broadcastStartTime;
        private long broadcastEndTime;
        private long initEndTime;
        private long lastFitStartTime;
        //TODO replace with fast int collection (no boxing)
        private List<Integer> fitTimes = new ArrayList<>();


        public void logBroadcastGetValueStart(){
            broadcastStartTime = System.currentTimeMillis();
        }

        public void logBroadcastGetValueEnd(){
            broadcastEndTime = System.currentTimeMillis();
        }

        public void logInitEnd(){
            initEndTime = System.currentTimeMillis();
        }

        public void logFitStart(){
            lastFitStartTime = System.currentTimeMillis();
        }

        public void logFitEnd(){
            long now = System.currentTimeMillis();
            fitTimes.add((int)(now - lastFitStartTime));
        }

        public ParameterAveragingTrainingWorkerStats build(){
            int bcast = (int)(broadcastEndTime - broadcastStartTime);
            int init = (int)(initEndTime - broadcastEndTime);   //Init starts at same time that broadcast ends
            int[] fitTimesArr = new int[fitTimes.size()];
            for( int i=0; i<fitTimesArr.length; i++ ) fitTimesArr[i] = fitTimes.get(i);
            return new ParameterAveragingTrainingWorkerStats(bcast, init, fitTimesArr);
        }
    }
}
