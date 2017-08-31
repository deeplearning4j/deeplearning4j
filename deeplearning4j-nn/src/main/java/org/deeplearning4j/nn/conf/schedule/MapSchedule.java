package org.deeplearning4j.nn.conf.schedule;

import java.util.HashMap;
import java.util.Map;

public class MapSchedule implements ISchedule {

    private ScheduleType scheduleType;
    private Map<Integer,Double> values;

    public MapSchedule(ScheduleType scheduleType, Map<Integer,Double> values){
        this.scheduleType = scheduleType;
        this.values = values;
    }

    @Override
    public double valueAt(double currentValue, int iteration, int epoch) {
        int key;
        if(scheduleType == ScheduleType.ITERATION){
            key = iteration;
        } else {
            key = epoch;
        }

        if(values.containsKey(key)){
            return values.get(key);
        } else {
            //Key doesn't exist - no change
            return currentValue;
        }
    }

    public static class Builder {

        private ScheduleType scheduleType;
        private Map<Integer,Double> values = new HashMap<>();

        public Builder builder(ScheduleType scheduleType){
            this.scheduleType = scheduleType;
            return this;
        }

        public Builder add(int position, double value){
            values.put(position, value);
            return this;
        }

        public MapSchedule build(){
            return new MapSchedule(scheduleType, values);
        }
    }
}
