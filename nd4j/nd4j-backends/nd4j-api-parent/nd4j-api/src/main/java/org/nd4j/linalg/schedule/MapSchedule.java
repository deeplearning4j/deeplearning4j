package org.nd4j.linalg.schedule;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * MapSchedule is a schedule based on specific values in a {@code Map<Integer,Double>}.<br>
 * For example, if the map contains the following: (0,1.0), (10,0.5), (20, 0.2) then iteration/epoch 0 to 9 inclusive
 * will have value 1.0, 10 to 19 will have 0.5, and 20+ will have value 0.2.<br>
 * Note that the map MUST have a key for position 0 - this is the initial value.
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode
@JsonIgnoreProperties({"allKeysSorted"})
public class MapSchedule implements ISchedule {

    private ScheduleType scheduleType;
    private Map<Integer, Double> values;

    private int[] allKeysSorted;

    public MapSchedule(@JsonProperty("scheduleType") @NonNull ScheduleType scheduleType,
                       @JsonProperty("values") @NonNull Map<Integer, Double> values) {
        if (!values.containsKey(0)) {
            throw new IllegalArgumentException("Invalid set of values: must contain initial value (position 0)");
        }
        this.scheduleType = scheduleType;
        this.values = values;

        this.allKeysSorted = new int[values.size()];
        int pos = 0;
        for (Integer i : values.keySet()) {
            allKeysSorted[pos++] = i;
        }
        Arrays.sort(allKeysSorted);
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        int i = (scheduleType == ScheduleType.ITERATION ? iteration : epoch);

        if (values.containsKey(i)) {
            return values.get(i);
        } else {
            //Key doesn't exist - find nearest key...
            if (i >= allKeysSorted[allKeysSorted.length - 1]) {
                return values.get(allKeysSorted[allKeysSorted.length - 1]);
            } else {
                /*
                Returned:
                index of the search key, if it is contained in the array; otherwise, (-(insertion point) - 1). The
                 insertion point is defined as the point at which the key would be inserted into the array: the index
                  of the first element greater than the key
                 */
                int pt = Arrays.binarySearch(allKeysSorted, i);
                int iPt = -(pt + 1);
                double d = values.get(allKeysSorted[iPt-1]);
                return d;
            }
        }
    }

    @Override
    public ISchedule clone() {
        return new MapSchedule(scheduleType, values);
    }

    /**
     * DynamicCustomOpsBuilder for conveniently constructing map schedules
     */
    public static class Builder {

        private ScheduleType scheduleType;
        private Map<Integer, Double> values = new HashMap<>();

        /**
         * @param scheduleType Schedule opType to use
         */
        public Builder(ScheduleType scheduleType) {
            this.scheduleType = scheduleType;
        }

        /**
         * Add a single point to the map schedule. Indexes start at 0
         *
         * @param position Position to add (iteration or epoch index, depending on setting)
         * @param value    Value for that iteraiton/epoch
         */
        public Builder add(int position, double value) {
            values.put(position, value);
            return this;
        }

        public MapSchedule build() {
            return new MapSchedule(scheduleType, values);
        }
    }
}
