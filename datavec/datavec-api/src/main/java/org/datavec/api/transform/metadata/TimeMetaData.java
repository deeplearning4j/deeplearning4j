/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.metadata;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.TimeZone;

/**
 * TimeMetaData: Meta data for a date/time column.
 * <b>NOTE:</b> Time values are stored in epoch (millisecond) format.
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class TimeMetaData extends BaseColumnMetaData {

    private final DateTimeZone timeZone;
    private final Long minValidTime;
    private final Long maxValidTime;

    /**
     * Create a TimeMetaData column with no restrictions and UTC timezone.
     */
    public TimeMetaData(String name) {
        this(name, DateTimeZone.UTC, null, null);
    }

    /**
     * Create a TimeMetaData column with no restriction on the allowable times
     *
     * @param timeZone Timezone for this column. Typically used for parsing
     */
    public TimeMetaData(String name, TimeZone timeZone) {
        this(name, timeZone, null, null);
    }

    /**
     * Create a TimeMetaData column with no restriction on the allowable times
     *
     * @param timeZone Timezone for this column.
     */
    public TimeMetaData(String name, DateTimeZone timeZone) {
        this(name, timeZone, null, null);
    }

    /**
     * @param timeZone     Timezone for this column. Typically used for parsing and some transforms
     * @param minValidTime Minimum valid time, in milliseconds (timestamp format). If null: no restriction
     * @param maxValidTime Maximum valid time, in milliseconds (timestamp format). If null: no restriction
     */
    public TimeMetaData(String name, TimeZone timeZone, Long minValidTime, Long maxValidTime) {
        super(name);
        this.timeZone = DateTimeZone.forTimeZone(timeZone);
        this.minValidTime = minValidTime;
        this.maxValidTime = maxValidTime;
    }

    /**
     * @param timeZone     Timezone for this column. Typically used for parsing and some transforms
     * @param minValidTime Minimum valid time, in milliseconds (timestamp format). If null: no restriction
     * @param maxValidTime Maximum valid time, in milliseconds (timestamp format). If null: no restriction
     */
    public TimeMetaData(@JsonProperty("name") String name, @JsonProperty("timeZone") DateTimeZone timeZone,
                    @JsonProperty("minValidTime") Long minValidTime, @JsonProperty("maxValidTime") Long maxValidTime) {
        super(name);
        this.timeZone = timeZone;
        this.minValidTime = minValidTime;
        this.maxValidTime = maxValidTime;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Time;
    }

    @Override
    public boolean isValid(Writable writable) {
        long epochMillisec;

        if (writable instanceof LongWritable) {
            epochMillisec = writable.toLong();
        } else {
            try {
                epochMillisec = Long.parseLong(writable.toString());
            } catch (NumberFormatException e) {
                return false;
            }
        }
        if (minValidTime != null && epochMillisec < minValidTime)
            return false;
        return !(maxValidTime != null && epochMillisec > maxValidTime);
    }

    /**
     * Is the given object valid for this column,
     * given the column type and any
     * restrictions given by the
     * ColumnMetaData object?
     *
     * @param input object to check
     * @return true if value, false if invalid
     */
    @Override
    public boolean isValid(Object input) {
        long epochMillisec;
        try {
            epochMillisec = Long.parseLong(input.toString());
        } catch (NumberFormatException e) {
            return false;
        }

        if (minValidTime != null && epochMillisec < minValidTime)
            return false;
        return !(maxValidTime != null && epochMillisec > maxValidTime);
    }

    @Override
    public TimeMetaData clone() {
        return new TimeMetaData(name, timeZone, minValidTime, maxValidTime);
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("TimeMetaData(name=\"").append(name).append("\",timeZone=").append(timeZone.getID());
        if (minValidTime != null)
            sb.append("minValidTime=").append(minValidTime);
        if (maxValidTime != null) {
            if (minValidTime != null)
                sb.append(",");
            sb.append("maxValidTime=").append(maxValidTime);
        }
        sb.append(")");
        return sb.toString();
    }
}
