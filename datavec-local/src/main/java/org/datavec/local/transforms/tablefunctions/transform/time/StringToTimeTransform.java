/*
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

package org.datavec.local.transforms.tablefunctions.transform.time;

import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.TimeMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.TimeZone;

/**
 * Convert a String column to a time column by parsing the date/time String, using a JodaTime.
 * <p>
 * Time format is specified as per http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
 *
 * @author Alex Black
 */
public class StringToTimeTransform extends BaseColumnTransform {

    private final String timeFormat;
    private final DateTimeZone timeZone;
    private final Long minValidTime;
    private final Long maxValidTime;

    private transient DateTimeFormatter formatter;

    /**
     * @param columnName Name of the String column
     * @param timeFormat Time format, as per http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
     * @param timeZone   Timezone for time parsing
     */
    public StringToTimeTransform(String columnName, String timeFormat, TimeZone timeZone) {
        this(columnName, timeFormat, timeZone, null, null);
    }

    /**
     * @param columnName Name of the String column
     * @param timeFormat Time format, as per http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
     * @param timeZone   Timezone for time parsing
     */
    public StringToTimeTransform(String columnName, String timeFormat, DateTimeZone timeZone) {
        this(columnName, timeFormat, timeZone, null, null);
    }

    /**
     * @param columnName   Name of the String column
     * @param timeFormat   Time format, as per http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
     * @param timeZone     Timezone for time parsing
     * @param minValidTime Min valid time (epoch millisecond format). If null: no restriction in min valid time
     * @param maxValidTime Max valid time (epoch millisecond format). If null: no restriction in max valid time
     */
    public StringToTimeTransform(@JsonProperty("columnName") String columnName, @JsonProperty("timeFormat") String timeFormat,
                                 @JsonProperty("timeZone") TimeZone timeZone, @JsonProperty("minValidTime") Long minValidTime,
                                 @JsonProperty("maxValidTime") Long maxValidTime) {
        this(columnName, timeFormat, DateTimeZone.forTimeZone(timeZone), minValidTime, maxValidTime);
    }

    /**
     * @param columnName   Name of the String column
     * @param timeFormat   Time format, as per http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
     * @param timeZone     Timezone for time parsing
     * @param minValidTime Min valid time (epoch millisecond format). If null: no restriction in min valid time
     * @param maxValidTime Max valid time (epoch millisecond format). If null: no restriction in max valid time
     */
    public StringToTimeTransform(String columnName, String timeFormat, DateTimeZone timeZone, Long minValidTime, Long maxValidTime) {
        super(columnName);
        this.timeFormat = timeFormat;
        this.timeZone = timeZone;
        this.minValidTime = minValidTime;
        this.maxValidTime = maxValidTime;

        this.formatter = DateTimeFormat.forPattern(timeFormat).withZone(timeZone);
    }


    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        return new TimeMetaData(newName, timeZone, minValidTime, maxValidTime);
    }

    @Override
    public Writable map(Writable columnWritable) {
        String str = columnWritable.toString();
        long time = formatter.parseMillis(str);
        return new LongWritable(time);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("StringToTimeTransform(timeZone=").append(timeZone);
        if (minValidTime != null) sb.append(",minValidTime=").append(minValidTime);
        if (maxValidTime != null) {
            if (minValidTime != null) sb.append(",");
            sb.append("maxValidTime=").append(maxValidTime);
        }
        sb.append(")");
        return sb.toString();
    }

    //Custom serialization methods, because Joda Time doesn't allow DateTimeFormatter objects to be serialized :(
    private void writeObject(ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        formatter = DateTimeFormat.forPattern(timeFormat).withZone(timeZone);
    }
}
