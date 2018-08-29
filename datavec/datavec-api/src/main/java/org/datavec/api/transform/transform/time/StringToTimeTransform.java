/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.transform.transform.time;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.TimeMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.DateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.TimeZone;
import java.util.regex.Pattern;

/**
 * Convert a String column to a time column by parsing the date/time String, using a JodaTime.
 * <p>
 * Time format is specified as per <a href="http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html">http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html</a>
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"formatter", "formatters"})
@JsonIgnoreProperties({"formatters", "formatter"})
public class StringToTimeTransform extends BaseColumnTransform {

    private final String timeFormat;
    private final DateTimeZone timeZone;
    private final Long minValidTime;
    private final Long maxValidTime;
    //formats from: http://www.java2s.com/Tutorials/Java/Data_Type_How_to/Legacy_Date_Format/Guess_the_format_pattern_based_on_date_value.htm
    //2017-09-21T17:06:29.064687
    // 12/1/2010 11:21
    private static final String[] formats = { "YYYY-MM-dd'T'HH:mm:ss","YYYY-MM-dd","YYYY-MM-dd'T'HH:mm:ss'Z'",
            "YYYY-MM-dd'T'HH:mm:ssZ",
            "YYYY-MM-dd'T'HH:mm:ss.SSS'Z'", "YYYY-MM-dd'T'HH:mm:ss.SSSZ",
            "YYYY-MM-dd HH:mm:ss", "MM/dd/YYYY HH:mm:ss",
            "MM/dd/YYYY'T'HH:mm:ss.SSS'Z'", "MM/dd/YYYY'T'HH:mm:ss.SSSZ",
            "MM/dd/YYYY'T'HH:mm:ss.SSS", "MM/dd/YYYY'T'HH:mm:ssZ",
            "MM/dd/YYYY'T'HH:mm:ss", "YYYY:MM:dd HH:mm:ss", "YYYYMMdd", "YYYY-MM-dd HH:mm:ss","MM/dd/YYYY HH:mm",

    };
    private transient DateTimeFormatter[] formatters;

    private transient DateTimeFormatter formatter;


    /**
     * Instantiate this without a time format specified.
     * If this constructor is used, this transform will be allowed
     * to handle several common transforms as defined in the
     * static formats array.
     *
     *
     * @param columnName Name of the String column
     * @param timeZone   Timezone for time parsing
     */
    public StringToTimeTransform(String columnName,  TimeZone timeZone) {
        this(columnName, null, timeZone, null, null);
    }

    /**
     * @param columnName Name of the String column
     * @param timeFormat Time format, as per <a href="http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html">http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html</a>
     * @param timeZone   Timezone for time parsing
     */
    public StringToTimeTransform(String columnName, String timeFormat, TimeZone timeZone) {
        this(columnName, timeFormat, timeZone, null, null);
    }


    /**
     * Instantiate this without a time format specified.
     * If this constructor is used, this transform will be allowed
     * to handle several common transforms as defined in the
     * static formats array.
     *
     *
     * @param columnName Name of the String column
     * @param timeZone   Timezone for time parsing
     */
    public StringToTimeTransform(String columnName, DateTimeZone timeZone) {
        this(columnName, null, timeZone, null, null);
    }


    /**
     * @param columnName Name of the String column
     * @param timeFormat Time format, as per <a href="http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html">http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html</a>
     * @param timeZone   Timezone for time parsing
     */
    public StringToTimeTransform(String columnName, String timeFormat, DateTimeZone timeZone) {
        this(columnName, timeFormat, timeZone, null, null);
    }

    /**
     * @param columnName   Name of the String column
     * @param timeFormat   Time format, as per <a href="http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html">http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html</a>
     * @param timeZone     Timezone for time parsing
     * @param minValidTime Min valid time (epoch millisecond format). If null: no restriction in min valid time
     * @param maxValidTime Max valid time (epoch millisecond format). If null: no restriction in max valid time
     */
    public StringToTimeTransform(@JsonProperty("columnName") String columnName,
                                 @JsonProperty("timeFormat") String timeFormat, @JsonProperty("timeZone") TimeZone timeZone,
                                 @JsonProperty("minValidTime") Long minValidTime, @JsonProperty("maxValidTime") Long maxValidTime) {
        this(columnName, timeFormat, DateTimeZone.forTimeZone(timeZone), minValidTime, maxValidTime);
    }

    /**
     * @param columnName   Name of the String column
     * @param timeFormat   Time format, as per <a href="http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html">http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html</a>
     * @param timeZone     Timezone for time parsing
     * @param minValidTime Min valid time (epoch millisecond format). If null: no restriction in min valid time
     * @param maxValidTime Max valid time (epoch millisecond format). If null: no restriction in max valid time
     */
    public StringToTimeTransform(String columnName, String timeFormat, DateTimeZone timeZone, Long minValidTime,
                                 Long maxValidTime) {
        super(columnName);
        this.timeFormat = timeFormat;
        this.timeZone = timeZone;
        this.minValidTime = minValidTime;
        this.maxValidTime = maxValidTime;
        if(timeFormat != null)
            this.formatter = DateTimeFormat.forPattern(timeFormat).withZone(timeZone);
        else {
            List<DateTimeFormatter> dateFormatList = new ArrayList<>();
            formatters = new DateTimeFormatter[formats.length];
            for(int i = 0; i < formatters.length; i++) {
                dateFormatList.add(DateTimeFormat.forPattern(formats[i]).withZone(timeZone));
            }

            formatters = dateFormatList.toArray(new DateTimeFormatter[dateFormatList.size()]);
        }
    }


    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        return new TimeMetaData(newName, timeZone, minValidTime, maxValidTime);
    }

    @Override
    public Writable map(Writable columnWritable) {
        String str = columnWritable.toString().trim();
        if(str.contains("'T'")) {
            str = str.replaceFirst("'T'","T");
        }



        if(formatter == null) {
            long result = -1;
            if(Pattern.compile("\\.[0-9]+").matcher(str).find()) {
                str = str.replaceAll("\\.[0-9]+","");
            }


            for(DateTimeFormatter formatter : formatters) {
                try {
                    result = formatter.parseMillis(str);
                    return new LongWritable(result);
                }catch (Exception e) {

                }


            }

            if(result  < 0) {
                throw new IllegalStateException("Unable to parse date time " + str);
            }
        }
        else {
            long time = formatter.parseMillis(str);
            return new LongWritable(time);
        }

        throw new IllegalStateException("Unable to parse date time " + str);

    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("StringToTimeTransform(timeZone=").append(timeZone);
        if (minValidTime != null)
            sb.append(",minValidTime=").append(minValidTime);
        if (maxValidTime != null) {
            if (minValidTime != null)
                sb.append(",");
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
        if(timeFormat != null)
            formatter = DateTimeFormat.forPattern(timeFormat).withZone(timeZone);
        else {
            List<DateTimeFormatter> dateFormatList = new ArrayList<>();
            formatters = new DateTimeFormatter[formats.length];
            for(int i = 0; i < formatters.length; i++) {
                dateFormatList.add(DateTimeFormat.forPattern(formats[i]).withZone(timeZone));
            }

            formatters = dateFormatList.toArray(new DateTimeFormatter[dateFormatList.size()]);
        }
    }

    /**
     * Transform an object
     * in to another object
     *
     * @param input the record to transform
     * @return the transformed writable
     */
    @Override
    public Object map(Object input) {
        return null;
    }
}
