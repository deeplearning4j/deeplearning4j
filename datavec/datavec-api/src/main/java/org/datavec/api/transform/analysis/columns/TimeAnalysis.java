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

package org.datavec.api.transform.analysis.columns;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.ColumnType;
import org.joda.time.DateTimeZone;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

/**
 * Analysis for Time columns
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
@NoArgsConstructor //For Jackson deserialization
public class TimeAnalysis extends NumericalColumnAnalysis {

    private static final DateTimeFormatter formatter =
                    DateTimeFormat.forPattern("YYYY-MM-dd HH:mm:ss.SSS zzz").withZone(DateTimeZone.UTC);
    private long min;
    private long max;

    private TimeAnalysis(Builder builder) {
        super(builder);
        this.min = builder.min;
        this.max = builder.max;
    }

    @Override
    public String toString() {
        return "TimeAnalysis(min=" + min + " (" + formatter.print(min) + "),max=" + max + " (" + formatter.print(max)
                        + ")," + super.toString() + ")";
    }

    @Override
    public double getMinDouble() {
        return min;
    }

    @Override
    public double getMaxDouble() {
        return max;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Long;
    }

    public static class Builder extends NumericalColumnAnalysis.Builder<Builder> {
        private long min;
        private long max;

        public Builder min(long min) {
            this.min = min;
            return this;
        }

        public Builder max(long max) {
            this.max = max;
            return this;
        }

        public TimeAnalysis build() {
            return new TimeAnalysis(this);
        }
    }

}
