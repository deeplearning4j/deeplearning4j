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

package org.datavec.api.transform.transform.time;

import lombok.Data;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.TimeMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.concurrent.TimeUnit;

/**
 * Transform math op on a time column
 * <p>
 * Note: only the following MathOps are supported: Add, Subtract, ScalarMin, ScalarMax<br>
 * For ScalarMin/Max, the TimeUnit must be milliseconds - i.e., value must be in epoch millisecond format
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"inputSchema", "columnNumber", "asMilliseconds"})
@Data
public class TimeMathOpTransform extends BaseColumnTransform {

    private final MathOp mathOp;
    private final long timeQuantity;
    private final TimeUnit timeUnit;
    private final long asMilliseconds;

    public TimeMathOpTransform(@JsonProperty("columnName") String columnName, @JsonProperty("mathOp") MathOp mathOp,
                    @JsonProperty("timeQuantity") long timeQuantity, @JsonProperty("timeUnit") TimeUnit timeUnit) {
        super(columnName);
        if (mathOp != MathOp.Add && mathOp != MathOp.Subtract && mathOp != MathOp.ScalarMin
                        && mathOp != MathOp.ScalarMax) {
            throw new IllegalArgumentException("Invalid MathOp: only Add/Subtract/ScalarMin/ScalarMax supported");
        }
        if ((mathOp == MathOp.ScalarMin || mathOp == MathOp.ScalarMax) && timeUnit != TimeUnit.MILLISECONDS) {
            throw new IllegalArgumentException(
                            "Only valid time unit for ScalarMin/Max is Milliseconds (i.e., timestamp format)");
        }

        this.mathOp = mathOp;
        this.timeQuantity = timeQuantity;
        this.timeUnit = timeUnit;
        this.asMilliseconds = TimeUnit.MILLISECONDS.convert(timeQuantity, timeUnit);
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        if (!(oldColumnType instanceof TimeMetaData))
            throw new IllegalStateException("Cannot execute TimeMathOpTransform on column with type " + oldColumnType);

        return new TimeMetaData(newName, ((TimeMetaData) oldColumnType).getTimeZone());
    }

    @Override
    public Writable map(Writable columnWritable) {
        long currTime = columnWritable.toLong();
        switch (mathOp) {
            case Add:
                return new LongWritable(currTime + asMilliseconds);
            case Subtract:
                return new LongWritable(currTime - asMilliseconds);
            case ScalarMax:
                return new LongWritable(Math.max(asMilliseconds, currTime));
            case ScalarMin:
                return new LongWritable(Math.min(asMilliseconds, currTime));
            default:
                throw new RuntimeException("Invalid MathOp for TimeMathOpTransform: " + mathOp);
        }
    }

    @Override
    public String toString() {
        return "TimeMathOpTransform(mathOp=" + mathOp + "," + timeQuantity + "-" + timeUnit + ")";
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
