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
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Metadata for a long column
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class LongMetaData extends BaseColumnMetaData {

    //minAllowedValue/maxAllowedValue are nullable: null -> no restriction on minAllowedValue/maxAllowedValue values
    private final Long minAllowedValue;
    private final Long maxAllowedValue;

    public LongMetaData(String name) {
        this(name, null, null);
    }

    public LongMetaData(@JsonProperty("name") String name, @JsonProperty("minAllowedValue") Long minAllowedValue,
                    @JsonProperty("maxAllowedValue") Long maxAllowedValue) {
        super(name);
        this.minAllowedValue = minAllowedValue;
        this.maxAllowedValue = maxAllowedValue;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Long;
    }

    @Override
    public boolean isValid(Writable writable) {
        long value;
        if (writable instanceof IntWritable || writable instanceof LongWritable) {
            value = writable.toLong();
        } else {
            try {
                value = Long.parseLong(writable.toString());
            } catch (NumberFormatException e) {
                return false;
            }
        }
        if (minAllowedValue != null && value < minAllowedValue)
            return false;
        if (maxAllowedValue != null && value > maxAllowedValue)
            return false;

        return true;
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
        long value;
        try {
            value = Long.parseLong(input.toString());
        } catch (NumberFormatException e) {
            return false;
        }

        if (minAllowedValue != null && value < minAllowedValue)
            return false;
        if (maxAllowedValue != null && value > maxAllowedValue)
            return false;

        return true;
    }

    @Override
    public LongMetaData clone() {
        return new LongMetaData(name, minAllowedValue, maxAllowedValue);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("LongMetaData(name=\"").append(name).append("\",");
        if (minAllowedValue != null)
            sb.append("minAllowed=").append(minAllowedValue);
        if (maxAllowedValue != null) {
            if (minAllowedValue != null)
                sb.append(",");
            sb.append("maxAllowed=").append(maxAllowedValue);
        }
        sb.append(")");
        return sb.toString();
    }
}
