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

package org.datavec.api.transform.metadata;

import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.writable.Writable;
import lombok.Data;

/**
 * Metadata for an long column
 *
 * @author Alex Black
 */
@Data
public class LongMetaData extends BaseColumnMetaData {

    //min/max are nullable: null -> no restriction on min/max values
    private final Long min;
    private final Long max;

    public LongMetaData(String name) {
        this(name, null, null);
    }

    public LongMetaData(String name, Long min, Long max) {
        super(name);
        this.min = min;
        this.max = max;
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
        if (min != null && value < min) return false;
        if (max != null && value > max) return false;

        return true;
    }

    @Override
    public LongMetaData clone() {
        return new LongMetaData(name, min, max);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("LongMetaData(");
        if (min != null) sb.append("minAllowed=").append(min);
        if (max != null) {
            if (min != null) sb.append(",");
            sb.append("maxAllowed=").append(max);
        }
        sb.append(")");
        return sb.toString();
    }
}
