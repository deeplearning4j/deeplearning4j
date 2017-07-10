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
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Metadata for an integer column
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class IntegerMetaData extends BaseColumnMetaData {

    //min/max are nullable: null -> no restriction on min/max values
    private final Integer minAllowedValue;
    private final Integer maxAllowedValue;

    public IntegerMetaData(String name) {
        this(name, null, null);
    }

    /**
     * @param min Min allowed value. If null: no restriction on min value value in this column
     * @param max Max allowed value. If null: no restiction on max value in this column
     */
    public IntegerMetaData(@JsonProperty("name") String name, @JsonProperty("minAllowedValue") Integer min,
                    @JsonProperty("maxAllowedValue") Integer max) {
        super(name);
        this.minAllowedValue = min;
        this.maxAllowedValue = max;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Integer;
    }

    @Override
    public boolean isValid(Writable writable) {
        int value;
        try {
            value = Integer.parseInt(writable.toString());
        } catch (NumberFormatException e) {
            return false;
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
        int value;
        try {
            value = Integer.parseInt(input.toString());
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
    public IntegerMetaData clone() {
        return new IntegerMetaData(name, minAllowedValue, maxAllowedValue);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("IntegerMetaData(name=\"").append(name).append("\",");
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
