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
 * MetaData for a Float column.
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class FloatMetaData extends BaseColumnMetaData {

    //minAllowedValue/maxAllowedValue are nullable: null -> no restriction on minAllowedValue/maxAllowedValue values
    private final Float minAllowedValue;
    private final Float maxAllowedValue;
    private final boolean allowNaN;
    private final boolean allowInfinite;

    public FloatMetaData(String name) {
        this(name, null, null, false, false);
    }

    /**
     * @param minAllowedValue Min allowed value. If null: no restriction on minAllowedValue value value in this column
     * @param maxAllowedValue Max allowed value. If null: no restiction on maxAllowedValue value in this column
     */
    public FloatMetaData(@JsonProperty("name") String name, @JsonProperty("minAllowedValue") Float minAllowedValue,
                    @JsonProperty("maxAllowedValue") Float maxAllowedValue) {
        this(name, minAllowedValue, maxAllowedValue, false, false);
    }

    /**
     * @param min           Min allowed value. If null: no restriction on minAllowedValue value value in this column
     * @param maxAllowedValue           Max allowed value. If null: no restiction on maxAllowedValue value in this column
     * @param allowNaN      Are NaN values ok?
     * @param allowInfinite Are +/- infinite values ok?
     */
    public FloatMetaData(String name, Float min, Float maxAllowedValue, boolean allowNaN, boolean allowInfinite) {
        super(name);
        this.minAllowedValue = min;
        this.maxAllowedValue = maxAllowedValue;
        this.allowNaN = allowNaN;
        this.allowInfinite = allowInfinite;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Float;
    }

    @Override
    public boolean isValid(Writable writable) {
        Float d;
        try {
            d = writable.toFloat();
        } catch (Exception e) {
            return false;
        }

        if (allowNaN && Float.isNaN(d))
            return true;
        if (allowInfinite && Float.isInfinite(d))
            return true;

        if (minAllowedValue != null && d < minAllowedValue)
            return false;
        if (maxAllowedValue != null && d > maxAllowedValue)
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
        Float d;
        try {
            d = (Float) input;
        } catch (Exception e) {
            return false;
        }

        if (allowNaN && Float.isNaN(d))
            return true;
        if (allowInfinite && Float.isInfinite(d))
            return true;

        if (minAllowedValue != null && d < minAllowedValue)
            return false;
        if (maxAllowedValue != null && d > maxAllowedValue)
            return false;

        return true;
    }

    @Override
    public FloatMetaData clone() {
        return new FloatMetaData(name, minAllowedValue, maxAllowedValue, allowNaN, allowInfinite);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("FloatMetaData(name=\"").append(name).append("\",");
        boolean needComma = false;
        if (minAllowedValue != null) {
            sb.append("minAllowed=").append(minAllowedValue);
            needComma = true;
        }
        if (maxAllowedValue != null) {
            if (needComma)
                sb.append(",");
            sb.append("maxAllowed=").append(maxAllowedValue);
            needComma = true;
        }
        if (needComma)
            sb.append(",");
        sb.append("allowNaN=").append(allowNaN).append(",allowInfinite=").append(allowInfinite).append(")");
        return sb.toString();
    }
}
