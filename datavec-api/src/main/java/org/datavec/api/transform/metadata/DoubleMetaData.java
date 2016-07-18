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

import org.datavec.api.transform.ColumnType;
import lombok.Data;
import org.datavec.api.writable.Writable;

/**
 * MetaData for a double column.
 *
 * @author Alex Black
 */
@Data
public class DoubleMetaData extends BaseColumnMetaData {

    //min/max are nullable: null -> no restriction on min/max values
    private final Double min;
    private final Double max;
    private final boolean allowNaN;
    private final boolean allowInfinite;

    public DoubleMetaData(String name) {
        this(name, null, null, false, false);
    }

    /**
     * @param min Min allowed value. If null: no restriction on min value value in this column
     * @param max Max allowed value. If null: no restiction on max value in this column
     */
    public DoubleMetaData(String name, Double min, Double max) {
        this(name, min, max, false, false);
    }

    /**
     * @param min           Min allowed value. If null: no restriction on min value value in this column
     * @param max           Max allowed value. If null: no restiction on max value in this column
     * @param allowNaN      Are NaN values ok?
     * @param allowInfinite Are +/- infinite values ok?
     */
    public DoubleMetaData(String name, Double min, Double max, boolean allowNaN, boolean allowInfinite) {
        super(name);
        this.min = min;
        this.max = max;
        this.allowNaN = allowNaN;
        this.allowInfinite = allowInfinite;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Double;
    }

    @Override
    public boolean isValid(Writable writable) {
        double d;
        try {
            d = writable.toDouble();
        } catch (Exception e) {
            return false;
        }

        if (allowNaN && Double.isNaN(d)) return true;
        if (allowInfinite && Double.isInfinite(d)) return true;

        if (min != null && d < min) return false;
        if (max != null && d > max) return false;

        return true;
    }

    @Override
    public DoubleMetaData clone() {
        return new DoubleMetaData(name, min, max, allowNaN, allowInfinite);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("DoubleMetaData(");
        boolean needComma = false;
        if (min != null) {
            sb.append("minAllowed=").append(min);
            needComma = true;
        }
        if (max != null) {
            if (needComma) sb.append(",");
            sb.append("maxAllowed=").append(max);
            needComma = true;
        }
        if (needComma) sb.append(",");
        sb.append("allowNaN=").append(allowNaN).append(",allowInfinite=").append(allowInfinite).append(")");
        return sb.toString();
    }
}
