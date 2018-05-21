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

package org.datavec.api.transform.quality.columns;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * Quality of an Integer column
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class IntegerQuality extends ColumnQuality {

    private final long countNonInteger;

    public IntegerQuality(long countValid, long countInvalid, long countMissing, long countTotal,
                    long countNonInteger) {
        super(countValid, countInvalid, countMissing, countTotal);
        this.countNonInteger = countNonInteger;
    }


    public IntegerQuality add(IntegerQuality other) {
        return new IntegerQuality(countValid + other.countValid, countInvalid + other.countInvalid,
                        countMissing + other.countMissing, countTotal + other.countTotal,
                        countNonInteger + other.countNonInteger);
    }

    @Override
    public String toString() {
        return "IntegerQuality(" + super.toString() + ", countNonInteger=" + countNonInteger + ")";
    }

}
