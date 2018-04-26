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

/**
 * TimeQuality: quality of a time column
 *
 * @author Alex Black
 */
@Data
public class TimeQuality extends ColumnQuality {


    public TimeQuality(long countValid, long countInvalid, long countMissing, long countTotal) {
        super(countValid, countInvalid, countMissing, countTotal);
    }

    public TimeQuality() {
        this(0, 0, 0, 0);
    }

    @Override
    public String toString() {
        return "TimeQuality(" + super.toString() + ")";
    }

    public TimeQuality add(TimeQuality other) {
        return new TimeQuality(countValid + other.countValid, countInvalid + other.countInvalid,
                        countMissing + other.countMissing, countTotal + other.countTotal);
    }
}
