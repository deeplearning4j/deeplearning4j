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

package org.datavec.api.transform.quality.columns;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * Quality of a Long column
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class LongQuality extends ColumnQuality {

    private final long countNonLong;

    public LongQuality() {
        this(0, 0, 0, 0, 0);
    }

    public LongQuality(long countValid, long countInvalid, long countMissing, long countTotal, long countNonLong) {
        super(countValid, countInvalid, countMissing, countTotal);
        this.countNonLong = countNonLong;
    }


    public LongQuality add(LongQuality other) {
        return new LongQuality(countValid + other.countValid, countInvalid + other.countInvalid,
                        countMissing + other.countMissing, countTotal + other.countTotal,
                        countNonLong + other.countNonLong);
    }

    @Override
    public String toString() {
        return "LongQuality(" + super.toString() + ", countNonLong=" + countNonLong + ")";
    }

}
