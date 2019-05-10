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
