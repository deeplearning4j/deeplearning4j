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
 * Created by Alex on 5/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class DoubleQuality extends ColumnQuality {

    private final long countNonReal;
    private final long countNaN;
    private final long countInfinite;

    public DoubleQuality() {
        this(0, 0, 0, 0, 0, 0, 0);
    }

    public DoubleQuality(long countValid, long countInvalid, long countMissing, long countTotal, long countNonReal,
                    long countNaN, long countInfinite) {
        super(countValid, countInvalid, countMissing, countTotal);
        this.countNonReal = countNonReal;
        this.countNaN = countNaN;
        this.countInfinite = countInfinite;
    }


    public DoubleQuality add(DoubleQuality other) {
        return new DoubleQuality(countValid + other.countValid, countInvalid + other.countInvalid,
                        countMissing + other.countMissing, countTotal + other.countTotal,
                        countNonReal + other.countNonReal, countNaN + other.countNaN,
                        countInfinite + other.countInfinite);
    }

    @Override
    public String toString() {
        return "DoubleQuality(" + super.toString() + ", countNonReal=" + countNonReal + ", countNaN=" + countNaN
                        + ", countInfinite=" + countInfinite + ")";
    }

}
