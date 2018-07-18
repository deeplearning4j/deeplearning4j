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

package org.nd4j.linalg.indexing.conditions;

import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Condition for whether an element is NaN
 *
 * @author Adam Gibson
 */
public class IsNaN extends BaseCondition {

    public IsNaN() {
        super(-1);
    }

    /**
     * Returns condition ID for native side
     *
     * @return
     */
    @Override
    public int condtionNum() {
        return 9;
    }


    @Override
    public Boolean apply(Number input) {
        return Double.isNaN(input.doubleValue());
    }

    @Override
    public Boolean apply(IComplexNumber input) {
        return Double.isNaN(input.absoluteValue().doubleValue());
    }
}
