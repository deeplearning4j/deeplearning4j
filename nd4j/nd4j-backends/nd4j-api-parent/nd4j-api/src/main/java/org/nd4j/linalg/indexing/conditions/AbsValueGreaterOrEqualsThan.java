/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.indexing.conditions;

import org.apache.commons.math3.util.FastMath;

public class AbsValueGreaterOrEqualsThan extends BaseCondition {

    /**
     * Special constructor for pairwise boolean operations.
     */
    public AbsValueGreaterOrEqualsThan() {
        super(0.0);
    }

    public AbsValueGreaterOrEqualsThan(Number value) {
        super(value);
    }

    @Override
    public void setValue(Number value) {
        //no op where we can pass values in
    }


    /**
     * Returns condition ID for native side
     *
     * @return
     */
    @Override
    public Conditions.ConditionMode conditionType() {
        return Conditions.ConditionMode.ABS_GREATER_OR_EQUAL;
    }

    @Override
    public Boolean apply(Number input) {
        return FastMath.abs(input.doubleValue()) >= value.doubleValue();
    }
}
