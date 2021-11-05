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

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

public class EqualsCondition extends BaseCondition {

    /**
     * Special constructor for pairwise boolean operations.
     */
    public EqualsCondition() {
        super(0.0);
    }

    public EqualsCondition(Number value) {
        super(value);
    }

    @Override
    public void setValue(Number value) {
        //no op where we can pass values in
    }

    /**
     * Returns condition ID for native side
     * Condition number is affected by:
     * https://github.com/eclipse/deeplearning4j/blob/0ba0f933a95d2dceeff3651bc540d03b5f3b1631/libnd4j/include/ops/ops.h#L2253
     *
     * @return
     */
    @Override
    public int conditionNum() {
        return 0;
    }

    @Override
    public Boolean apply(Number input) {
        if (Nd4j.dataType() == DataType.DOUBLE)
            return input.doubleValue() == value.doubleValue();
        else
            return input.floatValue() == value.floatValue();
    }
}
