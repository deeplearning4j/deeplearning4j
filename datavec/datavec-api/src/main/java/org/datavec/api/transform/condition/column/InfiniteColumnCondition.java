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

package org.datavec.api.transform.condition.column;

import lombok.Data;
import org.datavec.api.transform.condition.SequenceConditionMode;
import org.datavec.api.writable.Writable;

/**
 * A column condition that simply checks whether a floating point value is infinite
 *
 * @author Alex Black
 */
@Data
public class InfiniteColumnCondition extends BaseColumnCondition {

    /**
     * @param columnName Column check for the condition
     */
    public InfiniteColumnCondition(String columnName) {
        this(columnName, DEFAULT_SEQUENCE_CONDITION_MODE);
    }

    /**
     * @param columnName            Column to check for the condition
     * @param sequenceConditionMode Sequence condition mode
     */
    public InfiniteColumnCondition(String columnName, SequenceConditionMode sequenceConditionMode) {
        super(columnName, sequenceConditionMode);
    }

    @Override
    public boolean columnCondition(Writable writable) {
        return Double.isInfinite(writable.toDouble());
    }

    @Override
    public boolean condition(Object input) {
        return Double.isInfinite(((Number) input).doubleValue());
    }

    @Override
    public String toString() {
        return "InfiniteColumnCondition()";
    }

}
