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
import org.datavec.api.writable.BooleanWritable;
import org.datavec.api.writable.Writable;

/**
 * Created by agibsonccc on 11/26/16.
 */
@Data
public class BooleanColumnCondition extends BaseColumnCondition {
    protected BooleanColumnCondition(String columnName, SequenceConditionMode sequenceConditionMode) {
        super(columnName, sequenceConditionMode);
    }

    /**
     * Returns whether the given element
     * meets the condition set by this operation
     *
     * @param writable the element to test
     * @return true if the condition is met
     * false otherwise
     */
    @Override
    public boolean columnCondition(Writable writable) {
        BooleanWritable booleanWritable = (BooleanWritable) writable;
        return booleanWritable.get();
    }

    /**
     * Condition on arbitrary input
     *
     * @param input the input to return
     *              the condition for
     * @return true if the condition is met
     * false otherwise
     */
    @Override
    public boolean condition(Object input) {
        Boolean bool = (Boolean) input;
        return bool;
    }

    @Override
    public String toString() {
        return getClass().toString();
    }
}
