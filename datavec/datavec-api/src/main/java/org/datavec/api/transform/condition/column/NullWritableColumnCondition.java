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
import lombok.EqualsAndHashCode;
import org.datavec.api.writable.NullWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Condition that applies to the values in any column. Specifically, condition is true
 * if the Writable value is a NullWritable, and false for any other value
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class NullWritableColumnCondition extends BaseColumnCondition {

    public NullWritableColumnCondition(@JsonProperty("columnName") String columnName) {
        super(columnName, DEFAULT_SEQUENCE_CONDITION_MODE);
    }

    @Override
    public boolean columnCondition(Writable writable) {
        return writable instanceof NullWritable;
    }

    @Override
    public String toString() {
        return "NullWritableColumnCondition()";
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
        return input == null;
    }
}
