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

package org.datavec.api.transform.condition.string;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.condition.SequenceConditionMode;
import org.datavec.api.transform.condition.column.BaseColumnCondition;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Condition that applies to the values in a String column, using a provided regex.
 * Condition return true if the String matches the regex, or false otherwise<br>
 * <p>
 * <b>Note:</b> Uses Writable.toString(), hence can potentially be applied to non-String columns
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class StringRegexColumnCondition extends BaseColumnCondition {

    private final String regex;

    public StringRegexColumnCondition(@JsonProperty("columnName") String columnName,
                    @JsonProperty("regex") String regex) {
        this(columnName, regex, DEFAULT_SEQUENCE_CONDITION_MODE);
    }

    public StringRegexColumnCondition(String columnName, String regex, SequenceConditionMode sequenceConditionMode) {
        super(columnName, sequenceConditionMode);
        this.regex = regex;
    }

    @Override
    public boolean columnCondition(Writable writable) {
        return writable.toString().matches(regex);
    }

    @Override
    public String toString() {
        return "StringRegexColumnCondition(columnName=\"" + columnName + "\",regex=\"" + regex + "\")";
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
        return input.toString().matches(regex);
    }

}
