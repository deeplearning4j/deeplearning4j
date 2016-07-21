/*
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.condition.column;

import org.datavec.api.writable.Writable;

/**
 * A Condition that applies to a single column.
 * Whenever the specified value is invalid according to the schema, the condition applies.
 * <p>
 * For example, if a Writable contains String values in an Integer column (and these cannot be parsed to an integer), then
 * the condition would return true, as these values are invalid according to the schema.
 *
 * @author Alex Black
 */
public class InvalidValueColumnCondition extends BaseColumnCondition {


    public InvalidValueColumnCondition(String columnName) {
        super(columnName, DEFAULT_SEQUENCE_CONDITION_MODE);
    }

    @Override
    public boolean columnCondition(Writable writable) {
        return !schema.getMetaData(columnIdx).isValid(writable);
    }

    @Override
    public String toString() {
        return "InvalidValueColumnCondition(columnName=\"" + columnName + "\")";
    }
}
