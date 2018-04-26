/*-
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

import lombok.Data;
import org.datavec.api.writable.*;

/**
 * A Condition that applies to a single column.
 * Whenever the specified value is invalid according to the schema, the condition applies.
 * <p>
 * For example, if a Writable contains String values in an Integer column (and these cannot be parsed to an integer), then
 * the condition would return true, as these values are invalid according to the schema.
 *
 * @author Alex Black
 */
@Data
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
        if (input instanceof String) {
            return !schema.getMetaData(columnIdx).isValid(new Text(input.toString()));
        } else if (input instanceof Double) {
            Double d = (Double) input;
            return !schema.getMetaData(columnIdx).isValid(new DoubleWritable(d));
        } else if (input instanceof Integer) {
            Integer i = (Integer) input;
            return !schema.getMetaData(columnIdx).isValid(new IntWritable(i));
        } else if (input instanceof Long) {
            Long l = (Long) input;
            return !schema.getMetaData(columnIdx).isValid(new LongWritable(l));
        } else if (input instanceof Boolean) {
            Boolean b = (Boolean) input;
            return !schema.getMetaData(columnIdx).isValid(new BooleanWritable(b));
        }

        else
            throw new IllegalStateException("Illegal type " + input.getClass().toString());
    }
}
