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

package org.datavec.api.transform.condition;

import org.datavec.api.transform.ColumnOp;
import org.datavec.api.transform.condition.sequence.SequenceLengthCondition;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.datavec.api.transform.condition.column.*;
import org.datavec.api.transform.condition.string.StringRegexColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.List;

/**
 * The Condition interface defines a binary state that either holds/is satisfied for an example/sequence,
 * or does not hold.<br>
 * Example: number greater than x, String is one of {X,Y,Z}, etc.<br>
 * Typical uses for conditions: filtering, conditional replacement, conditional reduction, etc
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonSubTypes(value = {
                @JsonSubTypes.Type(value = CategoricalColumnCondition.class, name = "CategoricalColumnCondition"),
                @JsonSubTypes.Type(value = DoubleColumnCondition.class, name = "DoubleColumnCondition"),
                @JsonSubTypes.Type(value = IntegerColumnCondition.class, name = "IntegerColumnCondition"),
                @JsonSubTypes.Type(value = LongColumnCondition.class, name = "LongColumnCondition"),
                @JsonSubTypes.Type(value = NullWritableColumnCondition.class, name = "NullWritableColumnCondition"),
                @JsonSubTypes.Type(value = StringColumnCondition.class, name = "StringColumnCondition"),
                @JsonSubTypes.Type(value = TimeColumnCondition.class, name = "TimeColumnCondition"),
                @JsonSubTypes.Type(value = StringRegexColumnCondition.class, name = "StringRegexColumnCondition"),
                @JsonSubTypes.Type(value = BooleanCondition.class, name = "BooleanCondition"),
                @JsonSubTypes.Type(value = NaNColumnCondition.class, name = "NaNColumnCondition"),
                @JsonSubTypes.Type(value = InfiniteColumnCondition.class, name = "InfiniteColumnCondition"),
                @JsonSubTypes.Type(value = SequenceLengthCondition.class, name = "SequenceLengthCondition")})
public interface Condition extends Serializable, ColumnOp {

    /**
     * Is the condition satisfied for the current input/example?<br>
     * Returns true if condition is satisfied, or false otherwise.
     *
     * @param list Current example
     * @return true if condition satisfied, false otherwise
     */
    boolean condition(List<Writable> list);

    /**
     * Condition on arbitrary input
     * @param input the input to return
     *              the condition for
     * @return true if the condition is met
     * false otherwise
     */
    boolean condition(Object input);


    /**
     * Is the condition satisfied for the current input/sequence?<br>
     * Returns true if condition is satisfied, or false otherwise.
     *
     * @param sequence Current sequence
     * @return true if condition satisfied, false otherwise
     */
    boolean conditionSequence(List<List<Writable>> sequence);

    /**
     * Condition on arbitrary input
     * @param sequence the sequence to
     *                 do a condition on
     * @return true if the condition for the sequence is met false otherwise
     */
    boolean conditionSequence(Object sequence);


    /**
     * Setter for the input schema
     * @param schema
     */
    void setInputSchema(Schema schema);

    /**
     * Getter for the input schema
     * @return
     */
    Schema getInputSchema();


}
