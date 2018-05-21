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

package org.datavec.api.transform.sequence.window;

import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.serde.legacy.LegacyMappingHelper;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.List;

/**
 * A WindowFunction splits a sequence into a set of
 * (possibly overlapping) sub-sequences.
 * It is a general-purpose interface that can support
 * many different types of
 *
 * Typically used for example with a transform such as {@link ReduceSequenceByWindowTransform}
 *
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class",
        defaultImpl = LegacyMappingHelper.WindowFunctionHelper.class)
public interface WindowFunction extends Serializable {

    /**
     * Apply the windowing function to the given sequence
     * @param sequence the input sequence
     * @return the sequence with the window function applied
     */
    List<List<List<Writable>>> applyToSequence(List<List<Writable>> sequence);

    /**
     *
     * @param schema
     */
    void setInputSchema(Schema schema);

    /**
     *
     * @return
     */
    Schema getInputSchema();

    /** Get the output schema, given the input schema. Typically the output schema is the same as the input schema,
     * but not necessarily (for example, if the window function adds columns for the window start/end times)
     * @param inputSchema    Schema of the input data
     * @return Schema of the output windows
     */
    Schema transform(Schema inputSchema);


}
