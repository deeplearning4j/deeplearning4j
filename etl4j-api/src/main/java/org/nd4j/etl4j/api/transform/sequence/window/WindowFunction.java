package org.nd4j.etl4j.api.transform.sequence.window;

import org.nd4j.etl4j.api.writable.Writable;
import org.nd4j.etl4j.api.transform.schema.Schema;

import java.io.Serializable;
import java.util.List;

/**
 * A WindowFunction splits a sequence into a set of (possibly overlapping) sub-sequences.
 * It is a general-purpose interface that can support many different types of
 *
 * Typically used for example with a transform such as {@link ReduceSequenceByWindowTransform}
 *
 * @author Alex Black
 */
public interface WindowFunction extends Serializable {

    List<List<List<Writable>>> applyToSequence(List<List<Writable>> sequence);

    void setInputSchema(Schema schema);

    Schema getInputSchema();

    /** Get the output schema, given the input schema. Typically the output schema is the same as the input schema,
     * but not necessarily (for example, if the window function adds columns for the window start/end times)
     * @param inputSchema    Schema of the input data
     * @return Schema of the output windows
     */
    Schema transform(Schema inputSchema);


}
