package io.skymind.echidna.api;

import org.canova.api.writable.Writable;
import io.skymind.echidna.api.schema.Schema;

import java.io.Serializable;
import java.util.List;

/**A Transform converts an example to another example, or a sequence to another sequence
 */
public interface Transform extends Serializable {

    /** Get the output schema for this transformation, given an input schema */
    Schema transform(Schema inputSchema);

    /** Set the input schema. Should be done automatically in TransformProcess, and is often necessary
     * to do {@link #map(List)}
     */
    void setInputSchema(Schema inputSchema);

    Schema getInputSchema();

    List<Writable> map(List<Writable> writables);

    /** Transform a sequence */
    List<List<Writable>> mapSequence(List<List<Writable>> sequence);

}
