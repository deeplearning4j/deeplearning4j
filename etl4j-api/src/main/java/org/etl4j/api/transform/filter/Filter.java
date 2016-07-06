package io.skymind.echidna.api.filter;

import org.canova.api.writable.Writable;
import io.skymind.echidna.api.schema.Schema;

import java.io.Serializable;
import java.util.List;

/**
 * Filter: a method of removing examples (or sequences) according to some condition
 *
 * @author Alex Black
 */
public interface Filter extends Serializable {

    /**
     * @param writables Example
     * @return true if example should be removed, false to keep
     */
    boolean removeExample(List<Writable> writables);

    /**
     * @param sequence sequence example
     * @return true if example should be removed, false to keep
     */
    boolean removeSequence(List<List<Writable>> sequence);

    void setInputSchema(Schema schema);

    Schema getInputSchema();

}
