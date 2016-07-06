package io.skymind.echidna.api.sequence;

import org.canova.api.writable.Writable;
import io.skymind.echidna.api.schema.Schema;

import java.io.Serializable;
import java.util.List;

/**
 * Created by Alex on 16/03/2016.
 */
public interface SequenceSplit extends Serializable {

    List<List<List<Writable>>> split(List<List<Writable>> sequence);

    void setInputSchema(Schema inputSchema);

    Schema getInputSchema();

}
