package org.datavec.api.transform.sequence;

import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

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
