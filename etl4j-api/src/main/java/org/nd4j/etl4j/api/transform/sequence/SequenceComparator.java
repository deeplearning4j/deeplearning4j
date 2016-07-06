package org.nd4j.etl4j.api.transform.sequence;

import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.schema.Schema;

import java.io.Serializable;
import java.util.Comparator;
import java.util.List;

/**
 * Compare the time steps of a sequence
 * Created by Alex on 11/03/2016.
 */
public interface SequenceComparator extends Comparator<List<Writable>>, Serializable {

    void setSchema(Schema sequenceSchema);

}
