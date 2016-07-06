package io.skymind.echidna.api.sequence;

import org.canova.api.writable.Writable;
import io.skymind.echidna.api.schema.Schema;

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
