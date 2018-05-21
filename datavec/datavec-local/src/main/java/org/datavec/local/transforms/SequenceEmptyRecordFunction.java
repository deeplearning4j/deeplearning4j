package org.datavec.local.transforms;

import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;

import java.util.List;

/**
 * Used for filtering empty records
 *
 * @author Adam Gibson
 */
public class SequenceEmptyRecordFunction implements Function<List<List<Writable>>, Boolean> {
    @Override
    public Boolean apply(List<List<Writable>> v1) {
        return v1.isEmpty();
    }
}
