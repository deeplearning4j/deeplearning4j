package org.datavec.local.transforms.functions;

import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;

import java.util.List;

/**
 * Used for filtering empty records
 *
 * @author Adam Gibson
 */
public class EmptyRecordFunction implements Function<List<Writable>, Boolean> {
    @Override
    public Boolean apply(List<Writable> v1) {
        return v1.isEmpty();
    }
}
