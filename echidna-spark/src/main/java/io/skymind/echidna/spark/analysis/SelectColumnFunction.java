package io.skymind.echidna.spark.analysis;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.canova.api.writable.Writable;

import java.util.List;

/**
 * Select out the value from a single column
 */
@AllArgsConstructor
public class SelectColumnFunction implements Function<List<Writable>,Writable> {

    private final int column;

    @Override
    public Writable call(List<Writable> writables) throws Exception {
        return writables.get(column);
    }
}
