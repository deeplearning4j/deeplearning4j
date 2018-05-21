package org.datavec.local.transforms.join;

import lombok.AllArgsConstructor;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Created by huitseeker on 3/6/17. */
@AllArgsConstructor
public class ExtractKeysFunction implements Function<List<Writable>, Pair<List<Writable>, List<Writable>>> {
    private int[] columnIndexes;

    @Override
    public Pair<List<Writable>, List<Writable>> apply(List<Writable> writables) {

        List<Writable> keyValues;
        if (columnIndexes.length == 1) {
            keyValues = Collections.singletonList(writables.get(columnIndexes[0]));
        } else {
            keyValues = new ArrayList<>(columnIndexes.length);
            for (int i : columnIndexes) {
                keyValues.add(writables.get(i));
            }
        }

        return Pair.of(keyValues, writables);
    }
}
