package org.datavec.spark.transform.join;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.PairFunction;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Created by huitseeker on 3/6/17. */
@AllArgsConstructor
public class ExtractKeysFunction implements PairFunction<List<Writable>, List<Writable>, List<Writable>> {
    private int[] columnIndexes;

    @Override
    public Tuple2<List<Writable>, List<Writable>> call(List<Writable> writables) throws Exception {

        List<Writable> keyValues;
        if (columnIndexes.length == 1) {
            keyValues = Collections.singletonList(writables.get(columnIndexes[0]));
        } else {
            keyValues = new ArrayList<>(columnIndexes.length);
            for (int i : columnIndexes) {
                keyValues.add(writables.get(i));
            }
        }

        return new Tuple2<>(keyValues, writables);
    }
}
