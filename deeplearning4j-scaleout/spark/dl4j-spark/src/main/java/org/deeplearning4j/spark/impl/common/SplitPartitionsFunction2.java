package org.deeplearning4j.spark.impl.common;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function2;
import scala.Tuple2;

import java.util.*;

/**
 * Equivelent to {@link SplitPartitionsFunction}, but for {@code JavaPairRDD}s
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class SplitPartitionsFunction2<T, U>
                implements Function2<Integer, Iterator<Tuple2<T, U>>, Iterator<Tuple2<T, U>>> {
    private final int splitIndex;
    private final int numSplits;
    private final long baseRngSeed;

    @Override
    public Iterator<Tuple2<T, U>> call(Integer v1, Iterator<Tuple2<T, U>> iter) throws Exception {
        long thisRngSeed = baseRngSeed + v1;

        Random r = new Random(thisRngSeed);
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < numSplits; i++) {
            list.add(i);
        }

        List<Tuple2<T, U>> outputList = new ArrayList<>();
        int i = 0;
        while (iter.hasNext()) {
            if (i % numSplits == 0)
                Collections.shuffle(list, r);

            Tuple2<T, U> next = iter.next();
            if (list.get(i % numSplits) == splitIndex)
                outputList.add(next);
            i++;
        }

        return outputList.iterator();
    }
}
