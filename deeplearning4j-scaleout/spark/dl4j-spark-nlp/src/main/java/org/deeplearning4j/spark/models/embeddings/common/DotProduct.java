package org.deeplearning4j.spark.models.embeddings.common;

import com.hazelcast.core.IFunction;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Dot product
 * @author Adam Gibson
 */
public class DotProduct extends BaseWord2VecFunction implements IFunction<InMemoryLookupTable,Double> {
    public DotProduct(int fromIndex, int toIndex, String from, String to) {
        super(fromIndex, toIndex, from, to);
    }

    @Override
    public Double apply(InMemoryLookupTable input) {
        return Nd4j.getBlasWrapper().dot(getFrom(input),getTo(input));
    }
}
