package org.deeplearning4j.spark.models.embeddings.common;

import com.hazelcast.core.IFunction;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 3/30/15.
 */
public class GetVector extends BaseWord2VecFunction implements IFunction<InMemoryLookupTable,INDArray> {

    public GetVector(int index,String from) {
        this(index,0,from,null);
    }

    public GetVector(int fromIndex, int toIndex, String from, String to) {
        super(fromIndex, toIndex, from, to);
    }

    @Override
    public INDArray apply(InMemoryLookupTable input) {
        return getFrom(input);
    }
}
