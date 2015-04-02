package org.deeplearning4j.spark.models.embeddings.common;

import com.hazelcast.core.IFunction;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Run a saxpy function on a hazelcast object
 * @author Adam Gibson
 */
public class SaxpyMap extends  BaseWord2VecFunction implements IFunction<InMemoryLookupTable,InMemoryLookupTable> {
    private Number alpha;

    public SaxpyMap(Number alpha, int fromIndex, int toIndex, String from, String to) {
        super(fromIndex, toIndex, from, to);
        this.alpha = alpha;
    }

    @Override
    public InMemoryLookupTable apply(InMemoryLookupTable inMemoryLookupTable) {
        if(inMemoryLookupTable.getSyn0().data().dataType() == DataBuffer.DOUBLE)
            Nd4j.getBlasWrapper().axpy(alpha.doubleValue(),getFrom(inMemoryLookupTable),getTo(inMemoryLookupTable));
        else
            Nd4j.getBlasWrapper().axpy(alpha.floatValue(),getFrom(inMemoryLookupTable),getTo(inMemoryLookupTable));
        return inMemoryLookupTable;
    }


}
