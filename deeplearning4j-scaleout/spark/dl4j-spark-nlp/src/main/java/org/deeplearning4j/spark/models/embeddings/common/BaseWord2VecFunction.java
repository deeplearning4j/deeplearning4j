package org.deeplearning4j.spark.models.embeddings.common;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Base word2vec function for hazelcast.
 * Stores how the lookup table should be accessed
 *
 * @author Adam Gibson
 */
public class BaseWord2VecFunction implements Serializable {
    protected int fromIndex = 0,toIndex = 0;
    protected String from,to;
    public final static String SYN1 = "syn1";
    public final static String SYN0 = "syn0";
    public final static String SYN1_NEGATIVE = "syn1negative";

    public BaseWord2VecFunction(int fromIndex, int toIndex, String from, String to) {
        this.fromIndex = fromIndex;
        this.toIndex = toIndex;
        this.from = from;
        this.to = to;
    }


    protected INDArray getFrom(InMemoryLookupTable inMemoryLookupTable) {
        switch(from) {
            case SYN0: return inMemoryLookupTable.getSyn0().slice(fromIndex).ravel();
            case SYN1 : return inMemoryLookupTable.getSyn1().slice(fromIndex).ravel();
            case SYN1_NEGATIVE: return inMemoryLookupTable.getSyn1Neg().slice(fromIndex).ravel();
            default: throw new IllegalStateException("Illegal from type " + from);
        }
    }

    protected INDArray getTo(InMemoryLookupTable inMemoryLookupTable) {
        switch(to) {
            case SYN0: return inMemoryLookupTable.getSyn0().slice(toIndex).ravel();
            case SYN1 : return inMemoryLookupTable.getSyn1().slice(toIndex).ravel();
            case SYN1_NEGATIVE: return inMemoryLookupTable.getSyn1Neg().slice(toIndex).ravel();
            default: throw new IllegalStateException("Illegal to type " + to);
        }
    }
}
