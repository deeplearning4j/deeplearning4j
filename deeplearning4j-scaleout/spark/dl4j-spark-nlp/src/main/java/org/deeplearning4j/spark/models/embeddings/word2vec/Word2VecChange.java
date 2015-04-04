package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.commons.collections.map.HashedMap;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * @author Adam Gibson
 */
public class Word2VecChange implements Serializable {
    private Map<Integer,INDArray> syn0Vectors = new HashedMap();
    private Map<Integer,INDArray> syn1Vectors = new HashMap<>();
    private Map<Integer,INDArray> negSyn1Vectors = new HashMap<>();

    public Word2VecChange(List<Triple<Integer,Integer,Integer>> counterMap,Word2VecParam param) {
        Iterator<Triple<Integer,Integer,Integer>> iter = counterMap.iterator();
        while(iter.hasNext()) {
            Triple<Integer,Integer,Integer> next = iter.next();
            if(!syn0Vectors.containsKey(next.getFirst()))
                syn0Vectors.put(next.getFirst(),param.getWeights().getSyn0().slice(next.getFirst()));
            if(!syn1Vectors.containsKey(next.getSecond()))
                syn1Vectors.put(next.getFirst(),param.getWeights().getSyn1().slice(next.getSecond()));
            if(param.getNegative() > 0) {
                if(!negSyn1Vectors.containsKey(next.getThird()))
                    negSyn1Vectors.put(next.getFirst(), param.getWeights().getSyn1Neg().slice(next.getThird()));

            }

        }
    }

    /**
     * Take the changes and apply them
     * to the given table
     * @param table the memory lookup table
     *              to apply the changes to
     */
    public void apply(InMemoryLookupTable table) {
        for(Integer i : syn0Vectors.keySet())
            Nd4j.getBlasWrapper().axpy(1,syn0Vectors.get(i),table.getSyn0().slice(i));
        for(Integer i : syn1Vectors.keySet())
            Nd4j.getBlasWrapper().axpy(1,syn1Vectors.get(i),table.getSyn1().slice(i));
        for(Integer i : negSyn1Vectors.keySet())
            Nd4j.getBlasWrapper().axpy(1, negSyn1Vectors.get(i), table.getSyn1Neg().slice(i));
    }
}
