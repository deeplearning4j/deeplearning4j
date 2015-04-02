package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.commons.collections.map.HashedMap;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.berkeley.Pair;
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

    public void apply(InMemoryLookupTable table) {
        for(Integer i : syn0Vectors.keySet())
            Nd4j.getBlasWrapper().axpy(1,syn0Vectors.get(i),table.getSyn0().slice(i));
        for(Integer i : syn1Vectors.keySet())
            Nd4j.getBlasWrapper().axpy(1,syn1Vectors.get(i),table.getSyn1().slice(i));
        for(Integer i : negSyn1Vectors.keySet())
            Nd4j.getBlasWrapper().axpy(1, negSyn1Vectors.get(i), table.getSyn1Neg().slice(i));
    }

    public Map<Integer, INDArray> getSyn0Vectors() {
        return syn0Vectors;
    }

    public void setSyn0Vectors(Map<Integer, INDArray> syn0Vectors) {
        this.syn0Vectors = syn0Vectors;
    }

    public Map<Integer, INDArray> getSyn1Vectors() {
        return syn1Vectors;
    }

    public void setSyn1Vectors(Map<Integer, INDArray> syn1Vectors) {
        this.syn1Vectors = syn1Vectors;
    }

    public Map<Integer, INDArray> getNegSyn1Vectors() {
        return negSyn1Vectors;
    }

    public void setNegSyn1Vectors(Map<Integer, INDArray> negSyn1Vectors) {
        this.negSyn1Vectors = negSyn1Vectors;
    }
}
