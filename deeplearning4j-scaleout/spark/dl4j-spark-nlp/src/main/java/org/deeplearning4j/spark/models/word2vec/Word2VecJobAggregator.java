package org.deeplearning4j.spark.models.word2vec;


import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult;
import org.deeplearning4j.util.MultiDimensionalMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Handles creating a total word2vec model
 * @author Adam Gibson
 */
public class Word2VecJobAggregator implements Function<Collection<Word2VecResult>, Word2VecResult> {




    private INDArray average(List<INDArray> list) {
        if(list == null || list.isEmpty())
            throw new IllegalArgumentException("Can't average empty or null list");
        if(list.get(0) == null)
            return null;
        INDArray ret = Nd4j.create(list.get(0).shape());
        for(INDArray arr : list)
            ret.addi(arr);
        if(list.size() > 1)
            return ret.divi((double) list.size());

        return ret;
    }


    private List<INDArray> getOrPutIfNotExists( MultiDimensionalMap<String,String,List<INDArray>> workResults,String key,String otherKey) {
        List<INDArray> syn0List = workResults.get(key,otherKey);
        if(syn0List == null) {
            syn0List = new ArrayList<>();
            workResults.put(key,otherKey,syn0List);
        }
        return syn0List;
    }


    @Override
    public Word2VecResult call(Collection<Word2VecResult> work) throws Exception {
        Word2VecResult aggregateResult = new Word2VecResult();
        MultiDimensionalMap<String,String,List<INDArray>> workResults = MultiDimensionalMap.newHashBackedMap();
        Set<String> vocab = new HashSet<>();
        for(Word2VecResult r : work) {
            for(String syn0Key : r.getSyn0Change().keySet()) {
                List<INDArray> syn0List = getOrPutIfNotExists(workResults,syn0Key,"syn0");
                List<INDArray> syn1List = getOrPutIfNotExists(workResults,syn0Key,"syn1");
                List<INDArray> negList = getOrPutIfNotExists(workResults,syn0Key,"negative");
                syn0List.add(r.getSyn0Change().get(syn0Key));
                syn1List.add(r.getSyn1Change().get(syn0Key));
                if(r.getNegativeChange() != null)
                    negList.add(r.getNegativeChange().get(syn0Key));
                vocab.add(syn0Key);

            }
        }

        for(String key : vocab) {
            aggregateResult.getSyn0Change().put(key,average(workResults.get(key,"syn0")));
            aggregateResult.getSyn1Change().put(key,average(workResults.get(key,"syn1")));
            if(workResults.get(key,"negative") != null && workResults.get(key,"negative") != null && !workResults.get(key,"negative").isEmpty() && workResults.get(key,"negative").get(0) != null)
                aggregateResult.getNegativeChange().put(key,average(workResults.get(key,"negative")));
        }

        return aggregateResult;


    }
}
