/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.models.word2vec;


import org.canova.api.conf.Configuration;
import org.deeplearning4j.scaleout.aggregator.JobAggregator;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult;
import org.deeplearning4j.util.MultiDimensionalMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.*;

/**
 * Handles creating a total word2vec model
 * @author Adam Gibson
 */
public class Word2VecJobAggregator implements JobAggregator {
    private List<org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult> work = new ArrayList<>();


    @Override
    public void accumulate(Job job) {
        if(job.getResult() instanceof org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult) {
            org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult work = (org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult) job.getResult();
            this.work.add(work);
        }
        else if(job.getResult() instanceof Collection) {
            Collection<org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult> coll = (Collection<org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult>) job.getResult();
            work.addAll(coll);
        }

    }

    @Override
    public Job aggregate() {
        Job ret =  new Job("","");
        org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult aggregateResult = new org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult();
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




        ret.setResult((Serializable) Arrays.asList(aggregateResult));
        return ret;
    }


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
    public void init(Configuration conf) {

    }
}
