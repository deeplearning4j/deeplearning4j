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

package org.deeplearning4j.models.glove;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.scaleout.perform.models.glove.GloveResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGrad;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Glove work
 *
 * @author Adam Gibson
 */
public class GloveWork implements Serializable {

    private Map<String,Pair<VocabWord,INDArray>> vectors = new ConcurrentHashMap<>();

    private List<Pair<VocabWord,VocabWord>> coOccurrences;
    private Map<Integer,VocabWord> indexes = new ConcurrentHashMap<>();
    private Map<String,INDArray> originalVectors = new ConcurrentHashMap<>();
    private Map<String,Double> biases = new ConcurrentHashMap<>();
    private Map<String,AdaGrad> adaGrads = new ConcurrentHashMap<>();
    private Map<String,AdaGrad> biasAdaGrads = new ConcurrentHashMap<>();

    public GloveWork(GloveWeightLookupTable table, List<Pair<VocabWord,VocabWord>> coOccurrences) {
        this.coOccurrences = coOccurrences;
        for(Pair<VocabWord,VocabWord> coOccurrence : coOccurrences) {
            indexes.put(coOccurrence.getFirst().getIndex(),coOccurrence.getFirst());
            indexes.put(coOccurrence.getSecond().getIndex(),coOccurrence.getSecond());
            addWord(coOccurrence.getFirst(),table);
            addWord(coOccurrence.getSecond(),table);


        }

    }

    private void addWord(VocabWord word,GloveWeightLookupTable table) {
        if(word == null)
            throw new IllegalArgumentException("Word must not be null!");

        indexes.put(word.getIndex(),word);
        vectors.put(word.getWord(),new Pair<>(word,table.getSyn0().getRow(word.getIndex()).dup()));
        originalVectors.put(word.getWord(),table.getSyn0().getRow(word.getIndex()).dup());
        biases.put(word.getWord(),table.getBias().getDouble(word.getIndex()));
        adaGrads.put(word.getWord(),table.getWeightAdaGrad().createSubset(word.getIndex()));
        biasAdaGrads.put(word.getWord(),table.getBiasAdaGrad().createSubset(word.getIndex()));

    }



    public AdaGrad getBiasAdaGrad(String word) {
        return biasAdaGrads.get(word);
    }

    public AdaGrad getAdaGrad(String word) {
        return adaGrads.get(word);
    }


    public void updateBias(String word,double bias) {
        biases.put(word,bias);
    }


    public GloveResult addDeltas() {
        Map<String,INDArray> syn0Change = new HashMap<>();
        for(Pair<VocabWord,VocabWord> sentence : coOccurrences) {
            VocabWord w1 = sentence.getFirst();
            VocabWord w2 = sentence.getSecond();
            syn0Change.put(w1.getWord(),vectors.get(w1.getWord()).getSecond().sub(originalVectors.get(w1.getWord())));
            syn0Change.put(w2.getWord(),vectors.get(w2.getWord()).getSecond().sub(originalVectors.get(w2.getWord())));

        }



        return new GloveResult(syn0Change);
    }


    public double getBias(String word) {
        return biases.get(word);
    }


    public List<Pair<VocabWord, VocabWord>> getCoOccurrences() {
        return coOccurrences;
    }

    public void setCoOccurrences(List<Pair<VocabWord, VocabWord>> coOccurrences) {
        this.coOccurrences = coOccurrences;
    }

    public Map<String, Pair<VocabWord, INDArray>> getVectors() {
        return vectors;
    }

    public void setVectors(Map<String, Pair<VocabWord, INDArray>> vectors) {
        this.vectors = vectors;
    }

    public Map<Integer, VocabWord> getIndexes() {
        return indexes;
    }

    public void setIndexes(Map<Integer, VocabWord> indexes) {
        this.indexes = indexes;
    }



    public Map<String, INDArray> getOriginalVectors() {
        return originalVectors;
    }

    public void setOriginalVectors(Map<String, INDArray> originalVectors) {
        this.originalVectors = originalVectors;
    }

    public Map<String, Double> getBiases() {
        return biases;
    }

    public void setBiases(Map<String, Double> biases) {
        this.biases = biases;
    }
}
