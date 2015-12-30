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

package org.deeplearning4j.scaleout.perform.models.word2vec;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Word2vec work
 *
 * @author Adam Gibson
 */
public class Word2VecWork implements Serializable {

    private Map<String,Pair<VocabWord,INDArray>> vectors = new ConcurrentHashMap<>();
    private Map<String,Pair<VocabWord,INDArray>> negativeVectors = new ConcurrentHashMap<>();

    private List<List<VocabWord>> sentences;
    private Map<Integer,VocabWord> indexes = new ConcurrentHashMap<>();
    private Map<String,INDArray> originalVectors = new ConcurrentHashMap<>();
    private Map<String,INDArray> originalSyn1Vectors = new ConcurrentHashMap<>();
    private Map<String,INDArray> originalNegative = new ConcurrentHashMap<>();
    private Map<String,INDArray> syn1Vectors = new ConcurrentHashMap<>();

    public Word2VecWork(InMemoryLookupTable<VocabWord> table,InMemoryLookupCache cache,List<List<VocabWord>> sentences) {
        this.sentences = sentences;
        for(List<VocabWord> sentence : sentences)
            for(VocabWord word : sentence) {
                addWord(word,table);
                if(word.getPoints() != null) {
                    for(int i = 0; i < word.getCodeLength(); i++) {
                        VocabWord pointWord = cache.wordFor(cache.wordAtIndex(word.getPoints().get(i)));
                        addWord(pointWord,table);
                    }
                }
            }
    }

    private void addWord(VocabWord word,InMemoryLookupTable<VocabWord> table) {
        if(word == null)
            throw new IllegalArgumentException("Word must not be null!");

        indexes.put(word.getIndex(),  word);
        vectors.put(word.getLabel(),new Pair<>(word,table.getSyn0().getRow(word.getIndex()).dup()));
        originalVectors.put(word.getLabel(),table.getSyn0().getRow(word.getIndex()).dup());
        syn1Vectors.put(word.getLabel(), table.getSyn1().slice(word.getIndex()).dup());
        originalSyn1Vectors.put(word.getLabel(), table.getSyn1().slice(word.getIndex()).dup());
        if(table.getSyn1Neg() != null) {
            originalNegative.put(word.getLabel(), table.getSyn1Neg().slice(word.getIndex()).dup());
            negativeVectors.put(word.getLabel(), new Pair<>(word, table.getSyn1Neg().slice(word.getIndex()).dup()));
        }

    }


    public Word2VecResult addDeltas() {
        Map<String,INDArray> syn0Change = new HashMap<>();
        Map<String,INDArray> syn1Change = new HashMap<>();
        Map<String,INDArray> negativeChange = new HashMap<>();
        for(List<VocabWord> sentence : sentences)
            for(VocabWord word : sentence) {
                syn0Change.put(word.getLabel(),vectors.get(word.getLabel()).getSecond().sub(originalVectors.get(word.getLabel())));
                syn1Change.put(word.getLabel(),syn1Vectors.get(word.getLabel()).sub(originalSyn1Vectors.get(word.getLabel())));
                if(!negativeVectors.isEmpty())
                    negativeChange.put(word.getLabel(),negativeVectors.get(word.getLabel()).getSecond().subi(originalNegative.get(word.getLabel())));
            }

        return new Word2VecResult(syn0Change,syn1Change,negativeChange);
    }

    public List<List<VocabWord>> getSentences() {
        return sentences;
    }

    public void setSentences(List<List<VocabWord>> sentences) {
        this.sentences = sentences;
    }

    public Map<String, Pair<VocabWord, INDArray>> getNegativeVectors() {
        return negativeVectors;
    }

    public void setNegativeVectors(Map<String, Pair<VocabWord, INDArray>> negativeVectors) {
        this.negativeVectors = negativeVectors;
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

    public Map<String, INDArray> getOriginalSyn1Vectors() {
        return originalSyn1Vectors;
    }

    public void setOriginalSyn1Vectors(Map<String, INDArray> originalSyn1Vectors) {
        this.originalSyn1Vectors = originalSyn1Vectors;
    }

    public Map<String, INDArray> getOriginalNegative() {
        return originalNegative;
    }

    public void setOriginalNegative(Map<String, INDArray> originalNegative) {
        this.originalNegative = originalNegative;
    }

    public Map<String, INDArray> getSyn1Vectors() {
        return syn1Vectors;
    }

    public void setSyn1Vectors(Map<String, INDArray> syn1Vectors) {
        this.syn1Vectors = syn1Vectors;
    }
}
