package org.deeplearning4j.scaleout.perform.models.word2vec;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Word2vec work
 *
 * @author Adam Gibson
 */
public class Word2VecWork implements Serializable {

    private Map<String,Pair<VocabWord,INDArray>> vectors = new HashMap<>();
    private Map<String,Pair<VocabWord,INDArray>> negativeVectors = new HashMap<>();

    private List<VocabWord> sentence;
    private Map<Integer,VocabWord> indexes = new HashMap<>();


    public Word2VecWork(Word2Vec vec,List<VocabWord> sentence) {
        this.sentence = sentence;
        for(VocabWord word : sentence) {
            indexes.put(word.getIndex(),word);
            vectors.put(word.getWord(),new Pair<>(word,vec.getWordVectorMatrix(word.getWord())));
            if(vec.getLookupTable() instanceof InMemoryLookupTable) {
                InMemoryLookupTable l = (InMemoryLookupTable) vec.getLookupTable();
                negativeVectors.put(word.getWord(),new Pair<>(word,l.getSyn1Neg().slice(word.getIndex())));
            }

        }
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

    public List<VocabWord> getSentence() {
        return sentence;
    }

    public void setSentence(List<VocabWord> sentence) {
        this.sentence = sentence;
    }

    public Map<Integer, VocabWord> getIndexes() {
        return indexes;
    }

    public void setIndexes(Map<Integer, VocabWord> indexes) {
        this.indexes = indexes;
    }
}
