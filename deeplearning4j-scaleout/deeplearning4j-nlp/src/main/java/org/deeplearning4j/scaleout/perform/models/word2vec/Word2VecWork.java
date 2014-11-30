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
    private Map<String,INDArray> originalVectors = new HashMap<>();
    private Map<String,INDArray> originalSyn1Vectors = new HashMap<>();
    private Map<String,INDArray> originalNegative = new HashMap<>();
    private Map<String,INDArray> syn1Vectors = new HashMap<>();

    public Word2VecWork(Word2Vec vec,List<VocabWord> sentence) {
        this.sentence = sentence;
        for(VocabWord word : sentence) {
            indexes.put(word.getIndex(),word);
            vectors.put(word.getWord(),new Pair<>(word,vec.getWordVectorMatrix(word.getWord()).dup()));
            originalVectors.put(word.getWord(),vec.getWordVectorMatrix(word.getWord()).dup());
            if(vec.getLookupTable() instanceof InMemoryLookupTable) {
                InMemoryLookupTable l = (InMemoryLookupTable) vec.getLookupTable();
                syn1Vectors.put(word.getWord(),l.getSyn1().slice(word.getIndex()).dup());
                originalSyn1Vectors.put(word.getWord(),l.getSyn1().slice(word.getIndex()).dup());
                originalNegative.put(word.getWord(),l.getSyn1Neg().slice(word.getIndex()).dup());
                if(l.getSyn1Neg() != null)
                    negativeVectors.put(word.getWord(),new Pair<>(word,l.getSyn1Neg().slice(word.getIndex()).dup()));
            }

        }
    }



    public Word2VecResult addDeltas() {
        Map<String,INDArray> syn0Change = new HashMap<>();
        Map<String,INDArray> syn1Change = new HashMap<>();
        Map<String,INDArray> negativeChange = new HashMap<>();


        for(VocabWord word : sentence) {
            syn0Change.put(word.getWord(),vectors.get(word.getWord()).getSecond().subi(originalVectors.get(word.getWord())));
            syn1Change.put(word.getWord(),syn1Vectors.get(word.getWord()).subi(originalSyn1Vectors.get(word.getWord())));
            if(!negativeVectors.isEmpty())
                negativeChange.put(word.getWord(),negativeVectors.get(word.getWord()).getSecond().subi(originalNegative.get(word.getWord())));
        }

        return new Word2VecResult(syn0Change,syn1Change,negativeChange);
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
