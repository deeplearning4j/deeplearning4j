package org.deeplearning4j.word2vec.wordstore;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.word2vec.VocabWord;

import java.util.Collection;

/**
 * Created by agibsonccc on 8/31/14.
 */
public interface VocabCache  {


    void incrementWordCount(String word);


    void incrementWordCount(String word,int increment);

    int wordFrequency(String word);

    boolean containsWord(String word);

    String wordAtIndex(int index);

    int indexOf(String word);


    void putCode(int codeIndex,INDArray code);

    INDArray loadCodes(int[] codes);

    Collection<VocabWord> vocabWords();


    int totalWordOccurrences();

    void putVector(String word,INDArray vector);

    INDArray vector(String word);

    VocabWord wordFor(String word);


    void addWordToIndex(int index,String word);

    void putVocabWord(String word,VocabWord vocabWord);

    int numWords();

}
