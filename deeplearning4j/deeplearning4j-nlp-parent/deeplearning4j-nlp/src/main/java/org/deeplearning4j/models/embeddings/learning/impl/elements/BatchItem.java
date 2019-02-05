package org.deeplearning4j.models.embeddings.learning.impl.elements;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.linalg.api.ops.aggregates.Batch;

import java.util.concurrent.atomic.AtomicLong;

public class BatchItem<T extends SequenceElement>  {
    private T word;
    private T lastWord;
    private int[] windowWords; // CBOW only
    private long randomValue;
    private double alpha;
    private int windowWordsLength;

    public BatchItem(T word, T lastWord, long randomValue, double alpha) {
        this.word = word;
        this.lastWord = lastWord;
        this.randomValue = randomValue;
        this.alpha = alpha;
    }

    public BatchItem(T word, int[] windowWords, long randomValue, double alpha) {
        this.word = word;
        this.lastWord = lastWord;
        this.randomValue = randomValue;
        this.alpha = alpha;
        this.windowWords = windowWords.clone();
    }

    public T getWord() {
        return word;
    }

    public void setWord(T word) {
        this.word = word;
    }

    public T getLastWord() {
        return lastWord;
    }

    public void setLastWord(T lastWord) {
        this.lastWord = lastWord;
    }

    public long getRandomValue() {
        return randomValue;
    }

    public void setRandomValue(long randomValue) {
        this.randomValue = randomValue;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public int[] getWindowWords() {
        return windowWords;
    }

    public void setWindowWords(int[] windowWords) {
        this.windowWords = windowWords;
    }
}
