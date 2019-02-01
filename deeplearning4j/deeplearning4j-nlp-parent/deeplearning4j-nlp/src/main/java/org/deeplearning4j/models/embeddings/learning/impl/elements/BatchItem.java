package org.deeplearning4j.models.embeddings.learning.impl.elements;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.concurrent.atomic.AtomicLong;

class BatchItem<T extends SequenceElement>  {
    private T word;
    private T lastWord;
    private AtomicLong randomValue;
    private double alpha;

    public BatchItem(T word, T lastWord, AtomicLong randomValue, double alpha) {
        this.word = word;
        this.lastWord = lastWord;
        this.randomValue = randomValue;
        this.alpha = alpha;
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

    public AtomicLong getRandomValue() {
        return randomValue;
    }

    public void setRandomValue(AtomicLong randomValue) {
        this.randomValue = randomValue;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }
}
