package org.deeplearning4j.models.embeddings.learning.impl.elements;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

public class BatchItem<T extends SequenceElement>  {
    private T word;
    private T lastWord;

    private int[] windowWords; // CBOW only
    private boolean[] wordStatuses;

    private long randomValue;
    private double alpha;
    private int windowWordsLength;

    private int numLabel;

    public BatchItem(T word, T lastWord, long randomValue, double alpha) {
        this.word = word;
        this.lastWord = lastWord;
        this.randomValue = randomValue;
        this.alpha = alpha;
    }

    public BatchItem(T word, int[] windowWords, boolean[] wordStatuses, long randomValue, double alpha, int numLabel) {
        this.word = word;
        this.lastWord = lastWord;
        this.randomValue = randomValue;
        this.alpha = alpha;
        this.numLabel = numLabel;
        this.windowWords = windowWords.clone();
        this.wordStatuses = wordStatuses.clone();
    }

    public BatchItem(T word, int[] windowWords, boolean[] wordStatuses, long randomValue, double alpha) {
        this.word = word;
        this.lastWord = lastWord;
        this.randomValue = randomValue;
        this.alpha = alpha;
        this.windowWords = windowWords.clone();
        this.wordStatuses = wordStatuses.clone();
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

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public int[] getWindowWords() {
        return windowWords;
    }

    public boolean[] getWordStatuses() {
        return wordStatuses;
    }

    public int getNumLabel() {
        return numLabel;
    }
}
