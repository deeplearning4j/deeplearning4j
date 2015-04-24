package org.deeplearning4j.ui.nearestneighbors;

import java.io.Serializable;

/**
 * @author Adam Gibson
 */
public class NearestNeighborsQuery implements Serializable {
    private String word;
    private int numWords;

    public NearestNeighborsQuery(String word, int numWords) {
        this.word = word;
        this.numWords = numWords;
    }

    public NearestNeighborsQuery() {
    }

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }

    public int getNumWords() {
        return numWords;
    }

    public void setNumWords(int numWords) {
        this.numWords = numWords;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NearestNeighborsQuery that = (NearestNeighborsQuery) o;

        if (numWords != that.numWords) return false;
        return !(word != null ? !word.equals(that.word) : that.word != null);

    }

    @Override
    public int hashCode() {
        int result = word != null ? word.hashCode() : 0;
        result = 31 * result + numWords;
        return result;
    }
}
