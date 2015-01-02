package org.deeplearning4j.models.glove.actor;

import java.io.Serializable;

/**
 * Created by agibsonccc on 12/7/14.
 */
public class SentenceWork implements Serializable {
    private int id;
    private String sentence;

    public SentenceWork(int id, String sentence) {
        this.id = id;
        this.sentence = sentence;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getSentence() {
        return sentence;
    }

    public void setSentence(String sentence) {
        this.sentence = sentence;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof SentenceWork)) return false;

        SentenceWork that = (SentenceWork) o;

        if (id != that.id) return false;
        if (sentence != null ? !sentence.equals(that.sentence) : that.sentence != null) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = id;
        result = 31 * result + (sentence != null ? sentence.hashCode() : 0);
        return result;
    }
}
