package org.deeplearning4j.models.word2vec.actor;

import org.deeplearning4j.models.word2vec.VocabWord;

import java.io.Serializable;
import java.util.List;

/**
 * Created by agibsonccc on 9/16/14.
 */
public class SkipGramMessage implements Serializable {

    private int i;
    private int b;
    private List<VocabWord> sentence;

    public SkipGramMessage(int i, int b, List<VocabWord> sentence) {
        this.i = i;
        this.b = b;
        this.sentence = sentence;
    }

    public int getB() {
        return b;
    }

    public void setB(int b) {
        this.b = b;
    }

    public List<VocabWord> getSentence() {
        return sentence;
    }

    public void setSentence(List<VocabWord> sentence) {
        this.sentence = sentence;
    }

    public int getI() {

        return i;
    }

    public void setI(int i) {
        this.i = i;
    }
}
