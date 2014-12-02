package org.deeplearning4j.models.glove;

import org.deeplearning4j.models.word2vec.VocabWord;

import java.io.Serializable;

/**
 * Word co occurrence
 *
 * @author Adam Gibson
 */
public class Cooccurrence implements Serializable {
    private VocabWord w1;
    private VocabWord w2;
    private double score;

    public Cooccurrence(VocabWord w1, VocabWord w2, double score) {
        this.w1 = w1;
        this.w2 = w2;
        this.score = score;
    }

    public VocabWord getW1() {
        return w1;
    }

    public void setW1(VocabWord w1) {
        this.w1 = w1;
    }

    public VocabWord getW2() {
        return w2;
    }

    public void setW2(VocabWord w2) {
        this.w2 = w2;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }
}
