package org.deeplearning4j.models.word2vec;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Vocab work
 *
 *
 * @author Adam Gibson
 */
public class VocabWork implements Serializable {

    private AtomicInteger count = new AtomicInteger(0);
    private String work;
    private boolean stem = false;



    public VocabWork(AtomicInteger count,String work,boolean stem) {
        this.count = count;
        this.work = work;
        this.stem = stem;
    }

    public AtomicInteger getCount() {
        return count;
    }

    public void setCount(AtomicInteger count) {
        this.count = count;
    }

    public String getWork() {
        return work;
    }

    public void setWork(String work) {
        this.work = work;
    }

    public void increment() {
        count.incrementAndGet();
    }

    public boolean isStem() {
        return stem;
    }

    public void setStem(boolean stem) {
        this.stem = stem;
    }
}
