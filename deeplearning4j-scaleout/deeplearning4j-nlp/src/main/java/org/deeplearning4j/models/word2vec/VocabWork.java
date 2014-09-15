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



    public VocabWork(AtomicInteger count,String work) {
        this.count = count;
        this.work = work;
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

    public void countDown() {
        count.decrementAndGet();
    }

}
