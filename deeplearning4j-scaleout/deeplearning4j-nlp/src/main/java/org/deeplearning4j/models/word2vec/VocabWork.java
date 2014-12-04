package org.deeplearning4j.models.word2vec;

import java.io.Serializable;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Vocab work meant for use with the vocab actor
 *
 *
 * @author Adam Gibson
 */
public class VocabWork implements Serializable {

    private AtomicInteger count = new AtomicInteger(0);
    private String work;
    private boolean stem = false;
    private List<String> label;


    public VocabWork(AtomicInteger count,String work,boolean stem) {
        this(count,work,stem,null);
    }

    public VocabWork(AtomicInteger count,String work,boolean stem,List<String> label) {
        this.count = count;
        this.work = work;
        this.stem = stem;
        this.label = label;
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

    public List<String> getLabel() {
        return label;
    }

    public void setLabel(List<String> label) {
        this.label = label;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof VocabWork)) return false;

        VocabWork vocabWork = (VocabWork) o;

        if (stem != vocabWork.stem) return false;
        if (count != null ? !count.equals(vocabWork.count) : vocabWork.count != null) return false;
        if (label != null ? !label.equals(vocabWork.label) : vocabWork.label != null) return false;
        if (work != null ? !work.equals(vocabWork.work) : vocabWork.work != null) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = count != null ? count.hashCode() : 0;
        result = 31 * result + (work != null ? work.hashCode() : 0);
        result = 31 * result + (stem ? 1 : 0);
        result = 31 * result + (label != null ? label.hashCode() : 0);
        return result;
    }
}
