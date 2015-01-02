package org.deeplearning4j.scaleout.perform.text;

import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

/**
 * Job iterator for sentences
 * @author Adam Gibson
 */
public class SentenceJobIterator implements JobIterator {
    private SentenceIterator iterator;

    public SentenceJobIterator(SentenceIterator iterator) {
        this.iterator = iterator;
    }

    @Override
    public Job next(String workerId) {
        return new Job(iterator.nextSentence(),workerId);
    }

    @Override
    public Job next() {
        return new Job(iterator.nextSentence(),"");
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public void reset() {
       iterator.reset();
    }
}
