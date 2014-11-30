package org.deeplearning4j.scaleout.perform.models.word2vec.iterator;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.scaleout.api.statetracker.NewUpdateListener;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecWork;

import java.io.Serializable;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Created by agibsonccc on 11/29/14.
 */
public class Word2VecJobIterator implements JobIterator {

    private Iterator<List<VocabWord>> sentenceIterator;
    private Word2Vec vec;
    private StateTracker stateTracker;

    public Word2VecJobIterator(Iterator<List<VocabWord>> sentenceIterator, Word2Vec vec,StateTracker stateTracker) {
        this.sentenceIterator = sentenceIterator;
        this.vec = vec;
        this.stateTracker = stateTracker;
        stateTracker.addUpdateListener(new NewUpdateListener() {
            @Override
            public void onUpdate(Serializable update) {
                Collection<Word2VecWork> work = (Collection<Word2VecWork>) update;
                for(Word2VecWork work1 : work) {

                }

            }
        });
    }



    private Word2VecWork create(List<VocabWord> sentence) {
        Word2VecWork work = new Word2VecWork(vec,sentence);
        return work;
    }

    @Override
    public Job next(String workerId) {
        List<VocabWord> next = sentenceIterator.next();
        return new Job(create(next),workerId);
    }

    @Override
    public Job next() {
        List<VocabWord> next = sentenceIterator.next();
        return new Job(create(next),"");
    }

    @Override
    public boolean hasNext() {
        return sentenceIterator.hasNext();
    }

    @Override
    public void reset() {

    }
}
