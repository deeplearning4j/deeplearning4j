package org.deeplearning4j.scaleout.perform.models.word2vec.iterator;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.scaleout.api.statetracker.NewUpdateListener;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult;
import org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecWork;

import java.io.Serializable;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Word2vec job iterator
 *
 *
 * @author Adam Gibson
 */
public class Word2VecJobIterator implements JobIterator {

    private Iterator<List<VocabWord>> sentenceIterator;
    private Word2Vec vec;

    public Word2VecJobIterator(Iterator<List<VocabWord>> sentenceIterator, Word2Vec vec2,StateTracker stateTracker) {
        this.sentenceIterator = sentenceIterator;
        this.vec = vec2;
        stateTracker.addUpdateListener(new NewUpdateListener() {
            @Override
            public void onUpdate(Serializable update) {
                Collection<Word2VecResult> work = (Collection<Word2VecResult>) update;
                InMemoryLookupTable l = (InMemoryLookupTable) vec.getLookupTable();
                VocabCache cache = vec.getCache();
                for(Word2VecResult work1 : work) {
                   for(String s : work1.getSyn0Change().keySet()) {
                       l.getSyn0().getRow(cache.indexOf(s)).addi(work1.getSyn0Change().get(s));
                       l.getSyn1().getRow(cache.indexOf(s)).addi(work1.getSyn1Change().get(s));
                       if(l.getSyn1Neg() != null)
                           l.getSyn1Neg().getRow(cache.indexOf(s)).addi(work1.getNegativeChange().get(s));


                   }
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
