package org.deeplearning4j.scaleout.perform.models.glove;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.scaleout.api.statetracker.NewUpdateListener;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.text.invertedindex.InvertedIndex;

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
public class GloveJobIterator implements JobIterator {

    private Iterator<List<List<VocabWord>>> sentenceIterator;
    private GloveWeightLookupTable table;
    private VocabCache cache;
    private int batchSize = 100;



    public GloveJobIterator(Iterator<List<List<VocabWord>>> sentenceIterator, GloveWeightLookupTable table, VocabCache cache, StateTracker stateTracker, int batchSize) {
        this.sentenceIterator = sentenceIterator;
        this.table = table;
        this.cache = cache;
        addListener(stateTracker);
        this.batchSize = batchSize;

    }


    public GloveJobIterator(TextVectorizer textVectorizer, GloveWeightLookupTable table, VocabCache cache, StateTracker stateTracker, int batchSize) {
        this.sentenceIterator = textVectorizer.index().batchIter(batchSize);
        this.cache = cache;
        this.table = table;
        addListener(stateTracker);
        this.batchSize = batchSize;

    }

    public GloveJobIterator(Iterator<List<List<VocabWord>>> sentenceIterator, GloveWeightLookupTable table, VocabCache cache, StateTracker stateTracker) {
        this.sentenceIterator = sentenceIterator;
        this.table = table;
        this.cache = cache;
        addListener(stateTracker);

    }


    public GloveJobIterator(TextVectorizer textVectorizer, GloveWeightLookupTable table, VocabCache cache, StateTracker stateTracker) {
        this.sentenceIterator = textVectorizer.index().batchIter(batchSize);
        this.cache = cache;
        this.table = table;
        addListener(stateTracker);

    }
    public GloveJobIterator(InvertedIndex invertedIndex, GloveWeightLookupTable table, VocabCache cache, StateTracker stateTracker, int batchSize) {
        this.sentenceIterator = invertedIndex.batchIter(batchSize);
        this.cache = cache;
        this.table = table;
        this.batchSize = batchSize;
        addListener(stateTracker);

    }


    private void addListener(StateTracker stateTracker) {
        stateTracker.addUpdateListener(new NewUpdateListener() {
            @Override
            public void onUpdate(Serializable update) {
                Job j = (Job) update;
                Collection<GloveResult> work = (Collection<GloveResult>) j.getResult();
                if(work == null || work.isEmpty())
                    return;

                GloveWeightLookupTable l = table;

                for(GloveResult work1 : work) {
                    for(String s : work1.getSyn0Change().keySet()) {
                        l.getSyn0().putRow(cache.indexOf(s),work1.getSyn0Change().get(s));



                    }
                }



            }
        });
    }


    private GloveWork create(List<List<VocabWord>> sentence) {
        if(cache == null)
            throw new IllegalStateException("Unable to create work; no vocab found");
        if(table == null)
            throw new IllegalStateException("Unable to create work; no table found");
        if(sentence == null)
            throw new IllegalArgumentException("Unable to create work from null sentence");
        GloveWork work = new GloveWork(table,(InMemoryLookupCache) cache,sentence);
        return work;
    }

    @Override
    public Job next(String workerId) {

        List<List<VocabWord>> next = sentenceIterator.next();
        return new Job(create(next),workerId);
    }

    @Override
    public Job next() {
        List<List<VocabWord>> next = sentenceIterator.next();
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
