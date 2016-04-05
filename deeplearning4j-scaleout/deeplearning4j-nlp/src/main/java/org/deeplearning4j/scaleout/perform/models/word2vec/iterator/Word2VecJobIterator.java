/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.scaleout.perform.models.word2vec.iterator;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.scaleout.api.statetracker.NewUpdateListener;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecResult;
import org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecWork;
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
public class Word2VecJobIterator implements JobIterator {

    private Iterator<List<List<VocabWord>>> sentenceIterator;
    private WeightLookupTable table;
    private VocabCache cache;
    private int batchSize = 100;



    public Word2VecJobIterator(Iterator<List<List<VocabWord>>> sentenceIterator,WeightLookupTable table,VocabCache cache,StateTracker stateTracker,int batchSize) {
        this.sentenceIterator = sentenceIterator;
        this.table = table;
        this.cache = cache;
        addListener(stateTracker);
        this.batchSize = batchSize;

    }


    public Word2VecJobIterator(TextVectorizer textVectorizer,WeightLookupTable table,VocabCache cache,StateTracker stateTracker,int batchSize) {
        //this.sentenceIterator = textVectorizer.index().batchIter(batchSize);
        this.cache = cache;
        this.table = table;
        addListener(stateTracker);
        this.batchSize = batchSize;

    }

    public Word2VecJobIterator(Iterator<List<List<VocabWord>>> sentenceIterator,WeightLookupTable table,VocabCache cache,StateTracker stateTracker) {
        this.sentenceIterator = sentenceIterator;
        this.table = table;
        this.cache = cache;
        addListener(stateTracker);

    }


    public Word2VecJobIterator(TextVectorizer textVectorizer,WeightLookupTable table,VocabCache cache,StateTracker stateTracker) {
    //    this.sentenceIterator = textVectorizer.index().batchIter(batchSize);
        this.cache = cache;
        this.table = table;
        addListener(stateTracker);

    }
    public Word2VecJobIterator(InvertedIndex invertedIndex, WeightLookupTable table,VocabCache cache,StateTracker stateTracker,int batchSize) {
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
                Collection<Word2VecResult> work = (Collection<Word2VecResult>) j.getResult();
                if(work == null || work.isEmpty())
                    return;

                InMemoryLookupTable l = (InMemoryLookupTable) table;

                for(Word2VecResult work1 : work) {
                    for(String s : work1.getSyn0Change().keySet()) {
                        l.getSyn0().putRow(cache.indexOf(s),work1.getSyn0Change().get(s));
                        l.getSyn1().putRow(cache.indexOf(s),work1.getSyn1Change().get(s));
                        if(l.getSyn1Neg() != null)
                            l.getSyn1Neg().putRow(cache.indexOf(s),work1.getNegativeChange().get(s));


                    }
                }



            }
        });
    }


    private Word2VecWork create(List<List<VocabWord>> sentence) {
        if(cache == null)
            throw new IllegalStateException("Unable to create work; no vocab found");
        if(table == null)
            throw new IllegalStateException("Unable to create work; no table found");
        if(sentence == null)
            throw new IllegalArgumentException("Unable to create work from null sentence");
        Word2VecWork work = new Word2VecWork((InMemoryLookupTable) table,(InMemoryLookupCache) cache,sentence);
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
