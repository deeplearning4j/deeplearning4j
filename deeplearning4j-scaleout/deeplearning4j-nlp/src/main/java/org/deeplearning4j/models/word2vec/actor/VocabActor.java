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

package org.deeplearning4j.models.word2vec.actor;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicLong;

import akka.dispatch.Futures;
import akka.dispatch.OnFailure;
import akka.dispatch.OnSuccess;
import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.StreamWork;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.VocabWork;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.movingwindow.Util;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.UntypedActor;
import org.tartarus.snowball.ext.PorterStemmer;
import scala.concurrent.Future;

/**
 * Individual actor for updating the vocab cache
 *
 * @author Adam Gibson
 */
public class VocabActor extends UntypedActor {

    private transient TokenizerFactory tokenizer;
    private List<String> stopWords;
    private AtomicLong lastUpdate;
    private VocabCache cache;
    private int minWordFrequency;
    private AtomicLong numWordsEncountered;
    private InvertedIndex index;
    private static final Logger log = LoggerFactory.getLogger(VocabActor.class);
    private PorterStemmer stemmer = new PorterStemmer();

    public VocabActor(
            TokenizerFactory tokenizer,
            VocabCache cache,
            List<String> stopWords,
            AtomicLong lastUpdate,
            int minWordFrequency,
            AtomicLong numWordsEncountered,
            InvertedIndex index) {
        super();
        this.tokenizer = tokenizer;
        this.stopWords = stopWords;
        this.lastUpdate = lastUpdate;
        this.cache = cache;
        this.minWordFrequency = minWordFrequency;
        this.numWordsEncountered = numWordsEncountered;
        this.index = index;
    }




    @Override
    public void onReceive(Object message) throws Exception {
        final Set<String> encountered = new HashSet<>();

        if(message  instanceof VocabWork) {
            final List<SequenceElement> document = new ArrayList<>();
            final VocabWork work = (VocabWork) message;
            if(work.getWork() == null || work.getWork().isEmpty())
                return;

            final String sentence = work.getWork();
            if(sentence.isEmpty() || sentence.length() <= 2) {
                work.increment();
                lastUpdate.getAndSet(System.currentTimeMillis());
                return;
            }



            Future<Object> f = Futures.future(new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                    numWordsEncountered.set(numWordsEncountered.get() + document.size());
                    Tokenizer t = tokenizer.create(sentence);
                    while(t.hasMoreTokens())  {
                        String token = t.nextToken();
                        if(token.isEmpty())
                            break;
                        processToken(token,encountered,document,work.isStem());
                    }

                    if(work.getLabel() != null)
                        index.addWordsToDoc(index.numDocuments(),document,work.getLabel());
                    else
                        index.addWordsToDoc(index.numDocuments(),document);
                    return null;
                }
            },context().dispatcher());
            f.onFailure(new OnFailure() {
                @Override
                public void onFailure(Throwable failure) throws Throwable {
                    log.error("Failure on vocab actor ",failure);
                }
            },context().dispatcher());
            f.onSuccess(new OnSuccess<Object>() {
                @Override
                public void onSuccess(Object result) throws Throwable {
                    work.increment();
                    lastUpdate.getAndSet(System.currentTimeMillis());
                }
            },context().dispatcher());


        }


        else if(message instanceof StreamWork) {
            StreamWork work = (StreamWork) message;
            List<SequenceElement> document = new ArrayList<>();

            InputStream is = work.getIs();
            if(is == null)
                return;
            boolean tryRead = false;
            try {
                if(is.available() > 0) {
                    tryRead = true;
                }
            }catch(Exception e) {
                tryRead = false;
            }

            if(!tryRead)
                return;

            Tokenizer t = tokenizer.create(is);

            while(t.hasMoreTokens())  {
                String token = t.nextToken();
                if(token == null || token.isEmpty())
                    break;
                processToken(token,encountered,document,false);

            }

            //adds the words to the document after all of them have been processed
            index.addWordsToDoc(index.numDocuments(),document);
            numWordsEncountered.set(numWordsEncountered.get() + document.size());

            IOUtils.closeQuietly(is);
            work.countDown();

            lastUpdate.getAndSet(System.currentTimeMillis());


        }



        else
            unhandled(message);
    }



    protected synchronized void processToken(String token,Set<String> encountered,List<SequenceElement> words,boolean stem) {
        if(stopWords.contains(token))
            token = "STOP";
        if(token.isEmpty())
            return;

        String oldToken = token;
        if(stem) {
            synchronized (stemmer) {
                stemmer.setCurrent(token);

                if(stemmer.stem() && stemmer.getCurrent() != null && !stemmer.getCurrent().isEmpty())
                    token = stemmer.getCurrent();
            }

        }

        if(token.isEmpty())
            token = oldToken;

        cache.incrementWordCount(token);


        if(!encountered.contains(token)) {
            cache.incrementDocCount(token,1);
            encountered.add(token);
        }


        SequenceElement token2;
        if(cache.hasToken(token))
            token2 = cache.tokenFor(token);
        else {
            token2 = new VocabWord(1.0, token);
            cache.addToken(token2);


        }
        words.add(token2);


        //note that for purposes of word frequency, the
        //internal vocab and the final vocab
        //at the class level contain the same references
        if(!Util.matchesAnyStopWord(stopWords,token) && !token.isEmpty()) {
            if(!cache.containsWord(token) && cache.wordFrequency(token) >= minWordFrequency) {
                int idx = cache.numWords();
                token2.setIndex(idx);
                cache.putVocabWord(token);
            }

            else  if(Util.matchesAnyStopWord(stopWords,token) && !token.isEmpty()) {
                token = "STOP";
                if(!cache.containsWord(token) && cache.wordFrequency(token) >= minWordFrequency) {
                    int idx = cache.numWords();
                    token2.setIndex(idx);
                    cache.putVocabWord(token);
                }


            }



        }


    }



}
