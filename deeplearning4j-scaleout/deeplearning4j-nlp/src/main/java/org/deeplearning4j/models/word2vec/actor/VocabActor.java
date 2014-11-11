package org.deeplearning4j.models.word2vec.actor;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import akka.dispatch.Futures;
import akka.dispatch.OnFailure;
import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.lang3.time.StopWatch;
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
    private static Logger log = LoggerFactory.getLogger(VocabActor.class);


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
        Set<String> encountered = new HashSet<>();

        if(message  instanceof VocabWork) {
            final List<VocabWord> document = new ArrayList<>();
            final VocabWork work = (VocabWork) message;
            if(work.getWork() == null || work.getWork().isEmpty())
                return;
            //work.getCount().incrementAndGet();
            String sentence = work.getWork();
            if(sentence.isEmpty() || sentence.length() <= 2) {
                work.countDown();
                return;
            }
            Tokenizer t = tokenizer.create(sentence);
            while(t.hasMoreTokens())  {
                String token = t.nextToken();
                processToken(token,encountered,document);
            }


            Future<Object> f = Futures.future(new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                    index.addWordsToDoc(index.numDocuments(),document);
                    numWordsEncountered.set(numWordsEncountered.get() + document.size());
                    work.countDown();

                    lastUpdate.getAndSet(System.currentTimeMillis());
                    return null;
                }
            },context().dispatcher());
            f.onFailure(new OnFailure() {
                @Override
                public void onFailure(Throwable failure) throws Throwable {
                    log.error("Failure on vocab actor ",failure);
                }
            },context().dispatcher());



        }


        else if(message instanceof StreamWork) {
            StreamWork work = (StreamWork) message;
            List<VocabWord> document = new ArrayList<>();

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
                processToken(token,encountered,document);

            }

            //adds the words to the document after all of them have bene processed
            index.addWordsToDoc(index.numDocuments(),document);
            numWordsEncountered.set(numWordsEncountered.get() + document.size());

            IOUtils.closeQuietly(is);
            work.countDown();

            lastUpdate.getAndSet(System.currentTimeMillis());


        }



        else
            unhandled(message);
    }



    protected void processToken(String token,Set<String> encountered,List<VocabWord> words) {
        if(stopWords.contains(token))
            token = "STOP";
        if(token.isEmpty())
            return;

        cache.incrementWordCount(token);


        if(!encountered.contains(token)) {
            cache.incrementDocCount(token,1);
            encountered.add(token);
        }


        VocabWord token2 = null;
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
        if(!Util.matchesAnyStopWord(stopWords,token) && token != null && !token.isEmpty()) {
            if(!cache.containsWord(token) && cache.wordFrequency(token) >= minWordFrequency) {
                int idx = cache.numWords();
                token2.setIndex(idx);
                cache.putVocabWord(token);
            }

            else  if(Util.matchesAnyStopWord(stopWords,token) && token != null && !token.isEmpty()) {
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
