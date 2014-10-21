package org.deeplearning4j.bagofwords.vectorizer;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.routing.RoundRobinPool;
import org.deeplearning4j.models.word2vec.InputStreamCreator;
import org.deeplearning4j.models.word2vec.StreamWork;
import org.deeplearning4j.models.word2vec.VocabWork;
import org.deeplearning4j.models.word2vec.actor.VocabActor;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.Index;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Base text vectorizer for handling creation of vocab
 * @author Adam Gibson
 */
public abstract class BaseTextVectorizer implements TextVectorizer {

    protected VocabCache cache;
    protected static ActorSystem trainingSystem;
    protected TokenizerFactory tokenizerFactory;
    protected List<String> stopWords;
    private int layerSize = 0;
    protected int minWordFrequency = 5;
    protected DocumentIterator docIter;
    protected List<String> labels;
    protected SentenceIterator sentenceIterator;
    private static Logger log = LoggerFactory.getLogger(BaseTextVectorizer.class);

    public BaseTextVectorizer(){}

    protected BaseTextVectorizer(VocabCache cache, TokenizerFactory tokenizerFactory, List<String> stopWords, int layerSize, int minWordFrequency, DocumentIterator docIter, SentenceIterator sentenceIterator,List<String> labels) {
        this.cache = cache;
        this.tokenizerFactory = tokenizerFactory;
        this.stopWords = stopWords;
        this.layerSize = layerSize;
        this.minWordFrequency = minWordFrequency;
        this.docIter = docIter;
        this.sentenceIterator = sentenceIterator;
        this.labels = labels;
    }

    @Override
    public void fit() {
        if(trainingSystem == null)
            trainingSystem = ActorSystem.create();





        final AtomicLong semaphore = new AtomicLong(System.currentTimeMillis());
        final AtomicInteger queued = new AtomicInteger(0);

        final ActorRef vocabActor = trainingSystem.actorOf(
                new RoundRobinPool(Runtime.getRuntime().availableProcessors()).props(
                        Props.create(VocabActor.class, tokenizerFactory, cache, layerSize, stopWords, semaphore, minWordFrequency)));

		/* all words; including those not in the actual ending index */

        final AtomicInteger latch = new AtomicInteger(0);

        while(docIter != null && docIter.hasNext()) {

            vocabActor.tell(new StreamWork(new DefaultInputStreamCreator(docIter),latch),vocabActor);

            queued.incrementAndGet();
            if(queued.get() % 10000 == 0) {
                log.info("Sent " + queued);
                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();;
                }
            }

        }


        while(getSentenceIterator() != null && getSentenceIterator().hasNext()) {
            String sentence = getSentenceIterator().nextSentence();
            if(sentence == null)
                break;
            vocabActor.tell(new VocabWork(latch,sentence), vocabActor);
            queued.incrementAndGet();
            if(queued.get() % 10000 == 0) {
                log.info("Sent " + queued);
                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();;
                }
            }


        }





        while(latch.get() > 0) {
            log.info("Building vocab...");
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        trainingSystem.shutdown();


    }

    @Override
    public VocabCache vocab() {
        return cache;
    }

    public SentenceIterator getSentenceIterator() {
        return sentenceIterator;
    }

    public void setSentenceIterator(SentenceIterator sentenceIterator) {
        this.sentenceIterator = sentenceIterator;
    }

    public DocumentIterator getDocIter() {
        return docIter;
    }

    public void setDocIter(DocumentIterator docIter) {
        this.docIter = docIter;
    }

    public int getMinWordFrequency() {
        return minWordFrequency;
    }

    public void setMinWordFrequency(int minWordFrequency) {
        this.minWordFrequency = minWordFrequency;
    }

    public int getLayerSize() {
        return layerSize;
    }

    public void setLayerSize(int layerSize) {
        this.layerSize = layerSize;
    }

    public List<String> getStopWords() {
        return stopWords;
    }

    public void setStopWords(List<String> stopWords) {
        this.stopWords = stopWords;
    }

    public TokenizerFactory getTokenizerFactory() {
        return tokenizerFactory;
    }

    public void setTokenizerFactory(TokenizerFactory tokenizerFactory) {
        this.tokenizerFactory = tokenizerFactory;
    }


    public VocabCache getCache() {
        return cache;
    }

    public void setCache(VocabCache cache) {
        this.cache = cache;
    }



}
