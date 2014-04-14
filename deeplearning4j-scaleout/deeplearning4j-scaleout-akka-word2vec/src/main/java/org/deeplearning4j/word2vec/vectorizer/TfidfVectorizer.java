package org.deeplearning4j.word2vec.vectorizer;

import akka.actor.ActorSystem;
import akka.dispatch.Futures;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.stopwords.StopWords;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.SetUtils;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.DefaultTokenizerFactory;
import org.deeplearning4j.word2vec.tokenizer.Tokenizer;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.word2vec.util.Util;
import org.deeplearning4j.word2vec.viterbi.Index;
import org.jblas.DoubleMatrix;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

import scala.concurrent.Await;
import scala.concurrent.Future;
import scala.concurrent.duration.Duration;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Turns a set of documents in to a tfidf bag of words
 * @author Adam Gibson
 */
public class TfidfVectorizer implements TextVectorizer {

    private Index vocab = new Index();
    private LabelAwareSentenceIterator sentenceIterator;
    private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    private List<String> labels;
    private Counter<String> tfIdfWeights;
    private int numTop = -1;
    private List<String> stopWords;
    private Counter<String> tf = new Counter<>();
    private Counter<String> idf = new Counter<>();
    private int numDocs = 0;
    private static Logger log = LoggerFactory.getLogger(TfidfVectorizer.class);

    /**
     *
     * @param sentenceIterator the document iterator
     * @param tokenizerFactory the tokenizer for individual tokens
     * @param labels the possible labels
     */
    public TfidfVectorizer(LabelAwareSentenceIterator sentenceIterator,TokenizerFactory tokenizerFactory,List<String> labels,int numTop) {
        this.sentenceIterator = sentenceIterator;
        if(!this.sentenceIterator.hasNext())
            this.sentenceIterator.reset();
        this.tokenizerFactory = tokenizerFactory;
        this.labels = labels;
        this.numTop = numTop;
        stopWords = StopWords.getStopWords();

        process();
        initIndexFromTfIdf();


    }


    /**
     * Creates this tfidf vectorizer with no vocab limit
     * @param sentenceIterator the document iterator
     * @param tokenizerFactory the tokenizer for individual tokens
     * @param labels the possible labels
     */
    public TfidfVectorizer(LabelAwareSentenceIterator sentenceIterator,TokenizerFactory tokenizerFactory,List<String> labels) {
        this(sentenceIterator,tokenizerFactory,labels,-1);

    }

    private Collection<String> allWords() {
        return SetUtils.union(tf.keySet(),idf.keySet());
    }




    private Counter<String> tfIdfWeights() {
        Counter<String> ret = new Counter<>();
        for(String word  : allWords())
            ret.setMinCount(word,tfidfWord(word));
        return ret;
    }

    private void initIndexFromTfIdf() {
        for(String key : tfIdfWeights.keySet())
            this.vocab.add(key);
    }



    private double tfidfWord(String word) {
        return MathUtils.tfidf(tfForWord(word),idfForWord(word));
    }


    private double tfForWord(String word) {
        return MathUtils.tf((int) tf.getCount(word));
    }

    private double idfForWord(String word) {
        return MathUtils.idf(numDocs,idf.getCount(word));
    }

    /* calculate tfidf scores */
    private void process() {
        ActorSystem actorSystem = ActorSystem.create("Tfidf-vectorizer");
        List<Future<Void>> futures = new CopyOnWriteArrayList<>();
        while(sentenceIterator.hasNext()) {
            final Counter<String> runningTotal = Util.parallelCounter();
            final Counter<String> documentOccurrences = Util.parallelCounter();
            futures.add(Futures.future(new Callable<Void>() {

                /**
                 * Computes a result, or throws an exception if unable to do so.
                 *
                 * @return computed result
                 * @throws Exception if unable to compute a result
                 */
                @Override
                public Void call() throws Exception {
                    Tokenizer tokenizer = tokenizerFactory.create(sentenceIterator.nextSentence());
                    numDocs++;
                    for (String token : tokenizer.getTokens()) {
                        if (stopWords.contains(token))
                            continue;
                        runningTotal.incrementCount(token, 1.0);
                        //idf
                        if (!documentOccurrences.containsKey(token))
                            documentOccurrences.setMinCount(token, 1.0);
                    }

                    idf.incrementAll(documentOccurrences);
                    tf.incrementAll(runningTotal);
                    return null;
                }
            }, actorSystem.dispatcher()));

        }


        log.info("Number of documents was " + futures.size());

        Iterable<Future<Void>> fIter = (Iterable<Future<Void>>) futures;
        try {
            Future<Iterable<Void>> composed = Futures.sequence( fIter,actorSystem.dispatcher());
            log.info("Awaiting futures results");
            Await.result(composed, Duration.Inf());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        actorSystem.shutdown();

        tfIdfWeights = tfIdfWeights();
        if(numTop > 0)
            tfIdfWeights.keepTopNKeys(numTop);
    }


    @Override
    public DataSet vectorize() {
        process();
        sentenceIterator.reset();
        List<DataSet> data = new ArrayList<>();
        while(sentenceIterator.hasNext()) {
            data.add(vectorize(sentenceIterator.nextSentence(), sentenceIterator.currentLabel()));
        }

        return DataSet.merge(data);
    }

    private DoubleMatrix tfidfForInput(String text) {
        DoubleMatrix ret = new DoubleMatrix(1,vocab.size());
        Tokenizer tokenizer = tokenizerFactory.create(text);
        List<String> tokens = tokenizer.getTokens();

        for(int i = 0;i  < tokens.size(); i++) {
            int idx = vocab.indexOf(tokens.get(i));
            if(idx >= 0)
                ret.put(idx,tfidfWord(tokens.get(i)));
        }

        return ret;

    }

    private DoubleMatrix tfidfForInput(InputStream is) {
        try {
            String text = new String(IOUtils.toByteArray(is));
            return tfidfForInput(text);
        }catch(Exception e) {
            throw new RuntimeException(e);
        }


    }


    @Override
    public DataSet vectorize(InputStream is, String label) {
        return new DataSet(tfidfForInput(is),MatrixUtil.toOutcomeVector(labels.indexOf(label),labels.size()));
    }

    @Override
    public DataSet vectorize(String text, String label) {
        DoubleMatrix tfidf  = tfidfForInput(text);
        DoubleMatrix label2 = MatrixUtil.toOutcomeVector(labels.indexOf(label),labels.size());
        return new DataSet(tfidf,label2);
    }

    @Override
    public DataSet vectorize(File input, String label) {
        try {
            return vectorize(FileUtils.readFileToString(input),label);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
