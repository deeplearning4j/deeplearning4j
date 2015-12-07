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

package org.deeplearning4j.models.paragraphvectors;

import akka.actor.ActorSystem;
import lombok.Getter;
import lombok.NonNull;
import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.Word2VecConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyHolder;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.parallel.Parallelization;
import org.deeplearning4j.text.documentiterator.*;
import org.deeplearning4j.text.documentiterator.interoperability.DocumentIteratorConverter;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Basic idea behind ParagraphVectors is pretty simple: unsupervised way to learn differences/similarities between sentences/documents. It's main difference from w2v based on word order inference.
 * There's lots of practical uses for this algorithm: QA training, social activities differentiation, sentiment analysis, etc.
 * But please note: this algorithm requires serious hardware to be trained on large corpus since it's vocabulary size isn't limited to words, but documents are saved to vocabulary as well.
 *
 *
 * Paragraph Vectors:
 * [1] Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents. http://arxiv.org/pdf/1405.4053v2.pdf
 .. [2] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
 .. [3] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
 In Proceedings of NIPS, 2013.

 @author Adam Gibson

 */
public class ParagraphVectors extends Word2Vec {
    //labels are also vocab words

    protected Word2Vec existingModel;
    protected boolean trainWordVectors = false;

    /*
        labels were replaces with LabelsSource mechanics
     */
    //protected List<String> labels = new CopyOnWriteArrayList<>();

    @Getter protected LabelsSource labelsSource;

    /*
        That's new Iterator, that represents unified interface for Sentence/Document Iterators, with support of few label generation sources via LabelGenerator
     */
    protected LabelAwareIterator labelAwareIterator;

    protected static final Logger log = LoggerFactory.getLogger(ParagraphVectors.class);

    /**
     * Train the model
     */
    @Override
    public void fit() throws IOException {

    /*
        boolean loaded = buildVocab();
        //save vocab after building
        if (!loaded && saveVocab)
            vocab().saveVocab();
        if (stopWords == null)
            readStopWords();
*/

        log.info("Building vocab");
        // resetModel is always true for ParagraphVectors
        if (existingModel == null) {
            // initialize vector table
            // resetWeights is absolutely required after vocab transfer, due to algo internals.
            log.info("Building matrices & resetting weights...");

            buildVocab();
            lookupTable.resetWeights(true);
        } else {
            // if we have existing Word2Vec model provided, we have to use it, as source of syn0/syn1 changes.
            // good thing is that it can be used to implement vocab extension, required for w2v uptraining
            log.info("Importing matrices from existing Word2Vec model");
        }

        log.info("Total number of documents: " + labelsSource.getLabels().size());
        log.info("Training ParaVec multithreaded");
/*
        if (sentenceIter != null)
            sentenceIter.reset();
        if (docIter != null)
            docIter.reset();

*/

        totalWords = vocab.totalWordOccurrences(); //vectorizer.numWordsEncountered();
        totalWords *= numIterations;



        log.info("Processing sentences...");

        int epoch = 1;
        while (epoch <= epochs) {
            final AtomicLong wordsCount = new AtomicLong(0);
            LabelledAsyncIteratorDigitizer roller = new LabelledAsyncIteratorDigitizer(labelAwareIterator);
            roller.start();

            final AtomicLong documentCount = new AtomicLong(0);
            final VectorCalculationsThread[] threads = new VectorCalculationsThread[workers];
            // start processing threads
            for (int x = 0; x < workers; x++) {
                threads[x] = new VectorCalculationsThread(x, epoch, roller, documentCount, wordsCount);
                threads[x].start();
            }

            try {
                // block untill all lines are read at AsyncIteratorDigitizer
                roller.join();
            } catch (Exception e) {
                e.printStackTrace();
            }

            // wait untill all vector calculation threads are finished
            for (int x = 0; x < workers; x++) {
                try {
                    threads[x].join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            log.info("Epoch: " + epoch + "; Documents vectorized so far: " + documentCount.get());
            epoch++;
        }

/*
        final AtomicLong numWordsSoFar = new AtomicLong(0);


        final AtomicLong nextRandom = new AtomicLong(5);
        final AtomicInteger doc = new AtomicInteger(0);
        ExecutorService exec = new ThreadPoolExecutor(Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().availableProcessors(),
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<Runnable>(), new RejectedExecutionHandler() {
            @Override
            public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                executor.submit(r);
            }
        });



        final Queue<Pair<List<VocabWord>,Collection<VocabWord>>> batch2 = new ConcurrentLinkedDeque<>();
        int[] docs = vectorizer.index().allDocs();
        if(docs.length < 1)
            throw new IllegalStateException("No documents found");

        vectorizer.index().eachDocWithLabels(new Function<Pair<List<VocabWord>, Collection<String>>, Void>() {
            @Override
            public Void apply(@Nullable Pair<List<VocabWord>, Collection<String>> input) {
                List<VocabWord> batch = new ArrayList<>();
                addWords(input.getFirst(), nextRandom, batch);

                if (batch.isEmpty())
                    return null;

                Collection<VocabWord> docLabels = new ArrayList<>();
                for(String s : input.getSecond())
                    docLabels.add(vocab().wordFor(s));
                batch2.add(new Pair<>(batch, docLabels));

                doc.incrementAndGet();
                if (doc.get() > 0 && doc.get() % 10000 == 0)
                    log.info("Doc " + doc.get() + " done so far");

                return null;
            }



        }, exec);



        if(!batch2.isEmpty()) {
            jobQueue.add(new LinkedList<>(batch2));

        }


        exec.shutdown();
        try {
            exec.awaitTermination(1,TimeUnit.DAYS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        for(int i = 0; i < numIterations; i++)
            doIteration(batch2,numWordsSoFar,nextRandom);

*/
    }


    /**
     * Builds the vocabulary for training
     */
    @Override
    public boolean buildVocab() {


        VocabConstructor constructor = new VocabConstructor.Builder()
                .addSource(labelAwareIterator, minWordFrequency)
                .setTokenizerFactory(this.tokenizerFactory)
                .setStopWords(this.stopWords)
                .setTargetVocabCache(vocab)
                .fetchLabels(true)
                .build();

        constructor.buildJointVocabulary(false, true);

        /*
        super.buildVocab();

        for(String label : labels) {
            VocabWord word = new VocabWord(vocab.numWords(),label);

            // this sounds legit, since if we hav
            word.setIndex(vocab.numWords());
            vocab().addToken(word);
            vocab().putVocabWord(label);
        }
        */

        // add labels to vocab

/*
        readStopWords();

        if(vocab().vocabExists()) {
            log.info("Loading vocab...");
            vocab().loadVocab();
            lookupTable.resetWeights();
            return true;
        }


        if(invertedIndex == null)
            invertedIndex = new LuceneInvertedIndex.Builder()
                    .cache(vocab()).stopWords(stopWords)
                    .build();
        //vectorizer will handle setting up vocab meta data
        if(vectorizer == null) {
            vectorizer = new TfidfVectorizer.Builder().index(invertedIndex)
                    .cache(vocab()).iterate(docIter).iterate(sentenceIter).batchSize(batchSize)
                    .minWords(minWordFrequency).stopWords(stopWords)
                    .tokenize(tokenizerFactory).build();

            vectorizer.fit();

        }

        //includes unk
        else if(vocab().numWords() < 2)
            vectorizer.fit();

        for(String label : labels) {
            VocabWord word = new VocabWord(vocab.numWords(),label);
            word.setIndex(vocab.numWords());
            vocab().addToken(word);
            vocab().putVocabWord(label);
        }


        setup();
*/
        return false;
    }

    /**
     * Predict several based on the document.
     * Computes a similarity wrt the mean of the
     * representation of words in the document
     * @param document the document
     * @return the word distances for each label
     */
    public String predict(List<VocabWord> document) {
        INDArray arr = Nd4j.create(document.size(),this.layerSize);
        for(int i = 0; i < document.size(); i++) {
            arr.putRow(i,getWordVectorMatrix(document.get(i).getWord()));
        }

        INDArray docMean = arr.mean(0);
        Counter<String> distances = new Counter<>();

        for(String s : labelsSource.getLabels()) {
            INDArray otherVec = getWordVectorMatrix(s);
            double sim = Transforms.cosineSim(docMean, otherVec);
            distances.incrementCount(s, sim);
        }

        return distances.argMax();

    }


    /**
     * Predict several based on the document.
     * Computes a similarity wrt the mean of the
     * representation of words in the document
     * @param document the document
     * @return the word distances for each label
     */
    public Counter<String> predictSeveral(List<VocabWord> document) {
        INDArray arr = Nd4j.create(document.size(),this.layerSize);
        for(int i = 0; i < document.size(); i++) {
            arr.putRow(i,getWordVectorMatrix(document.get(i).getWord()));
        }

        INDArray docMean = arr.mean(0);
        Counter<String> distances = new Counter<>();

        for(String s : labelsSource.getLabels()) {
            INDArray otherVec = getWordVectorMatrix(s);
            double sim = Transforms.cosineSim(docMean, otherVec);
            distances.incrementCount(s, sim);
        }

        return distances;

    }




    /**
     * Train on a list of vocab words
     * @param sentenceWithLabel the list of vocab words to train on
     */
    public void trainSentence(final Pair<List<SequenceElement>, Collection<SequenceElement>> sentenceWithLabel,AtomicLong nextRandom,double alpha) {
        if(sentenceWithLabel == null || sentenceWithLabel.getFirst().isEmpty())
            return;

        /*
                if trainWordVectors == true, we'll train Word2Vec representations at the same time
                This option should be used with caution, since currently dl4j meta does not distinguish sentences and lines of text.
                So, this will effectively mean improper w2v models if input is not single-line documents.
                TODO: to be fixed ^^^ proposed fix: introduce sentence boundaries detector in the w2v tokenization pipeline.
          */
        if (trainWordVectors && existingModel == null) for(int i = 0; i < sentenceWithLabel.getFirst().size(); i++) {
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            skipGram(i, sentenceWithLabel.getFirst(), (int) nextRandom.get() % window,nextRandom,alpha);
        }

        for(int i = 0; i < sentenceWithLabel.getFirst().size(); i++) {
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            dbow(i, sentenceWithLabel, (int) nextRandom.get() % window, nextRandom, alpha);
        }

    }

    /**
     * Train the distributed bag of words
     * model
     * @param i the word to train
     * @param sentenceWithLabel the sentence with labels to train
     * @param b
     * @param nextRandom
     * @param alpha
     */
    public void dbow(int i, Pair<List<VocabWord>, Collection<VocabWord>> sentenceWithLabel, int b, AtomicLong nextRandom, double alpha) {

        final VocabWord word = sentenceWithLabel.getFirst().get(i);
        List<VocabWord> sentence = sentenceWithLabel.getFirst();
        List<VocabWord> labels = (List<VocabWord>) sentenceWithLabel.getSecond();
    //    final VocabWord word = labels.get(0);

        if(word == null || sentence.isEmpty())
            return;


        int end =  window * 2 + 1 - b;
        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < labels.size()) {
                    VocabWord lastWord = labels.get(c);
                    iterate(word, lastWord,nextRandom,alpha);
                }
            }
        }
    }

    public List<String> getLabels() {
        return labelsSource.getLabels();
    }

    @Deprecated
    public void setLabels(List<String> labels) {
        //this.labels = labels;
    }

    /*
            This method is marked as @Deprecated, since it's not used anymore: calculations were moved into VectorCalculationsThread
     */
    @Deprecated
    private void doIteration(Queue<Pair<List<VocabWord>,Collection<VocabWord>>> batch2,final AtomicLong numWordsSoFar,final AtomicLong nextRandom) {
        ActorSystem actorSystem = ActorSystem.create();
        final AtomicLong lastReport = new AtomicLong(System.currentTimeMillis());
        Parallelization.iterateInParallel(batch2, new Parallelization.RunnableWithParams<Pair<List<VocabWord>, Collection<VocabWord>>>() {
            @Override
            public void run(Pair<List<VocabWord>, Collection<VocabWord>> sentenceWithLabel, Object[] args) {
                double alpha = Math.max(minLearningRate, ParagraphVectors.this.alpha.get() * (1 - (1.0 * (double) numWordsSoFar.get() / (double) totalWords)));
                long diff = Math.abs(lastReport.get() - numWordsSoFar.get());
                if(numWordsSoFar.get() > 0 && diff >=  10000) {
                    log.info("Words so far " + numWordsSoFar.get() + " with alpha at " + alpha);
                    lastReport.set(numWordsSoFar.get());
                }
                long increment = 0;
                double diff2 = 0.0;
                trainSentence(sentenceWithLabel, nextRandom, alpha);
                increment += sentenceWithLabel.getFirst().size();


         //       log.info("Train sentence avg took " + diff2 / (double) sentenceWithLabel.getFirst().size());
                numWordsSoFar.set(numWordsSoFar.get() + increment);
            }
        },actorSystem);
    }

    /**
     * This is temporary solution. Mechanics will be unified for w2v and d2v with PrefetchingSentenceInterator
     */
    protected class LabelledAsyncIteratorDigitizer extends Thread implements Runnable {
        private LabelAwareIterator backendIterator;
        private LinkedBlockingQueue<LabelledDocument> buffer = new LinkedBlockingQueue<>();
        private AtomicBoolean isRunning = new AtomicBoolean(false);
        private AtomicLong nextRandom;

        public LabelledAsyncIteratorDigitizer(LabelAwareIterator iterator) {
            this.backendIterator = iterator;
            this.nextRandom = new AtomicLong(workers + 1);

            this.backendIterator.reset();
        }

        @Override
        public void run() {
            isRunning.set(true);
            while (this.backendIterator.hasNextDocument()) {
                if (buffer.size() < 1000) {
                    int cnt = 0;
                    while (cnt < 1000 && this.backendIterator.hasNextDocument()) {
                        LabelledDocument document = this.backendIterator.nextDocument();

                        Tokenizer tokenizer = tokenizerFactory.create(document.getContent());
                        List<String> tokens = tokenizer.getTokens();

                        List<VocabWord> words =  ParagraphVectors.this.digitizeSentence(tokens, nextRandom);

                        document.setReferencedContent(words);

                        // nullify string document representation, since we don't really need it
                        document.setContent(null);

                        buffer.add(document);
                        cnt++;
                    }
                } else try {
                    Thread.sleep(10);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            isRunning.set(false);
            log.debug("AsyncIterator finished");
        }

        public boolean hasMoreDocuments() {
            return buffer.size() > 0 || isRunning.get();
        }

        public LabelledDocument nextLabelledDocument() {
            try {
                return buffer.poll(3L, TimeUnit.SECONDS);
            } catch (Exception e) {
                return null;
            }
        }
    }

    public static class Builder extends Word2Vec.Builder {
        protected List<String> labels;
        protected LabelsSource generator;
        protected LabelAwareIterator labelAwareIterator;
        protected Word2Vec existingW2V;
        protected boolean trainWordVectors;
        protected double sampling  = 0;


        public Builder() {
            super();
        }

        public Builder(@NonNull Word2VecConfiguration configuration) {
            super(configuration);
        }

        @Deprecated
        @Override
        public Builder index(InvertedIndex index) {
            super.index(index);
            return this;
        }

        @Override
        @Deprecated
        public Builder hugeModelExpected(boolean reallyExpected) {
            throw new IllegalStateException("This method is NOT supported in ParagraphVectors");
        }

        @Override
        public Builder workers(int workers) {
            super.workers(workers);
            return this;
        }

        /**
         * You can provide existing WordVectors model to be used as vocab/weights source for your new ParagraphVectors model.
         *
         * @param word2Vec existing Word2Vec model
         * @return builder object
         */
        protected Builder wordVectorsModel(@NonNull Word2Vec word2Vec) {
            this.existingW2V = word2Vec;
            return this;
        }

        @Override
        public Builder sampling(double sample) {
            super.sampling(sample);
            return this;
        }

        @Override
        public Builder negativeSample(double negative) {
            super.negativeSample(negative);
            return this;
        }

        @Override
        public Builder minLearningRate(double minLearningRate) {
            super.minLearningRate(minLearningRate);
            return this;
        }

        @Override
        public Builder useAdaGrad(boolean useAdaGrad) {
            super.useAdaGrad(useAdaGrad);
            return this;
        }

        @Override
        @Deprecated
        public Builder vectorizer(TextVectorizer textVectorizer) {
            super.vectorizer(textVectorizer);
            return this;
        }

        /**
         * This method has no effect in ParagraphVectors, due to the nature of algorithm.
         *
         * @param learningRateDecayWords
         * @return
         */
        @Override
        @Deprecated
        public Builder learningRateDecayWords(int learningRateDecayWords) {
            super.learningRateDecayWords(learningRateDecayWords);
            return this;
        }

        /**
         * This method has no effect in ParagraphVectors, due to the nature of algorithm.
         *
         * @param
         * @return
         */
        @Override
        @Deprecated
        public Builder resetModel(boolean reallyReset) {
            return this;
        }

        /**
         * This method has no effect in ParagraphVectors, due to the nature of algorithm.
         *
         * @param batchSize
         * @return
         */
        @Override
        @Deprecated
        public Builder batchSize(int batchSize) {
            super.batchSize(batchSize);
            return this;
        }

        /**
         * This method has no effect in ParagraphVectors, due to the nature of algorithm.
         *
         * @param saveVocab
         * @return
         */
        @Override
        @Deprecated
        public Builder saveVocab(boolean saveVocab) {
            super.saveVocab(saveVocab);
            return this;
        }

        @Override
        public Builder seed(long seed) {
            super.seed(seed);
            return this;
        }

        @Override
        public Builder iterations(int iterations) {
            super.iterations(iterations);
            return this;
        }

        @Override
        public Builder learningRate(double lr) {
            super.learningRate(lr);
            return this;
        }


        public Builder iterate(@NonNull LabelAwareIterator iter) {
            this.labelAwareIterator = iter;
            return this;
        }

        @Override
        public Builder iterate(@NonNull DocumentIterator iter) {
            this.docIter = iter;
            return this;
        }

        @Override
        public  Builder vocabCache(VocabCache cache) {
            super.vocabCache(cache);
            return this;
        }

        @Override
        public Builder minWordFrequency(int minWordFrequency) {
            super.minWordFrequency(minWordFrequency);
            return this;
        }

        @Override
        public Builder tokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            super.tokenizerFactory(tokenizerFactory);
            return this;
        }

        @Override
        public Builder layerSize(int layerSize) {
            super.layerSize(layerSize);
            return this;
        }

        @Override
        public Builder stopWords(@NonNull List<String> stopWords) {
            super.stopWords(stopWords);
            return this;
        }

        @Override
        public Builder windowSize(int window) {
            super.windowSize(window);
            return this;
        }

        @Override
        public Builder iterate(@NonNull SentenceIterator iter) {
            this.iter = iter;
            return this;
        }

        /**
         * If set to true, model will build ParagraphVectors together with WordVectors over the same corpus.
         * Please note, you shouldn't set this to true unless your documents are single-line. Otherwise WordVectors model will be highly inaccurate.
         * If your documents are NOT single-line, you can build WordVectors using Word2Vec implementation, and pass obtained model into ParagraphVectors as features source.
         *
         * @param reallyTrain
         * @return
         */
        public Builder trainWordVectors(boolean reallyTrain) {
            this.trainWordVectors = reallyTrain;
            return this;
        }

        @Override
        public Builder lookupTable(@NonNull WeightLookupTable lookupTable) {
            super.lookupTable(lookupTable);
            return this;
        }

        /**
         * Specify labels source to be used
         * @param labels the labels to specify
         * @return builder pattern
         */
        @Deprecated
        public Builder labels(@NonNull List<String> labels) {
            this.labels = labels;
            this.generator = new LabelsSource(labels);
            return this;
        }

        /**
         * Specify template for labels generation.
         * For more info: LabelsSource javadoc
         *
         * @param template the template to be used
         * @return builder pattern
         */
        public Builder labelsTemplate(@NonNull String template) {
            this.generator = new LabelsSource(template);
            return this;
        }

        /**
         * Specify labels source to be used
         *
         * @param generator LabelsSource to be used
         * @return builder pattern
         */
        public Builder labelsGenerator(@NonNull LabelsSource generator) {
            this.generator = generator;
            return this;
        }

        @Override
        public Builder epochs(int numEpochs) {
            this.numEpochs = numEpochs;
            return this;
        }

        @Override
        public ParagraphVectors build() {
            ParagraphVectors ret = new ParagraphVectors();

            /*
                    Before proceeding, we have to convert passed in iterators to LabelAwareIterator, so can be sure any type of iterator works equally: we have both documents, and labels.
             */
            if (this.generator == null) this.generator = new LabelsSource();
            if (docIter != null) {
                /*
                        we're going to work with DocumentIterator.
                        First, we have to assume that user can provide LabelAwareIterator. In this case we'll use them, as provided source, and collec labels provided there
                        Otherwise we'll go for own labels via LabelsSource
                */

                if (docIter instanceof LabelAwareDocumentIterator) this.labelAwareIterator = new DocumentIteratorConverter((LabelAwareDocumentIterator) docIter, generator);
                    else this.labelAwareIterator = new DocumentIteratorConverter(docIter, generator);
            } else if (iter != null) {
                // we have SentenceIterator. Mechanics will be the same, as above
                if (iter instanceof LabelAwareSentenceIterator) this.labelAwareIterator = new SentenceIteratorConverter((LabelAwareSentenceIterator) iter, generator);
                    else this.labelAwareIterator = new SentenceIteratorConverter(iter, generator);
            } else if (labelAwareIterator != null) {
                // if we have LabelAwareIterator defined, we have to be sure that LabelsSource is propagated properly
                this.generator = labelAwareIterator.getLabelsSource();
            } else  {
                // we have nothing, probably that's restored model building. ignore iterator for now.
                // probably there's few reasons to move iterator initialization code into ParagraphVectors methos. Like protected setLabelAwareIterator method.
                // TODO: to be investigated ^^^
            }

            // whole corpus will be read from unified LabelAwareIterator
            ret.labelAwareIterator = this.labelAwareIterator;

            ret.alpha.set(lr);
            ret.window = window;
            ret.useAdaGrad = useAdaGrad;
            ret.minLearningRate = minLearningRate;
            ret.vectorizer = textVectorizer;
            ret.stopWords = stopWords;
            ret.minWordFrequency = minWordFrequency;
            ret.setVocab(vocabCache);
            ret.minWordFrequency = minWordFrequency;
            ret.numIterations = iterations;
            ret.seed = seed;
            ret.numIterations = iterations;
            ret.saveVocab = saveVocab;
            ret.batchSize = batchSize;
            ret.sample = sampling;
            ret.workers = workers;
            ret.invertedIndex = index;
            ret.lookupTable = lookupTable;
            ret.epochs = this.numEpochs;

            // resetModel should be always true for ParagraphVectors
            ret.resetModel = true;
            ret.existingModel = this.existingW2V;
            ret.trainWordVectors = this.trainWordVectors;
            ret.labelsSource = this.generator;

            try {
                if (tokenizerFactory == null)
                    tokenizerFactory = new UimaTokenizerFactory();
            }catch(Exception e) {
                throw new RuntimeException(e);
            }

            if(vocabCache == null) {
                vocabCache = new InMemoryLookupCache();

                ret.setVocab(vocabCache);
            }

            if(lookupTable == null) {
                lookupTable = new InMemoryLookupTable.Builder().negative(negative)
                        .useAdaGrad(useAdaGrad).lr(lr).cache(vocabCache)
                        .vectorLength(layerSize).build();
            }
            ret.lookupTable = lookupTable;
            ret.tokenizerFactory = tokenizerFactory;

            // VocabularyHolder is used ONLY for fit() purposes, as intermediate data storage
            if (this.vocabCache!= null)
                // if VocabCache is set, build VocabHolder on top of it. Just for compatibility
                // please note: all words in VocabCache will be treated as SPECIAL, so they wont be affected by minWordFrequency
                ret.vocabularyHolder = new VocabularyHolder.Builder()
                        .externalCache(vocabCache)
                        .hugeModelExpected(hugeModelExpected)
                        .minWordFrequency(minWordFrequency)
                        .scavengerActivationThreshold(this.configuration.getScavengerActivationThreshold())
                        .scavengerRetentionDelay(this.configuration.getScavengerRetentionDelay())
                        .build();
            else ret.vocabularyHolder = new VocabularyHolder.Builder()
                    .hugeModelExpected(hugeModelExpected)
                    .minWordFrequency(minWordFrequency)
                    .scavengerActivationThreshold(this.configuration.getScavengerActivationThreshold())
                    .scavengerRetentionDelay(this.configuration.getScavengerRetentionDelay())
                    .build();

            ret.labelsSource = this.generator;

            this.configuration.setLearningRate(lr);
            this.configuration.setLayersSize(layerSize);
            this.configuration.setHugeModelExpected(hugeModelExpected);
            this.configuration.setWindow(window);
            this.configuration.setMinWordFrequency(minWordFrequency);
            this.configuration.setIterations(iterations);
            this.configuration.setSeed(seed);
            this.configuration.setBatchSize(batchSize);
            this.configuration.setLearningRateDecayWords(learningRateDecayWords);
            this.configuration.setMinLearningRate(minLearningRate);
            this.configuration.setSampling(this.sampling);
            this.configuration.setUseAdaGrad(useAdaGrad);
            this.configuration.setNegative(negative);
            this.configuration.setEpochs(this.numEpochs);

            ret.configuration = this.configuration;

            return ret;
/*
            if(iter == null) {
                ParagraphVectors ret = new ParagraphVectors();
                ret.window = window;
                ret.alpha.set(lr);
                ret.vectorizer = textVectorizer;
                ret.stopWords = stopWords;
                ret.setVocab(vocabCache);
                ret.numIterations = iterations;
                ret.minWordFrequency = minWordFrequency;
                ret.seed = seed;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
                ret.sample = sampling;
                ret.workers = workers;
                ret.invertedIndex = index;
                ret.lookupTable = lookupTable;
                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache();

                    ret.setVocab(vocabCache);
                }

                if(lookupTable == null) {
                    lookupTable = new InMemoryLookupTable.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr).cache(vocabCache)
                            .vectorLength(layerSize).build();
                }


                ret.docIter = docIter;
                ret.lookupTable = lookupTable;
                ret.tokenizerFactory = tokenizerFactory;
                ret.labels = labels;
                return ret;
            }

            else {
                ParagraphVectors ret = new ParagraphVectors();
                ret.alpha.set(lr);
                ret.sentenceIter = iter;
                ret.window = window;
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
                ret.vectorizer = textVectorizer;
                ret.stopWords = stopWords;
                ret.minWordFrequency = minWordFrequency;
                ret.setVocab(vocabCache);
                ret.docIter = docIter;
                ret.minWordFrequency = minWordFrequency;
                ret.numIterations = iterations;
                ret.seed = seed;
                ret.numIterations = iterations;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.sample = sampling;
                ret.workers = workers;
                ret.invertedIndex = index;
                ret.lookupTable = lookupTable;

                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache();

                    ret.setVocab(vocabCache);
                }

                if(lookupTable == null) {
                    lookupTable = new InMemoryLookupTable.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr).cache(vocabCache)
                            .vectorLength(layerSize).build();
                }
                ret.lookupTable = lookupTable;
                ret.tokenizerFactory = tokenizerFactory;
                ret.labels = labels;
                return ret;
            }
         */
        }
    }

    private class VectorCalculationsThread extends Thread implements Runnable {
        private int epoch;
        private LabelledAsyncIteratorDigitizer roller;
        private AtomicLong nextRandom;
        private AtomicLong documentCounter;
        private AtomicLong numWordsSoFar;

        public VectorCalculationsThread(int threadId, int epochId, @NonNull LabelledAsyncIteratorDigitizer roller, AtomicLong documentCounter, AtomicLong wordsCounter) {
            this.roller = roller;
            this.epoch = epochId;
            this.nextRandom = new AtomicLong(threadId);
            this.documentCounter = documentCounter;
            this.numWordsSoFar = wordsCounter;

            this.setName("ParaVec calculations thread. Thread id: " + threadId);
        }

        @Override
        public void run() {
            // this queue is used for training each word from sentence against label, derived for each sentence
            final Queue<Pair<List<SequenceElement>,Collection<SequenceElement>>> batch2 = new ConcurrentLinkedDeque<>();

            while (roller.hasMoreDocuments()) {
                LabelledDocument document = roller.nextLabelledDocument();

                if (document != null && document.getReferencedContent() != null && !document.getReferencedContent().isEmpty()) {
                    List<VocabWord> batch = new ArrayList<>();
                    ParagraphVectors.this.addWords(document.getReferencedContent(), nextRandom, batch);
//                    documentCounter.incrementAndGet();

                    if (batch.isEmpty()) {
                        log.info("Empty batch!");
                        continue;
                    }

                    Collection<SequenceElement> docLabels = new ArrayList<>();
                    docLabels.add(vocab.wordFor(document.getLabel()));
                    //batch2.add();

                    double alpha = 0.025;
                    for(int i = 0; i < numIterations; i++) {
                //        doIteration(batch2, numWordsSoFar, nextRandom);

                        alpha = Math.max(minLearningRate, ParagraphVectors.this.alpha.get() * (1 - (1.0 * (double) numWordsSoFar.get() / (double) totalWords)));

                        trainSentence(new Pair<>(batch, docLabels), nextRandom, alpha);


                        //       log.info("Train sentence avg took " + diff2 / (double) sentenceWithLabel.getFirst().size());
                        numWordsSoFar.addAndGet(document.getReferencedContent().size());
                    }

                    if (documentCounter.incrementAndGet() % 10000 == 0) log.info("Epoch: [" + epoch + "], Documents learned: [" + documentCounter.get() + "], Alpha: ["+alpha+"]");
                }
            }
        }
    }
}
