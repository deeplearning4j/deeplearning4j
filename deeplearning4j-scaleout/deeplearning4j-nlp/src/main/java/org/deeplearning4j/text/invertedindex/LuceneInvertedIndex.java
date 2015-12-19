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

package org.deeplearning4j.text.invertedindex;

import com.google.common.base.Function;
import lombok.NonNull;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.Bits;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.stopwords.StopWords;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Lucene based inverted index
 *
 * @author Adam Gibson
 */
public class LuceneInvertedIndex<T extends SequenceElement> implements InvertedIndex<T>, IndexReader.ReaderClosedListener,
    Iterator<List<T>> {

    private transient  Directory dir;
    private transient IndexReader reader;
    private   transient Analyzer analyzer;
    private VocabCache<T> vocabCache;
    public final static String WORD_FIELD = "word";
    public final static String LABEL = "label";

    private int numDocs = 0;    // FIXME this is never used
    private AtomicBoolean indexBeingCreated = new AtomicBoolean(false);
    private static final Logger log = LoggerFactory.getLogger(LuceneInvertedIndex.class);
    public final static String INDEX_PATH = "word2vec-index";
    private AtomicBoolean readerClosed = new AtomicBoolean(false);
    private AtomicInteger totalWords = new AtomicInteger(0);
    private int batchSize = 1000;
    private List<List<T>> miniBatches = new CopyOnWriteArrayList<>();
    private List<T> currMiniBatch = Collections.synchronizedList(new ArrayList<T>());
    private double sample = 0;
    private AtomicLong nextRandom = new AtomicLong(5);
    private String indexPath = INDEX_PATH;
    private Queue<List<T>> miniBatchDocs = new ConcurrentLinkedDeque<>();   // FIXME this is never used
    private AtomicBoolean miniBatchGoing = new AtomicBoolean(true);
    private boolean miniBatch = false;
    public final static String DEFAULT_INDEX_DIR = "word2vec-index";
    private transient SearcherManager searcherManager;
    private transient ReaderManager  readerManager;
    private transient TrackingIndexWriter indexWriter;
    private transient LockFactory lockFactory;

    public LuceneInvertedIndex(VocabCache vocabCache,boolean cache) {
        this(vocabCache,cache,DEFAULT_INDEX_DIR);
    }



    public LuceneInvertedIndex(VocabCache vocabCache,boolean cache,String indexPath) {  // FIXME cache is not used
        this.vocabCache = vocabCache;
        this.indexPath = indexPath;
        if(new File(indexPath).exists()) {
            indexPath = UUID.randomUUID().toString();
            log.warn("Changing index path to" + indexPath);
        }
        initReader();
    }

    public LuceneInvertedIndex(VocabCache vocabCache) {
        this(vocabCache,false,DEFAULT_INDEX_DIR);
    }

    @Override
    public Iterator<List<List<T>>> batchIter(int batchSize) {
        return new BatchDocIter(batchSize);
    }

    @Override
    public Iterator<List<T>> docs() {
        return new DocIter();
    }

    @Override
    public void unlock() {
        if(lockFactory == null)
            lockFactory = NativeFSLockFactory.getDefault();
    }

    @Override
    public void cleanup() {
        try {
            indexWriter.deleteAll();
            indexWriter.getIndexWriter().commit();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public double sample() {
        return sample;
    }

    @Override
    public Iterator<List<T>> miniBatches() {
        return this;
    }

    @Override
    public synchronized List<T> document(int index) {
        List<T> ret = new CopyOnWriteArrayList<>();
        try {
            DirectoryReader reader = readerManager.acquire();
            Document doc = reader.document(index);
            reader.close();
            String[] values = doc.getValues(WORD_FIELD);
            for(String s : values) {
                T word = vocabCache.wordFor(s);
                if(word != null)
                    ret.add(word);
            }



        }


        catch (Exception e) {
            e.printStackTrace();
        }
        return ret;
    }



    @Override
    public int[] documents(T vocabWord) {
        try {
            TermQuery query = new TermQuery(new Term(WORD_FIELD,vocabWord.getLabel().toLowerCase()));
            searcherManager.maybeRefresh();
            IndexSearcher searcher = searcherManager.acquire();
            TopDocs topdocs = searcher.search(query,Integer.MAX_VALUE);
            int[] ret = new int[topdocs.totalHits];
            for(int i = 0; i < topdocs.totalHits; i++) {
                ret[i] = topdocs.scoreDocs[i].doc;
            }


            searcherManager.release(searcher);

            return ret;
        }
        catch(AlreadyClosedException e) {
            return documents(vocabWord);
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public int numDocuments() {
        try {
            readerManager.maybeRefresh();
            DirectoryReader reader = readerManager.acquire();
            int ret = reader.numDocs();
            readerManager.release(reader);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public int[] allDocs() {


        DirectoryReader reader;
        try {
            readerManager.maybeRefreshBlocking();
            reader = readerManager.acquire();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        int[] docIds = new int[reader.maxDoc()];
        if(docIds.length < 1)
            throw new IllegalStateException("No documents found");
        int count = 0;
        Bits liveDocs = MultiFields.getLiveDocs(reader);

        for(int i = 0; i < reader.maxDoc(); i++) {
            if (liveDocs != null && !liveDocs.get(i))
                continue;

            if(count > docIds.length) {
                int[] newCopy = new int[docIds.length * 2];
                System.arraycopy(docIds,0,newCopy,0,docIds.length);
                docIds = newCopy;
                log.info("Reallocating doc ids");
            }

            docIds[count++] = i;
        }

        try {
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return docIds;
    }

    @Override
    public void addWordToDoc(int doc, T word) {
        Field f = new TextField(WORD_FIELD,word.getLabel(),Field.Store.YES);
        try {
            IndexSearcher searcher = searcherManager.acquire();
            Document doc2 = searcher.doc(doc);
            if(doc2 != null)
                doc2.add(f);
            else {
                Document d = new Document();
                d.add(f);
            }

            searcherManager.release(searcher);


        } catch (IOException e) {
            e.printStackTrace();
        }


    }


    private  void initReader() {
        if(reader == null) {
            try {
                ensureDirExists();
                if(getWriter() == null) {
                    this.indexWriter = null;
                    while(getWriter() == null) {
                        log.warn("Writer was null...reinitializing");
                        Thread.sleep(1000);
                    }
                }
                IndexWriter writer = getWriter().getIndexWriter();
                if(writer == null)
                    throw new IllegalStateException("index writer was null");
                searcherManager = new SearcherManager(writer,true,new SearcherFactory());
                writer.commit();
                readerManager = new ReaderManager(dir);
                DirectoryReader reader = readerManager.acquire();
                numDocs = readerManager.acquire().numDocs();
                readerManager.release(reader);
            }catch (IOException | InterruptedException e) {
                throw new RuntimeException(e);
            }

        }
    }

    @Override
    public void addWordsToDoc(int doc,final List<T> words) {


        Document d = new Document();

        for (T word : words)
            d.add(new TextField(WORD_FIELD, word.getLabel(), Field.Store.YES));



        totalWords.set(totalWords.get() + words.size());
        addWords(words);
        try {
            getWriter().addDocument(d);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    @Override
    public Pair<List<T>, String> documentWithLabel(int index) {
        List<T> ret = new CopyOnWriteArrayList<>();
        String label = "NONE";
        try {
            DirectoryReader reader = readerManager.acquire();
            Document doc = reader.document(index);
            reader.close();
            String[] values = doc.getValues(WORD_FIELD);
            label = doc.get(LABEL);
            if(label == null)
                label = "NONE";
            for(String s : values) {
                T tok = vocabCache.wordFor(s);
                if (tok != null)
                    ret.add(tok);
            }



        }


        catch (Exception e) {
            e.printStackTrace();
        }
        return new Pair<>(ret,label);


    }

    @Override
    public Pair<List<T>, Collection<String>> documentWithLabels(int index) {
        List<T> ret = new CopyOnWriteArrayList<>();
        Collection<String> labels = new ArrayList<>();

        try {
            DirectoryReader reader = readerManager.acquire();
            Document doc = reader.document(index);
            readerManager.release(reader);
            String[] values = doc.getValues(WORD_FIELD);
            String[] labels2 = doc.getValues(LABEL);

            for(String s : values) {
                T tok = vocabCache.wordFor(s);
                if (tok != null)
                    ret.add(tok);
            }

            Collections.addAll(labels, labels2);

        }


        catch (Exception e) {
            e.printStackTrace();
        }
        return new Pair<>(ret,labels);


    }

    @Override
    public void addLabelForDoc(int doc, T word) {
        addLabelForDoc(doc,word.getLabel());
    }

    @Override
    public void addLabelForDoc(int doc, String label) {
        try {
            DirectoryReader reader = readerManager.acquire();
            Document doc2 = reader.document(doc);
            doc2.add(new TextField(LABEL,label,Field.Store.YES));
            readerManager.release(reader);
            TrackingIndexWriter writer = getWriter();
            Term term = new Term(LABEL,label);
            writer.updateDocument(term,doc2);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void addWordsToDoc(int doc, List<T> words, String label) {
        Document d = new Document();

        for (T word : words)
            d.add(new TextField(WORD_FIELD, word.getLabel(), Field.Store.YES));

        d.add(new TextField(LABEL,label,Field.Store.YES));

        totalWords.set(totalWords.get() + words.size());
        addWords(words);

        try {
            getWriter().addDocument(d);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void addWordsToDoc(int doc, List<T> words, T label) {
        addWordsToDoc(doc,words,label.getLabel());
    }

    @Override
    public void addLabelsForDoc(int doc, List<T> label) {
        try {
            DirectoryReader reader = readerManager.acquire();

            Document doc2 = reader.document(doc);
            for(T s : label)
                doc2.add(new TextField(LABEL,s.getLabel(),Field.Store.YES));
            readerManager.release(reader);
            TrackingIndexWriter writer = getWriter();
            List<Term> terms = new ArrayList<>();
            for(T s : label) {
                Term term = new Term(LABEL,s.getLabel());
                terms.add(term);
            }
            writer.addDocument(doc2);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void addLabelsForDoc(int doc, Collection<String> label) {
        try {
            DirectoryReader reader = readerManager.acquire();

            Document doc2 = reader.document(doc);
            for(String s : label)
                doc2.add(new TextField(LABEL,s,Field.Store.YES));
            readerManager.release(reader);
            TrackingIndexWriter writer = getWriter();
            List<Term> terms = new ArrayList<>();
            for(String s : label) {
                Term term = new Term(LABEL,s);
                terms.add(term);
            }
            writer.addDocument(doc2);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void addWordsToDoc(int doc, List<T> words, Collection<String> label) {
        Document d = new Document();

        for (T word : words)
            d.add(new TextField(WORD_FIELD, word.getLabel(), Field.Store.YES));

        for(String s : label)
            d.add(new TextField(LABEL,s,Field.Store.YES));

        totalWords.set(totalWords.get() + words.size());
        addWords(words);
        try {
            getWriter().addDocument(d);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void addWordsToDocVocabWord(int doc, List<T> words, Collection<T> label) {
        Document d = new Document();

        for (T word : words)
            d.add(new TextField(WORD_FIELD, word.getLabel(), Field.Store.YES));

        for(T s : label)
            d.add(new TextField(LABEL,s.getLabel(),Field.Store.YES));

        totalWords.set(totalWords.get() + words.size());
        addWords(words);
        try {
            getWriter().addDocument(d);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private void addWords(List<T> words) {
        if(!miniBatch)
            return;
        for (T word : words) {
            // The subsampling randomly discards frequent words while keeping the ranking same
            if (sample > 0) {
                double ran = (Math.sqrt(word.getElementFrequency() / (sample * numDocuments())) + 1)
                        * (sample * numDocuments()) / word.getElementFrequency();

                if (ran < (nextRandom.get() & 0xFFFF) / (double) 65536) {
                    continue;
                }

                currMiniBatch.add(word);
            } else {
                currMiniBatch.add(word);
                if (currMiniBatch.size() >= batchSize) {
                    miniBatches.add(new ArrayList<>(currMiniBatch));
                    currMiniBatch.clear();
                }
            }

        }
    }




    private void ensureDirExists() throws IOException {
        if (dir == null) {
            File directory = new File(indexPath);
            log.info("Creating directory " + directory.getCanonicalPath());
            FileUtils.deleteDirectory(directory);
            dir = FSDirectory.open(Paths.get(directory.toURI()));
            if (!directory.exists()) {
                directory.mkdir();
            }
        }
    }



    private synchronized TrackingIndexWriter getWriterWithRetry() {
        if(this.indexWriter != null)
            return this.indexWriter;

        IndexWriterConfig iwc;
        IndexWriter writer;

        try {
            if(analyzer == null)
                analyzer  = new StandardAnalyzer(new InputStreamReader(new ByteArrayInputStream("".getBytes())));


            ensureDirExists();


            if(this.indexWriter == null) {
                indexBeingCreated.set(true);



                iwc = new IndexWriterConfig(analyzer);
                iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
                iwc.setWriteLockTimeout(1000);

                log.info("Creating new index writer");
                while((writer = tryCreateWriter(iwc)) == null) {
                    log.warn("Failed to create writer...trying again");
                    iwc = new IndexWriterConfig(analyzer);
                    iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
                    iwc.setWriteLockTimeout(1000);

                    Thread.sleep(10000);
                }
                this.indexWriter = new TrackingIndexWriter(writer);

            }

        }

        catch(Exception e) {
            throw new IllegalStateException(e);
        }




        return this.indexWriter;
    }


    private IndexWriter tryCreateWriter(IndexWriterConfig iwc) {

        try {
            dir.close();
            dir = null;
            FileUtils.deleteDirectory(new File(indexPath));

            ensureDirExists();
            if(lockFactory == null)
                lockFactory = NativeFSLockFactory.getDefault();
            return new IndexWriter(dir,iwc);
        }
        catch (Exception e) {
            String id = UUID.randomUUID().toString();
            indexPath = id;
            log.warn("Setting index path to " + id);
            log.warn("Couldn't create index ",e);
            return null;
        }

    }

    private synchronized  TrackingIndexWriter getWriter() {
        int attempts = 0;
        while(getWriterWithRetry() == null && attempts < 3) {

            try {
                Thread.sleep(1000 * attempts);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            if(attempts >= 3)
                throw new IllegalStateException("Can't obtain write lock");
            attempts++;

        }

        return this.indexWriter;
    }



    @Override
    public void finish() {
        try {
            initReader();
            DirectoryReader reader = readerManager.acquire();
            numDocs = reader.numDocs();
            readerManager.release(reader);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public long totalWords() {
        return totalWords.get();
    }

    @Override
    public int batchSize() {
        return batchSize;
    }

    @Override
    public void eachDocWithLabels(final Function<Pair<List<T>, Collection<String>>, Void> func, ExecutorService exec) {
        int[] docIds = allDocs();
        for(int i : docIds) {
            final int j = i;
            exec.execute(new Runnable() {
                @Override
                public void run() {
                    func.apply(documentWithLabels(j));
                }
            });
        }
    }

    @Override
    public void eachDocWithLabel(final Function<Pair<List<T>, String>, Void> func, ExecutorService exec) {
        int[] docIds = allDocs();
        for(int i : docIds) {
            final int j = i;
            exec.execute(new Runnable() {
                @Override
                public void run() {
                    func.apply(documentWithLabel(j));
                }
            });
        }
    }

    @Override
    public void eachDoc(final Function<List<T>, Void> func,ExecutorService exec) {
        int[] docIds = allDocs();
        for(int i : docIds) {
            final int j = i;
            exec.execute(new Runnable() {
                @Override
                public void run() {
                    func.apply(document(j));
                }
            });
        }
    }







    @Override
    public void onClose(IndexReader reader) {
        readerClosed.set(true);
    }

    @Override
    public boolean hasNext() {
        if(!miniBatch)
            throw new IllegalStateException("Mini batch mode turned off");
        return !miniBatchDocs.isEmpty() ||
                miniBatchGoing.get();
    }

    @Override
    public List<T> next() {
        if(!miniBatch)
            throw new IllegalStateException("Mini batch mode turned off");
        if(!miniBatches.isEmpty())
            return miniBatches.remove(0);
        else if(miniBatchGoing.get()) {
            while(miniBatches.isEmpty()) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }

                log.warn("Waiting on more data...");

                if(!miniBatches.isEmpty())
                    return miniBatches.remove(0);
            }
        }
        return null;
    }




    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }



    public class BatchDocIter implements Iterator<List<List<T>>> {

        private int batchSize = 1000;
        private Iterator<List<T>> docIter = new DocIter();

        public BatchDocIter(int batchSize) {
            this.batchSize = batchSize;
        }

        @Override
        public boolean hasNext() {
            return docIter.hasNext();
        }

        @Override
        public List<List<T>> next() {
            List<List<T>> ret = new ArrayList<>();
            for(int i = 0; i < batchSize; i++) {
                if(!docIter.hasNext())
                    break;
                ret.add(docIter.next());
            }
            return ret;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }


    public class DocIter implements Iterator<List<T>> {
        private int currIndex = 0;
        private int[] docs = allDocs();


        @Override
        public boolean hasNext() {
            return currIndex < docs.length;
        }

        @Override
        public List<T> next() {
            return document(docs[currIndex++]);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }


    public static class Builder<T extends SequenceElement> {
        private File indexDir;
        private  Directory dir;
        private IndexReader reader;
        private   Analyzer analyzer;
        private IndexSearcher searcher;
        private IndexWriter writer;
        private  IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
        private VocabCache<T> vocabCache;
        private List<String> stopWords = StopWords.getStopWords();
        private boolean cache = false;
        private int batchSize = 1000;
        private double sample = 0;
        private boolean miniBatch = false;



        public Builder<T> miniBatch(boolean miniBatch) {
            this.miniBatch = miniBatch;
            return this;
        }
        public Builder<T> cacheInRam(boolean cache) {
            this.cache = cache;
            return this;
        }

        public Builder<T> sample(double sample) {
            this.sample = sample;
            return this;
        }

        public Builder<T> batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }



        public Builder<T> indexDir(File indexDir) {
            this.indexDir = indexDir;
            return this;
        }

        public Builder<T> cache(@NonNull VocabCache<T> cache) {
            this.vocabCache = cache;
            return this;
        }

        public Builder<T> stopWords(@NonNull List<String> stopWords) {
            this.stopWords = stopWords;
            return this;
        }

        public Builder<T> dir(Directory dir) {
            this.dir = dir;
            return this;
        }

        public Builder<T> reader(IndexReader reader) {
            this.reader = reader;
            return this;
        }

        public Builder<T> writer(IndexWriter writer) {
            this.writer = writer;
            return this;
        }

        public Builder<T> analyzer(Analyzer analyzer) {
            this.analyzer = analyzer;
            return this;
        }

        public InvertedIndex<T> build() {
            LuceneInvertedIndex<T> ret;
            if(indexDir != null) {
                ret = new LuceneInvertedIndex<T>(vocabCache,cache,indexDir.getAbsolutePath());
            }
            else
                ret = new LuceneInvertedIndex<T>(vocabCache);
            try {
                ret.batchSize = batchSize;

                if(dir != null)
                    ret.dir = dir;
                ret.miniBatch = miniBatch;
                if(reader != null)
                    ret.reader = reader;
                if(analyzer != null)
                    ret.analyzer = analyzer;
            }catch(Exception e) {
                throw new RuntimeException(e);
            }

            return ret;

        }

    }


}
