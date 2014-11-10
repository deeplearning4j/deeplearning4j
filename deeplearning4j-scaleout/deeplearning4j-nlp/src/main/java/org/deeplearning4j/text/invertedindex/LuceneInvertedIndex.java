package org.deeplearning4j.text.invertedindex;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.LockObtainFailedException;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.Version;
import org.deeplearning4j.berkeley.StringUtils;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.stopwords.StopWords;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Lucene based inverted index
 *
 * @author Adam Gibson
 */
public class LuceneInvertedIndex implements InvertedIndex,IndexReader.ReaderClosedListener,Iterator<List<VocabWord>> {

    private transient  Directory dir;
    private transient IndexReader reader;
    private   transient Analyzer analyzer;
    private transient IndexSearcher searcher;
    private  transient IndexWriter writer;
    private VocabCache vocabCache;
    public final static String WORD_FIELD = "word";
    private int numDocs = 0;
    private List<List<VocabWord>> words = Collections.synchronizedList(new ArrayList<List<VocabWord>>());
    private boolean cache = true;
    private transient ExecutorService indexManager;
    private AtomicBoolean indexBeingCreated = new AtomicBoolean(false);
    private static Logger log = LoggerFactory.getLogger(LuceneInvertedIndex.class);
    public final static String INDEX_PATH = "word2vec-index";
    private AtomicBoolean readerClosed = new AtomicBoolean(false);
    private AtomicInteger totalWords = new AtomicInteger(0);
    private int batchSize = 1000;
    private List<List<VocabWord>> miniBatches = new CopyOnWriteArrayList<>();
    private double sample = 0;
    private AtomicLong nextRandom = new AtomicLong(5);
    private String indexPath = INDEX_PATH;
    private ScheduledExecutorService miniBatchManager = Executors.newScheduledThreadPool(1);
    private LinkedBlockingDeque<List<VocabWord>> miniBatchDocs = new LinkedBlockingDeque<>();
    private AtomicBoolean miniBatchGoing = new AtomicBoolean(true);

    public LuceneInvertedIndex(VocabCache vocabCache,boolean cache) {
        this.vocabCache = vocabCache;
        this.cache = cache;
        indexManager = Executors.newFixedThreadPool(1);
        startMiniBatches();
    }


    public LuceneInvertedIndex(VocabCache vocabCache,boolean cache,String indexPath) {
        this.vocabCache = vocabCache;
        this.cache = cache;
        this.indexPath = indexPath;
        indexManager = Executors.newFixedThreadPool(1);
        startMiniBatches();
    }


    private LuceneInvertedIndex(){
        indexManager = Executors.newFixedThreadPool(1);
        startMiniBatches();
    }

    @Override
    public double sample() {
        return sample;
    }

    @Override
    public Iterator<List<VocabWord>> miniBatches() {
        return this;
    }

    @Override
    public List<VocabWord> document(int index) {
        if(cache) {
            List<VocabWord> ret = words.get(index);
            List<VocabWord> ret2 = new ArrayList<>();
            //remove all non vocab words
            for(VocabWord word : ret) {
                if(vocabCache.containsWord(word.getWord()))
                    ret2.add(word);
            }


            return ret2;
        }


        List<VocabWord> ret = new CopyOnWriteArrayList<>();
        try {
            initReader();

            Document doc = reader.document(index);

            String[] values = doc.getValues(WORD_FIELD);
            for(String s : values) {
                ret.add(vocabCache.wordFor(s));
            }


        }

        catch(AlreadyClosedException e1) {
            reader = null;
            readerClosed.set(false);
            return document(index);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return ret;
    }

    @Override
    public List<Integer> documents(VocabWord vocabWord) {
        try {
            TermQuery query = new TermQuery(new Term(WORD_FIELD,vocabWord.getWord()));


            TopDocs topdocs = searcher.search(query,Integer.MAX_VALUE);
            List<Integer> ret = new ArrayList<>();
            for(int i = 0; i < topdocs.totalHits; i++) {
                ret.add(topdocs.scoreDocs[i].doc);
            }



            return ret;
        }
        catch(AlreadyClosedException e) {
            initReader();
            return documents(vocabWord);
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public int numDocuments() {
        if(numDocs > 0)
            return numDocs;

        int ret;
        try {

            initReader();
            ret = reader.numDocs();

        }catch(Exception e) {
            return 0;
        }
        return ret;
    }

    @Override
    public Collection<Integer> allDocs() {
        if(cache){
            List<Integer> ret = new ArrayList<>();
            for(int i = 0; i < words.size(); i++)
                ret.add(i);
            return ret;
        }



        List<Integer> docIds = new ArrayList<>();
        Bits liveDocs = MultiFields.getLiveDocs(reader);
        for(int i = 0; i < reader.maxDoc() + 1; i++) {
            if (liveDocs != null && !liveDocs.get(i))
                continue;

            docIds.add(i);
        }
        return docIds;
    }

    @Override
    public void addWordToDoc(int doc, VocabWord word) {
        Field f = new TextField(WORD_FIELD,word.getWord(),Field.Store.YES);
        try {
            initReader();

            Document doc2 = searcher.doc(doc);
            if(doc2 != null)
                doc2.add(f);
            else {
                Document d = new Document();
                d.add(f);
            }



        } catch (IOException e) {
            e.printStackTrace();
        }

        initReader();

    }


    private void initReader() {
        if(reader == null) {
            try {
                writer.commit();
                reader = DirectoryReader.open(dir);
                searcher = new IndexSearcher(reader);
            }catch(Exception e) {
                throw new RuntimeException(e);
            }

        }

        else if(readerClosed.get()) {
            try {
                reader = DirectoryReader.open(dir);
                searcher = new IndexSearcher(reader);
                readerClosed.set(false);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }



    }

    @Override
    public void addWordsToDoc(int doc,final List<VocabWord> words) {
        if (cache) {
            this.words.add(words);
            indexManager.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        if (writer == null) {
                            if(!indexBeingCreated.get())
                                createWriter();
                            else {
                                waitOnWriter();
                            }
                        }
                        Document d = new Document();

                        for (VocabWord word : words) {
                            d.add(new TextField(WORD_FIELD, word.getWord(), Field.Store.YES));
                        }

                        writer.addDocument(d, analyzer);

                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }

                    initReader();
                }
            });


        } else {

            Document d = new Document();

            for (VocabWord word : words) {
                d.add(new TextField(WORD_FIELD, word.getWord(), Field.Store.YES));
            }
            try {

                while(writer == null) {
                    if(!indexBeingCreated.get() || writer == null)
                        createWriter();
                    else {
                        waitOnWriter();
                    }
                }
                writer.addDocument(d, analyzer);

            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            initReader();
        }

        totalWords.set(totalWords.get() + words.size());
        miniBatchDocs.add(words);

    }


    private void startMiniBatches() {
        miniBatchManager.schedule(new Runnable() {
            @Override
            public void run() {
                List<VocabWord> currMiniBatch = new ArrayList<>();
                while(miniBatchGoing.get()) {
                    List<VocabWord> words = miniBatchDocs.poll();
                    if(words == null || words.isEmpty()) {
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                        continue;
                    }
                    for (VocabWord word : words) {
                        // The subsampling randomly discards frequent words while keeping the ranking same
                        if (sample > 0) {
                            double ran = (Math.sqrt(word.getWordFrequency() / (sample * numDocuments())) + 1)
                                    * (sample * numDocuments()) / word.getWordFrequency();

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

                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }

                }


            }
        }, 1, TimeUnit.SECONDS);
    }


    private void ensureDirExists() throws Exception {
        if(dir == null) {
            dir = FSDirectory.open(new File(indexPath));
            File dir2 = new File(indexPath);
            if (!dir2.exists())
                dir2.mkdir();
        }


    }

    private void createWriter() {
        try {
            ensureDirExists();
            if(analyzer == null)
                analyzer  = new StandardAnalyzer(new InputStreamReader(new ByteArrayInputStream("".getBytes())));
            IndexWriterConfig iwc = new IndexWriterConfig(Version.LATEST, analyzer);

            if(!indexBeingCreated.get() && writer == null) {
                indexBeingCreated.set(true);
                writer = new IndexWriter(dir, iwc);
            }
            else if(!indexBeingCreated.get()) {
                indexBeingCreated.set(true);
                writer = new IndexWriter(dir,iwc);

            }

        }catch(LockObtainFailedException e) {
            try {
                IndexWriter.unlock(dir);
            } catch (IOException e1) {
                throw new RuntimeException("Unable to unlock directory " + indexPath);
            }
            try {
                IndexWriterConfig iwc = new IndexWriterConfig(Version.LATEST, analyzer);
                writer = new IndexWriter(dir,iwc);
            } catch (IOException e1) {
                log.warn("Failed to created writer...trying again");
                createWriter();
            }

        }
        catch(Exception e) {
            throw new IllegalStateException("Failed to created writer",e);
        }
    }


    private void waitOnWriter() {
        while(writer == null) {
            try {
                Thread.sleep(1000);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    @Override
    public void finish() {
        if(cache) {
            indexManager.execute(new Runnable() {
                public void run() {
                    try {
                        IndexWriterConfig iwc = new IndexWriterConfig(Version.LATEST, analyzer);

                        if (dir == null)
                            dir = FSDirectory.open(new File(INDEX_PATH));
                        if (writer == null)
                            if(!indexBeingCreated.get())
                                writer = new IndexWriter(dir, iwc);
                            else
                                waitOnWriter();
                        if(IndexWriter.isLocked(dir))
                            IndexWriter.unlock(dir);
                        writer.forceMerge(1);
                        writer.commit();

                        initReader();
                        numDocs = reader.numDocs();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }


                }
            });


        }

        else {
            try {
                log.info("Committing index...");
                writer.forceMerge(1);
                writer.commit();
                log.info("Finished committing changes");

            } catch (IOException e) {
                e.printStackTrace();
            }

            initReader();
            numDocs = reader.numDocs();

            try {
                reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

        indexManager.shutdown();
        miniBatchManager.shutdown();
        try {
            indexManager.awaitTermination(1,TimeUnit.DAYS);
            miniBatchGoing.set(false);
            miniBatchManager.awaitTermination(1,TimeUnit.MINUTES);

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }


    }

    @Override
    public int totalWords() {
        return totalWords.get();
    }

    @Override
    public int batchSize() {
        return batchSize;
    }



    @Override
    public void onClose(IndexReader reader) {
        readerClosed.set(true);
    }

    @Override
    public boolean hasNext() {
        return !miniBatchDocs.isEmpty() ||
                miniBatchGoing.get() ||
                !miniBatchManager.isShutdown();
    }

    @Override
    public List<VocabWord> next() {
        if(!miniBatchDocs.isEmpty())
            return miniBatchDocs.poll();
        else if(!miniBatchManager.isShutdown()) {
            while(miniBatchDocs.isEmpty()) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }

                log.warn("Waiting on more data...");

                if(miniBatchManager.isShutdown())
                    return miniBatchDocs.poll();
            }
        }
        return null;
    }


    public static class Builder {
        private File indexDir;
        private  Directory dir;
        private IndexReader reader;
        private   Analyzer analyzer;
        private IndexSearcher searcher;
        private IndexWriter writer;
        private  IndexWriterConfig iwc = new IndexWriterConfig(Version.LUCENE_4_10_0, analyzer);
        private VocabCache vocabCache;
        private List<String> stopWords = StopWords.getStopWords();
        private boolean cache = true;
        private int batchSize = 1000;
        private double sample = 0;


        public Builder cacheInRam(boolean cache) {
            this.cache = cache;
            return this;
        }

        public Builder sample(double sample) {
            this.sample = sample;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }



        public Builder indexDir(File indexDir) {
            this.indexDir = indexDir;
            return this;
        }

        public Builder cache(VocabCache cache) {
            this.vocabCache = cache;
            return this;
        }

        public Builder stopWords(List<String> stopWords) {
            this.stopWords = stopWords;
            return this;
        }

        public Builder dir(Directory dir) {
            this.dir = dir;
            return this;
        }

        public Builder reader(IndexReader reader) {
            this.reader = reader;
            return this;
        }

        public Builder writer(IndexWriter writer) {
            this.writer = writer;
            return this;
        }

        public Builder analyzer(Analyzer analyzer) {
            this.analyzer = analyzer;
            return this;
        }

        public InvertedIndex build() {
            LuceneInvertedIndex ret = new LuceneInvertedIndex();
            try {
                if(analyzer == null)
                    analyzer  = new StandardAnalyzer(new InputStreamReader(new ByteArrayInputStream(StringUtils.join(stopWords,"\n").getBytes())));
                if(indexDir != null && dir != null)
                    throw new IllegalStateException("Please define only a directory or a file directory");
                if(iwc == null)
                    iwc = new IndexWriterConfig(Version.LATEST, analyzer);

                if(indexDir != null && !cache) {
                    if(!indexDir.exists())
                        indexDir.mkdirs();
                    dir = FSDirectory.open(indexDir);
                    if(writer == null)
                        writer = new IndexWriter(dir, iwc);

                }

                if(vocabCache == null)
                    throw new IllegalStateException("Vocab cache must not be null");


                ret.batchSize = batchSize;
                ret.vocabCache = vocabCache;
                ret.dir = dir;
                ret.writer = writer;
                ret.cache = cache;

                ret.reader = reader;
                ret.searcher = searcher;
                ret.analyzer = analyzer;
                ret.vocabCache = vocabCache;

            }catch(Exception e) {
                throw new RuntimeException(e);
            }

            return ret;

        }

    }


}
