package org.deeplearning4j.text.invertedindex;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;
import org.deeplearning4j.berkeley.StringUtils;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.stopwords.StopWords;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Lucene based inverted index
 *
 * @author Adam Gibson
 */
public class LuceneInvertedIndex implements InvertedIndex {
    private transient  Directory dir;
    private transient IndexReader reader;
    private   transient Analyzer analyzer;
    private transient IndexSearcher searcher;
    private  transient IndexWriter writer;
    private  transient IndexWriterConfig iwc = new IndexWriterConfig(Version.LATEST, analyzer);
    private VocabCache vocabCache;
    public final static String WORD_FIELD = "word";
    private int numDocs = 0;
    private List<List<VocabWord>> words = new CopyOnWriteArrayList<>();
    private boolean cache = true;
    private transient ExecutorService indexManager;
    public LuceneInvertedIndex(VocabCache vocabCache,boolean cache) {
        try {
            index("word2vec-index",cache);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        this.vocabCache = vocabCache;
        this.cache = cache;
        indexManager = Executors.newFixedThreadPool(1);
    }


    private LuceneInvertedIndex(){
        indexManager = Executors.newFixedThreadPool(1);
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
            Document doc = reader.document(index);

            String[] values = doc.getValues(WORD_FIELD);
            for(String s : values) {
                ret.add(vocabCache.wordFor(s));
            }


        } catch (Exception e) {
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
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public int numDocuments() {
        if(numDocs > 0)
            return numDocs;

        int ret = 0;
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
        for(int i = 0; i < reader.maxDoc(); i++)
            docIds.add(i);
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

    }

    @Override
    public void addWordsToDoc(int doc,final List<VocabWord> words) {
        if (cache) {
            this.words.add(words);
            indexManager.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        if(writer == null) {
                            index("word2vec-path", true);
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
                writer.addDocument(d, analyzer);

            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            initReader();
        }
    }



    @Override
    public void finish() {
        if(cache) {
            indexManager.execute(new Runnable(){
                public void run() {
                    try {
                        if(dir == null)
                            dir = FSDirectory.open(new File("word2vec-index"));
                        if(writer == null)
                            writer = new IndexWriter(dir, iwc);
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

                writer.forceMerge(1);
                writer.commit();
            } catch (IOException e) {
                e.printStackTrace();
            }

            initReader();
            numDocs = reader.numDocs();

        }

    }


    private void index(String indexPath,boolean create) throws IOException {
        File dir2 = new File(indexPath);
        if(!dir2.exists())
            dir2.mkdir();

        analyzer  = new StandardAnalyzer(new InputStreamReader(new ByteArrayInputStream("".getBytes())));

        if (create) {
            // Create a new index in the directory, removing any
            // previously indexed documents:
            iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        } else {
            // Add new documents to an existing index:
            iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);
        }

        // Optional: for better indexing performance, if you
        // are indexing many documents, increase the RAM
        // buffer.  But if you do this, increase the max heap
        // size to the JVM (eg add -Xmx512m or -Xmx1g):
        //
        iwc.setRAMBufferSizeMB(5000);
        dir = FSDirectory.open(dir2);

        writer = new IndexWriter(dir, iwc);
        //indexDocs(writer, docDir);

        // NOTE: if you want to maximize search performance,
        // you can optionally call forceMerge here.  This can be
        // a terribly costly operation, so generally it's only
        // worth it when your index is relatively static (ie
        // you're done adding documents to it):
        //
        // writer.forceMerge(1);
        initReader();



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


        public Builder cacheDocsInMemory(boolean cache) {
            this.cache = cache;
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
