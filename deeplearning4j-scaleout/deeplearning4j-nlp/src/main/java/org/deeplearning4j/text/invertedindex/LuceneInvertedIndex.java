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

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Lucene based inverted index
 *
 * @author Adam Gibson
 */
public class LuceneInvertedIndex implements InvertedIndex {
    private  Directory dir;
    private IndexReader reader;
    private   Analyzer analyzer;
    private IndexSearcher searcher;
    private IndexWriter writer;
    private  IndexWriterConfig iwc = new IndexWriterConfig(Version.LATEST, analyzer);
    private VocabCache vocabCache;
    public final static String WORD_FIELD = "word";
    private int numDocs = 0;
    public LuceneInvertedIndex(VocabCache vocabCache) {
        try {
            index("word2vec-index",true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        this.vocabCache = vocabCache;
    }


    private LuceneInvertedIndex(){}

    @Override
    public List<VocabWord> document(int index) {
        List<VocabWord> ret = new ArrayList<>();
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

            ret = reader.numDocs();

        }catch(Exception e) {
            return 0;
        }
        return ret;
    }

    @Override
    public Collection<Integer> allDocs() {
        List<Integer> docIds = new ArrayList<>();
        for(int i = 0; i < reader.maxDoc(); i++)
            docIds.add(i);
        return docIds;
    }

    @Override
    public void addWordToDoc(int doc, VocabWord word) {
        Field f = new TextField(WORD_FIELD,word.getWord(),Field.Store.YES);
        try {

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

    }

    @Override
    public void addWordsToDoc(int doc, List<VocabWord> words) {
        Document d = new Document();

        for(VocabWord word : words) {
            d.add(new TextField(WORD_FIELD, word.getWord() ,Field.Store.YES));
        }
        try {
            writer.addDocument(d,analyzer);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }




    }

    @Override
    public void finish() {
        try {

            writer.forceMerge(1);
            writer.commit();
        } catch (IOException e) {
            e.printStackTrace();
        }
        if(reader == null) {
            try {
                reader = DirectoryReader.open(dir);
                searcher = new IndexSearcher(reader);
            }catch(Exception e) {
                throw new RuntimeException(e);
            }

        }
        numDocs = reader.numDocs();

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
        if(reader == null) {
            try {
                reader = DirectoryReader.open(dir);
                searcher = new IndexSearcher(reader);
            }catch(Exception e) {
                throw new RuntimeException(e);
            }

        }



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
                if(indexDir != null && dir == null)
                    throw new IllegalStateException("Please define only a directory or a file directory");
                if(iwc == null)
                    iwc = new IndexWriterConfig(Version.LATEST, analyzer);

                if(indexDir != null) {
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
                if(reader == null) {
                    try {
                        reader = DirectoryReader.open(dir);
                        searcher = new IndexSearcher(reader);
                    }catch(Exception e) {
                        throw new RuntimeException(e);
                    }

                }

                ret.reader = reader;
                ret.searcher = searcher;
                ret.analyzer = analyzer;

            }catch(Exception e) {

            }

            return ret;

        }

    }


}
