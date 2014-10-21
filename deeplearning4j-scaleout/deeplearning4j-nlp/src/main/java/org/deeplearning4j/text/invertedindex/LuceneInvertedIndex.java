package org.deeplearning4j.text.invertedindex;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.*;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;
import org.deeplearning4j.models.word2vec.VocabWord;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Created by agibsonccc on 10/21/14.
 */
public class LuceneInvertedIndex implements InvertedIndex {
    private  Directory dir;
    private IndexReader reader;
    private   Analyzer analyzer = new StandardAnalyzer();
    private IndexSearcher searcher = new IndexSearcher(reader);
    private IndexWriter writer;
    private FieldType wordType;

    public LuceneInvertedIndex() {
        try {
            index("word2vec-index",true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public List<VocabWord> document(int index) {
        List<VocabWord> ret = new ArrayList<>();
        try {
            Document doc = searcher.doc(index);
            Iterator<IndexableField> iter = doc.iterator();
            //Terms terms = SlowCompositeReaderWrapper.wrap(reader).terms("field");
            while(iter.hasNext()) {
                IndexableField curr = iter.next();
                VocabWord word = new VocabWord(1.0,curr.stringValue());

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public List<Integer> documents(VocabWord vocabWord) {
        return null;
    }

    @Override
    public int numDocuments() {
        return 0;
    }

    @Override
    public Collection<Integer> allDocs() {
        return null;
    }

    @Override
    public void addWordToDoc(int doc, VocabWord word) {
        Field f = new Field("word",word.getWord(),wordType);
        try {
            Document doc2 = searcher.doc(doc);
            doc2.add(f);

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    @Override
    public void addWordsToDoc(int doc, List<VocabWord> words) {

    }


    private void index(String indexPath,boolean create) throws IOException {
        dir = FSDirectory.open(new File(indexPath));
        reader = DirectoryReader.open(FSDirectory.open(new File("word2vec-index")));
        searcher = new IndexSearcher(reader);


        wordType = new FieldType();
        wordType.setStoreTermVectors(true);
        wordType.setStoreTermVectorPositions(true);
        wordType.setIndexed(true);

        IndexWriterConfig iwc = new IndexWriterConfig(Version.LUCENE_4_10_0, analyzer);

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
        // iwc.setRAMBufferSizeMB(256.0);

        writer = new IndexWriter(dir, iwc);
        //indexDocs(writer, docDir);

        // NOTE: if you want to maximize search performance,
        // you can optionally call forceMerge here.  This can be
        // a terribly costly operation, so generally it's only
        // worth it when your index is relatively static (ie
        // you're done adding documents to it):
        //
        // writer.forceMerge(1);

        writer.close();

    }

}
