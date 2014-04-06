package org.deeplearning4j.datasets.loader;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.ArchiveUtils;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.word2vec.vectorizer.BagOfWordsVectorizer;
import org.deeplearning4j.word2vec.vectorizer.TextVectorizer;
import org.deeplearning4j.word2vec.vectorizer.TfidfVectorizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class ReutersNewsGroupsLoader extends BaseDataFetcher {

    private TextVectorizer textVectorizer;
    private boolean tfidf;
    public final static String NEWSGROUP_URL = "http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz";
    private File reutersRootDir;
    private static Logger log = LoggerFactory.getLogger(ReutersNewsGroupsLoader.class);
    private DataSet load;


    public ReutersNewsGroupsLoader(boolean tfidf) throws Exception {
        getIfNotExists();
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(reutersRootDir);
        List<String> labels = Arrays.asList("label1", "label2");
        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();

        if(tfidf)
            this.textVectorizer = new TfidfVectorizer(iter,tokenizerFactory,labels);

        else
            this.textVectorizer = new BagOfWordsVectorizer(iter,tokenizerFactory,labels);

        load = this.textVectorizer.vectorize();
    }

    private void getIfNotExists() throws Exception {
        String home = System.getProperty("user.home");
        String rootDir = home + File.separator + "reuters";
        reutersRootDir = new File(rootDir);
        if(!reutersRootDir.exists())
            reutersRootDir.mkdir();



        File rootTarFile = new File(reutersRootDir,"20news-18828.tar.gz");
        if(rootTarFile.exists())
            rootTarFile.delete();
        rootTarFile.createNewFile();

        FileUtils.copyURLToFile(new URL(NEWSGROUP_URL), rootTarFile);
        ArchiveUtils.unzipFileTo(rootTarFile.getAbsolutePath(), reutersRootDir.getAbsolutePath());
        rootTarFile.delete();
        FileUtils.copyDirectory(new File(reutersRootDir,"20news-18828"),reutersRootDir);
        FileUtils.deleteDirectory(new File(reutersRootDir,"20news-18828"));
        if(reutersRootDir.listFiles() == null)
            throw new IllegalStateException("No files found!");

    }


    /**
     * Fetches the next dataset. You need to call this
     * to get a new dataset, otherwise {@link #next()}
     * just returns the last data set fetch
     *
     * @param numExamples the number of examples to fetch
     */
    @Override
    public void fetch(int numExamples) {
        List<DataSet> newData = new ArrayList<>();
        for(int grabbed = 0; grabbed < numExamples && cursor < load.numExamples(); cursor++,grabbed++) {
            newData.add(load.get(cursor));
        }

        this.curr = DataSet.merge(newData);

    }
}
