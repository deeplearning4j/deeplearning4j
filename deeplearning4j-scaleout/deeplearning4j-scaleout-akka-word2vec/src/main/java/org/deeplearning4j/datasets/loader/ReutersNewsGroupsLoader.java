package org.deeplearning4j.datasets.loader;

import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.word2vec.vectorizer.BagOfWordsVectorizer;
import org.deeplearning4j.word2vec.vectorizer.TextVectorizer;
import org.deeplearning4j.word2vec.vectorizer.TfidfVectorizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
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


    public ReutersNewsGroupsLoader(boolean tfidf) throws Exception {
        String home = System.getProperty("user.home");
        getIfNotExists();
        String rootDir = home + File.separator + "reuters";
        reutersRootDir = new File(rootDir);
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(reutersRootDir);
        List<String> labels = Arrays.asList("label1", "label2");
        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();


        if(tfidf)
            this.textVectorizer = new TfidfVectorizer(iter,tokenizerFactory,labels);

        else
            this.textVectorizer = new BagOfWordsVectorizer(iter,tokenizerFactory,labels);

    }

    private void getIfNotExists() {

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

    }
}
