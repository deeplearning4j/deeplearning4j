package org.deeplearning4j.example.text;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.word2vec.vectorizer.TextVectorizer;
import org.deeplearning4j.word2vec.vectorizer.TfidfVectorizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by agibsonccc on 4/13/14.
 */
public class TweetOpinionMining {

    private static Logger log = LoggerFactory.getLogger(TweetOpinionMining.class);

    public static void main(String[] args) throws Exception {

        ClassPathResource resource = new ClassPathResource("/tweets_clean.txt");
        InputStream is = resource.getInputStream();


        LabelAwareListSentenceIterator iterator = new LabelAwareListSentenceIterator(is);
        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
        TextVectorizer vectorizor = new TfidfVectorizer(iterator,tokenizerFactory,Arrays.asList("0","1","2"));
        DataSet data = vectorizor.vectorize();



        log.info("Example tweets " + data.numExamples());


    }

}
