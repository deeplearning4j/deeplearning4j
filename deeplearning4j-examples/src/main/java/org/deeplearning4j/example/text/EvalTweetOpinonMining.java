package org.deeplearning4j.example.text;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.deeplearning4j.word2vec.inputsanitation.InputHomogenization;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.word2vec.vectorizer.TextVectorizer;
import org.deeplearning4j.word2vec.vectorizer.TfidfVectorizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.InputStream;
import java.util.Arrays;

/**
 * Created by agibsonccc on 5/4/14.
 */
public class EvalTweetOpinonMining {

    private static Logger log = LoggerFactory.getLogger(EvalTweetOpinonMining.class);

    public static void main(String[] args) throws  Exception {
        ClassPathResource resource = new ClassPathResource("/tweets_clean.txt");
        InputStream is = resource.getInputStream();


        LabelAwareListSentenceIterator iterator = new LabelAwareListSentenceIterator(is);
        iterator.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return new InputHomogenization(sentence).transform();
            }
        });
        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
        TextVectorizer vectorizor = new TfidfVectorizer(iterator,tokenizerFactory, Arrays.asList("0", "1", "2"),2000);
        DataSet data = vectorizor.vectorize();
        log.info("Vocab " + vectorizor.vocab());
        DataSetIterator iter = new ListDataSetIterator(data.asList(),10);

        Evaluation eval = new Evaluation();

        DBN d = SerializationUtils.readObject(new File("model-saver"));

        while(iter.hasNext()) {
            DataSet next = iter.next();
            eval.eval(next.getSecond(),d.output(next.getFirst()));
        }

        log.info(eval.stats());


    }

}
