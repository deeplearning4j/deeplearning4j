package org.deeplearning4j.example.text;

import org.apache.commons.io.FileUtils;
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
 * Created by agibsonccc on 4/13/14.
 */
public class TweetOpinionMining {

    private static Logger log = LoggerFactory.getLogger(TweetOpinionMining.class);

    public static void main(String[] args) throws Exception {


        InputStream is = FileUtils.openInputStream(new File(args[0]));


        LabelAwareListSentenceIterator iterator = new LabelAwareListSentenceIterator(is);
        iterator.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return new InputHomogenization(sentence).transform();
            }
        });



        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
        TextVectorizer vectorizor = new TfidfVectorizer(iterator,tokenizerFactory,Arrays.asList("0","1"),2000);
        DataSet data = vectorizor.vectorize();
        data.binarize();


        SerializationUtils.saveObject(data,new File("tweet-data.ser"));

        log.info("Vocab " + vectorizor.vocab());
        DataSetIterator iter = new ListDataSetIterator(data.asList(),10);

        DBN dbn = new DBN.Builder()
                .hiddenLayerSizes(new int[]{iter.inputColumns() / 2,iter.inputColumns() / 4,iter.inputColumns() / 6})
                .numberOfInputs(iter.inputColumns()).numberOfOutPuts(iter.totalOutcomes())
                .build();

        while(iter.hasNext()) {
            DataSet next = iter.next();
            dbn.pretrain(next.getFirst(), 1, 1e-1, 10000);
        }

        iter.reset();

        while(iter.hasNext()) {
            DataSet next = iter.next();
            dbn.setInput(next.getFirst());
            dbn.finetune(next.getSecond(), 1e-1, 10000);
        }


        Evaluation eval = new Evaluation();
        eval.eval(data.getSecond(),dbn.predict(data.getFirst()));
        log.info(eval.stats());
        log.info("Example tweets " + data.numExamples());


    }

}
