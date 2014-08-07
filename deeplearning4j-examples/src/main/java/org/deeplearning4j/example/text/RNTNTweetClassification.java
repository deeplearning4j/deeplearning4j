package org.deeplearning4j.example.text;

import net.didion.jwnl.data.Word;
import org.apache.commons.lang.StringUtils;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rntn.RNTN;
import org.deeplearning4j.rntn.RNTNEval;
import org.deeplearning4j.rntn.Tree;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.text.treeparser.TreeIterator;
import org.deeplearning4j.text.treeparser.TreeVectorizer;
import org.deeplearning4j.util.SerializationUtils;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.inputsanitation.InputHomogenization;
import org.deeplearning4j.word2vec.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * Created by agibsonccc on 7/5/14.
 */
public class RNTNTweetClassification {

    private static Logger log = LoggerFactory.getLogger(RNTNTweetClassification.class);


    public static void main(String[] args) throws Exception {
        InputStream is = new ClassPathResource("sentiment-milliontweets.csv").getInputStream();
        LabelAwareSentenceIterator lineIter = new LabelAwareListSentenceIterator(is,",",1,3);
        //getFromOrigin rid of @mentions
        lineIter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                String base = new InputHomogenization(sentence).transform();
                base = StringUtils.normalizeSpace(base.replaceAll("@.*", ""));
                return base;
            }
        });



        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
        File wordVectors = new File("tweet-wordvectors.ser");

        Word2Vec vec = wordVectors.exists() ? (Word2Vec) SerializationUtils.readObject(wordVectors) : new Word2Vec(tokenizerFactory, lineIter, 5);
        if(!wordVectors.exists()) {
            vec.train();
            SerializationUtils.saveObject(vec,wordVectors);
        }

        log.info("Initialized word vectors of size " + vec.getLayerSize());
        vec.setTokenizerFactory(tokenizerFactory);

        lineIter.reset();


        RNTN r = new RNTN.Builder().setActivationFunction(Activations.tanh())
                .setNumHidden(vec.getLayerSize()).setFeatureVectors(vec)
                .build();


        tokenizerFactory = new UimaTokenizerFactory(false);

        TreeVectorizer vectorizer = new TreeVectorizer();
        Iterator<List<Tree>> treeIter = new TreeIterator(lineIter,Arrays.asList("0", "1", "2"),vectorizer,10);
        while (treeIter.hasNext()) {
            List<Tree> trees = treeIter.next();
            if(trees.isEmpty())
                continue;
           for(int i = 0; i < 100; i++) {
               r.train(trees);
               RNTNEval eval = new RNTNEval();

               log.info("Value for batch " + r.getValue() + " at iteration " + i);
               eval.eval(r,trees);
               log.info("Eval stats " + eval.stats());

           }


           //log.info("Value for iteration " + i + " is " + r.getValue());



        }

    }


}
