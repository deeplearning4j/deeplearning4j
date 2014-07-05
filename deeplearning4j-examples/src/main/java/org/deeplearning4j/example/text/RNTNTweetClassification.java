package org.deeplearning4j.example.text;

import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rntn.RNTN;
import org.deeplearning4j.rntn.Tree;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.text.treeparser.TreeVectorizer;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.inputsanitation.InputHomogenization;
import org.deeplearning4j.word2vec.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

/**
 * Created by agibsonccc on 7/5/14.
 */
public class RNTNTweetClassification {

    private static Logger log = LoggerFactory.getLogger(RNTNTweetClassification.class);


    public static void main(String[] args) throws Exception {
        InputStream is = new ClassPathResource("tweets_clean.txt").getInputStream();
        LabelAwareSentenceIterator lineIter = new LabelAwareListSentenceIterator(is);
        //get rid of @mentions
        lineIter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                String base =  new InputHomogenization(sentence).transform();
                base = base.replaceAll("@.*","");
                return base;
            }
        });


        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();

        Word2Vec vec = new Word2Vec(tokenizerFactory,lineIter,5);
        vec.train();


        lineIter.reset();


        RNTN r = new RNTN.Builder().setActivationFunction(Activations.hardTanh())
                .setNumHidden(50).setFeatureVectors(vec).setCombineClassification(true)
                .build();



        tokenizerFactory = new UimaTokenizerFactory(false);

        TreeVectorizer vectorizer = new TreeVectorizer();
        while(lineIter.hasNext()) {
            String sentence = lineIter.nextSentence();
            List<Tree> trees = vectorizer.getTreesWithLabels(sentence,lineIter.currentLabel(),Arrays.asList("0", "1", "2"));
            for(int i = 0; i < 10; i++)
                r.train(trees);
        }


    }


}
