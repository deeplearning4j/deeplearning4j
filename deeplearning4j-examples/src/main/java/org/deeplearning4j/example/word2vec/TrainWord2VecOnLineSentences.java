package org.deeplearning4j.example.word2vec;

import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.inputsanitation.InputHomogenization;
import org.deeplearning4j.word2vec.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;

import java.io.File;

/**
 * Example on training word2vec with
 *
 * a sentence on each line
 */
public class TrainWord2VecOnLineSentences {

    public static void main(String[] args) throws Exception {
        File lines = new File(args[0]);
        SentenceIterator lineIter = new LineSentenceIterator(lines);
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

        SerializationUtils.saveObject(vec,new File(args[1]));


    }


}
