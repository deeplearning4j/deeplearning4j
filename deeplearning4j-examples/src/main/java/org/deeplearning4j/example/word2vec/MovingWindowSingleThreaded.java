package org.deeplearning4j.example.word2vec;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.inputsanitation.InputHomogenization;
import org.deeplearning4j.word2vec.iterator.Word2VecDataSetIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.InputStream;
import java.util.Arrays;

/**
 * Created by agibsonccc on 6/13/14.
 */
public class MovingWindowSingleThreaded {

    private static Logger log = LoggerFactory.getLogger(MovingWindowSingleThreaded.class);

    public static void main(String[] args) throws Exception {
        InputStream is = FileUtils.openInputStream(new File(args[0]));


        LabelAwareListSentenceIterator iterator = new LabelAwareListSentenceIterator(is,"\t");
        //get rid of @mentions
        iterator.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                String base =  new InputHomogenization(sentence).transform();
                base = base.replaceAll("@.*","");
                return base;
            }
        });


        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
        File vecModel = new File("tweet-wordvectors.ser");
        Word2Vec vec = vecModel.exists() ? (Word2Vec) SerializationUtils.readObject(vecModel) : new Word2Vec(tokenizerFactory,iterator,5);
        if(!vecModel.exists()) {
            vec.train();

            log.info("Saving word 2 vec model...");

            SerializationUtils.saveObject(vec,new File("tweet-wordvectors.ser"));


        }

        else
            vec.setTokenizerFactory(tokenizerFactory);







        iterator.reset();



        DataSetIterator iter = new Word2VecDataSetIterator(vec,iterator, Arrays.asList("0", "1", "2"));
        log.info("Training on num columns " + iter.inputColumns());
        if(!iter.hasNext())
            throw new IllegalStateException("No data found");

        DBN dbn = new DBN.Builder().withHiddenUnits(RBM.HiddenUnit.RECTIFIED)
                .withVisibleUnits(RBM.VisibleUnit.GAUSSIAN).withL2(2e-4).useRegularization(true)
                .numberOfInputs(vec.getWindow() * vec.getLayerSize())
                .hiddenLayerSizes(new int[]{200})
                .numberOfOutPuts(3).withActivation(Activations.hardTanh())
                .build();
        while(iter.hasNext()) {
            DataSet d = iter.next();
            dbn.setInput(d.getFirst());
            dbn.finetune(d.getSecond(),1e-2,1000);
        }

        iter.reset();

        Evaluation eval = new Evaluation();

        while(iter.hasNext()) {
            DataSet d = iter.next();
            eval.eval(d.getSecond(),dbn.output(d.getFirst()));
        }

        log.info(eval.stats());


    }

}
