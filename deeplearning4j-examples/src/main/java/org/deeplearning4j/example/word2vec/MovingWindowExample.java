package org.deeplearning4j.example.word2vec;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.core.DefaultModelSaver;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.deeplearning4j.util.Viterbi;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.inputsanitation.InputHomogenization;
import org.deeplearning4j.word2vec.iterator.Word2VecDataSetIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

/**
 * Completed textual moving window example
 */
public class MovingWindowExample {

    private static Logger log = LoggerFactory.getLogger(MovingWindowExample.class);

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



        DataSetIterator iter = new Word2VecDataSetIterator(vec,iterator,Arrays.asList("0","1"));
          /*
        Note that this is an example of how to train. The parameters are not optimally tuned here, but serve to demonstrate
        how to use bag of words classification
         */
        HazelCastStateTracker tracker = new HazelCastStateTracker(2200);
        if(!tracker.isPretrain())
            throw new IllegalStateException("Tracker should be on pretrain");
        Conf c = new Conf();
        c.setFinetuneEpochs(10000);
        c.setPretrainEpochs(10000);
        c.setFinetuneLearningRate(1e-1);

        c.setLayerSizes(new int[]{iter.inputColumns() / 4,iter.inputColumns() / 4, iter.inputColumns() / 3});
        c.setUseAdaGrad(true);
        c.setMomentum(0.3);
        //c.setRenderWeightEpochs(1000);
        c.setnOut(2);
        c.setSplit(10);
        c.setnIn(vec.getLayerSize() * vec.getWindow());
        c.setHiddenUnit(RBM.HiddenUnit.RECTIFIED);
        c.setVisibleUnit(RBM.VisibleUnit.GAUSSIAN);
        c.setMultiLayerClazz(DBN.class);
        c.setUseRegularization(true);
        c.setL2(2e-4);
        c.setDeepLearningParams(new Object[]{1,1e-1,1000});
        ActorNetworkRunner runner = new ActorNetworkRunner("master",iter);
        runner.setModelSaver(new DefaultModelSaver(new File("word2vec-modelsaver.ser")));
        runner.setStateTracker(tracker);
        runner.setup(c);
        runner.train();


        iterator.reset();

        //viterbi optimization
        Viterbi viterbi = new Viterbi(new DoubleMatrix(new double[]{0,1}));
        iter.reset();
        while(iter.hasNext()) {
            DataSet next = iter.next();

            Pair<Double,DoubleMatrix> decoded = viterbi.decode(next.getSecond());
            log.info("Pair " + decoded.getSecond());
        }


    }



}
