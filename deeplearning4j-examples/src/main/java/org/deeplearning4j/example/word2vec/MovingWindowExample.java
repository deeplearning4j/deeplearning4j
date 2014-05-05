package org.deeplearning4j.example.word2vec;

import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.inputsanitation.InputHomogenization;
import org.deeplearning4j.word2vec.iterator.Word2VecDataSetIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.word2vec.util.Window;
import org.deeplearning4j.word2vec.util.Windows;
import org.deeplearning4j.word2vec.viterbi.CounterUtil;
import org.deeplearning4j.word2vec.viterbi.Index;
import org.deeplearning4j.word2vec.viterbi.Viterbi;
import org.deeplearning4j.word2vec.viterbi.ViterbiUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

/**
 * Completed textual moving window example
 */
public class MovingWindowExample {

    private static Logger log = LoggerFactory.getLogger(MovingWindowExample.class);

    public static void main(String[] args) throws Exception {

        ClassPathResource resource = new ClassPathResource("/tweets_clean.txt");
        InputStream is = resource.getInputStream();

        List<String> labels = Arrays.asList("0","1","2");

        LabelAwareListSentenceIterator iterator = new LabelAwareListSentenceIterator(is);
        iterator.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return new InputHomogenization(sentence).transform();
            }
        });


        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();

        Word2Vec vec = new Word2Vec(tokenizerFactory,iterator,1);
        vec.train();

        log.info("Saving word 2 vec model...");

        iterator.reset();



        DataSetIterator iter = new Word2VecDataSetIterator(vec,iterator,Arrays.asList("0","1","2"));

          /*
        Note that this is an example of how to train. The parameters are not optimally tuned here, but serve to demonstrate
        how to use bag of words classification
         */
        HazelCastStateTracker tracker = new HazelCastStateTracker();
        if(!tracker.isPretrain())
            throw new IllegalStateException("Tracker should be on pretrain");
        Conf c = new Conf();
        c.setFinetuneEpochs(10000);
        c.setPretrainEpochs(10000);
        c.setFinetuneLearningRate(1e-1);

        c.setLayerSizes(new int[]{iter.inputColumns() / 2,iter.inputColumns() / 4, iter.inputColumns() / 3});
        c.setnIn(iter.inputColumns());
        c.setUseAdaGrad(true);
        c.setMomentum(0.3);
        c.setNormalizeZeroMeanAndUnitVariance(false);
        //c.setRenderWeightEpochs(1000);
        c.setnOut(3);
        c.setSplit(10);
        c.setHiddenUnit(RBM.HiddenUnit.RECTIFIED);
        c.setVisibleUnit(RBM.VisibleUnit.GAUSSIAN);
        c.setMultiLayerClazz(DBN.class);
        c.setUseRegularization(false);
        c.setDeepLearningParams(new Object[]{1,1e-1,1000});
        ActorNetworkRunner runner = new ActorNetworkRunner("master",iter);
        runner.setStateTracker(tracker);
        runner.setup(c);
        runner.train();


        iterator.reset();

        Index labelIndex = new Index();
        for(int i = 0; i < labels.size(); i++)
            labelIndex.add(labels.get(i));
        Index featureIndex = ViterbiUtil.featureIndexFromLabelIndex(labelIndex);
        CounterMap<Integer,Integer> transitions = new CounterMap<>();
        while(iterator.hasNext()) {
            String sentence = iterator.nextSentence();
            if(sentence.isEmpty())
                continue;
            List<Window> windows = Windows.windows(sentence);

            for(int i = 1; i < windows.size(); i++) {
                String firstLabel = windows.get(i - 1).getLabel();
                String secondLabel = windows.get(i).getLabel();
                int first = labelIndex.indexOf(firstLabel);
                int second = labelIndex.indexOf(secondLabel);


                transitions.incrementCount(first,second,1.0);
            }

            DoubleMatrix transitionWeights = CounterUtil.convert(transitions);
            //viterbi optimization
            Viterbi viterbi = new Viterbi(labelIndex,featureIndex,transitionWeights);



        }


    }

}
