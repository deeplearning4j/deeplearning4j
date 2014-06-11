package org.deeplearning4j.example.text;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.text.tokenizerfactory.PosUimaTokenizerFactory;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
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
 * @author Adam Gibson
 */
public class MultiThreadedTweetOpinionMining {
    private static Logger log = LoggerFactory.getLogger(MultiThreadedTweetOpinionMining.class);

    public static void main(String[] args) throws Exception {
        InputStream is = FileUtils.openInputStream(new File(args[0]));


        LabelAwareListSentenceIterator iterator = new LabelAwareListSentenceIterator(is,",");
        iterator.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return new InputHomogenization(sentence).transform();
            }
        });


        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
        //note that we are only using 2000 length feature vectors. Due to the sparsity of bag of words, this is actually very little.
        //We need to tune the number of words (bump them up, or use better filtering mechanisms)

        TextVectorizer vectorizor = new TfidfVectorizer(iterator,tokenizerFactory, Arrays.asList("0", "1"),4000);
        DataSet data = vectorizor.vectorize();
        data.binarize();
        log.info("Vocab " + vectorizor.vocab());
        DataSetIterator iter = new ListDataSetIterator(data.asList(),10);


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
        c.setnOut(2);
        c.setSplit(5);
        c.setHiddenUnit(RBM.HiddenUnit.BINARY);
        c.setVisibleUnit(RBM.VisibleUnit.BINARY);
        c.setMultiLayerClazz(DBN.class);
        c.setUseRegularization(true);
        c.setL2(2e-4);
        c.setDeepLearningParams(new Object[]{1,1e-1,1000});
        ActorNetworkRunner runner = new ActorNetworkRunner("master",iter);
        runner.setStateTracker(tracker);
        runner.setup(c);
        runner.train();


    }


}
