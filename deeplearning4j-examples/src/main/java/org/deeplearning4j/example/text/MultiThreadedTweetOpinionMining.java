package org.deeplearning4j.example.text;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.scaleout.conf.Conf;
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

import java.io.InputStream;
import java.util.Arrays;

/**
 * Created by agibsonccc on 4/14/14.
 */
public class MultiThreadedTweetOpinionMining {
    private static Logger log = LoggerFactory.getLogger(MultiThreadedTweetOpinionMining.class);

    public static void main(String[] args) throws Exception {
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

        HazelCastStateTracker tracker = new HazelCastStateTracker();
        Conf c = new Conf();
        c.setFinetuneEpochs(10000);
        c.setPretrainEpochs(10000);
        c.setFinetuneLearningRate(1e-1);

        c.setLayerSizes(new int[]{iter.inputColumns() / 2,iter.inputColumns() / 4, iter.inputColumns() / 3});
        c.setnIn(iter.inputColumns());
        c.setUseAdaGrad(true);
        c.setNormalizeZeroMeanAndUnitVariance(true);
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


    }


}
