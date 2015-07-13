package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * @author Adam Gibson
 */
public class UpdaterCreator {
    public static GradientUpdater getUpdater(NeuralNetConfiguration conf) {
        boolean hasMomentum = conf.getMomentum() > 0;
        boolean hasLearningRate = conf.getLr() > 0;
        boolean hasRms = conf.getRmsDecay() > 0;
        boolean isAdaGrad = conf.isUseAdaGrad();

        if(conf.getMomentum() > 0) {

        }

        return null;
    }



}
