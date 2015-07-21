package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;

/**
 *
 *
 * @author Adam Gibson
 */
public class UpdaterCreator {

    /**
     * Create an updater based on the configuration
     * @param conf the configuration to get the updater for
     * @return the updater for the configuration
     */
    public static org.deeplearning4j.nn.api.Updater getUpdater(NeuralNetConfiguration conf) {
        Updater updater = conf.getUpdater();

        switch(updater) {
            case ADADELTA: return new AdaDeltaUpdater();
            case ADAGRAD: return new AdaGradUpdater();
            case ADAM: return new AdamUpdater();
            case NESTEROVS: return new NesterovsUpdater();
            case RMSPROP: return new RmsPropUpdater();
            case SGD: return new SgdUpdater();
            case CUSTOM: throw new UnsupportedOperationException("Not implemented yet.");

        }

        return null;
    }



}
