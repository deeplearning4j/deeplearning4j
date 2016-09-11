package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 *
 *
 * @author Adam Gibson
 */
public class UpdaterCreator {

    private UpdaterCreator() {
    }

    /**
     * Create an updater based on the configuration
     * @param conf the configuration to get the updater for
     * @return the updater for the configuration
     * @deprecated As of 0.6.0. Use {@link LayerUpdater instead}
     */
    @Deprecated
    private static org.deeplearning4j.nn.api.Updater getUpdater(NeuralNetConfiguration conf) {
        Updater updater = conf.getLayer().getUpdater();

        switch(updater) {
            case ADADELTA: return new AdaDeltaUpdater();
            case ADAGRAD: return new AdaGradUpdater();
            case ADAM: return new AdamUpdater();
            case NESTEROVS: return new NesterovsUpdater();
            case RMSPROP: return new RmsPropUpdater();
            case SGD: return new SgdUpdater();
            case NONE: return new NoOpUpdater();
            case CUSTOM: throw new UnsupportedOperationException("Not implemented yet.");
        }

        return null;
    }

    public static org.deeplearning4j.nn.api.Updater getUpdater(Model layer) {
    	if( layer instanceof MultiLayerNetwork ){
    		return new MultiLayerUpdater((MultiLayerNetwork)layer);
    	} else {
            return new LayerUpdater();
    	}
    }

}
