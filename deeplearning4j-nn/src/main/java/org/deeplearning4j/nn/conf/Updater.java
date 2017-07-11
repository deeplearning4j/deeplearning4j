package org.deeplearning4j.nn.conf;

import org.nd4j.linalg.learning.config.*;

/**
 *
 * All the possible different updaters
 *
 * @author Adam Gibson
 */
public enum Updater {
    SGD, ADAM, ADAMAX, ADADELTA, NESTEROVS, NADAM, ADAGRAD, RMSPROP, NONE, @Deprecated CUSTOM;


    public IUpdater getIUpdaterWithDefaultConfig() {
        switch (this) {
            case SGD:
                return new Sgd();
            case ADAM:
                return new Adam();
            case ADAMAX:
                return new AdaMax();
            case ADADELTA:
                return new AdaDelta();
            case NESTEROVS:
                return new Nesterovs();
            case NADAM:
                return new Nadam();
            case ADAGRAD:
                return new AdaGrad();
            case RMSPROP:
                return new RmsProp();
            case NONE:
                return new NoOp();
            case CUSTOM:
            default:
                throw new UnsupportedOperationException("Unknown or not supported updater: " + this);
        }
    }
}
