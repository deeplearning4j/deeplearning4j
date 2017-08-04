package org.deeplearning4j.nn.conf.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.OneTimeLogger;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.*;
import org.nd4j.linalg.learning.config.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Alex on 22/02/2017.
 */
@Slf4j
public class LayerValidation {

    /**
     * Validate the updater configuration - setting the default updater values, if necessary
     */
    public static void updaterValidation(String layerName, Layer layer, Double learningRate, Double momentum,
                    Map<Integer, Double> momentumSchedule, Double adamMeanDecay, Double adamVarDecay, Double rho,
                    Double rmsDecay, Double epsilon) {
        BaseLayer bLayer;
        if (layer instanceof FrozenLayer && ((FrozenLayer) layer).getLayer() instanceof BaseLayer) {
            bLayer = (BaseLayer) ((FrozenLayer) layer).getLayer();
        } else if (layer instanceof BaseLayer) {
            bLayer = (BaseLayer) layer;
        } else {
            return;
        }
        updaterValidation(layerName, bLayer, learningRate == null ? Double.NaN : learningRate,
                        momentum == null ? Double.NaN : momentum, momentumSchedule,
                        adamMeanDecay == null ? Double.NaN : adamMeanDecay,
                        adamVarDecay == null ? Double.NaN : adamVarDecay, rho == null ? Double.NaN : rho,
                        rmsDecay == null ? Double.NaN : rmsDecay, epsilon == null ? Double.NaN : epsilon);
    }

    /**
     * Validate the updater configuration - setting the default updater values, if necessary
     */
    public static void updaterValidation(String layerName, BaseLayer layer, double learningRate, double momentum,
                    Map<Integer, Double> momentumSchedule, double adamMeanDecay, double adamVarDecay, double rho,
                    double rmsDecay, double epsilon) {
        if ((!Double.isNaN(momentum) || !Double.isNaN(layer.getMomentum())) && layer.getUpdater() != Updater.NESTEROVS)
            OneTimeLogger.warn(log,"Layer \"" + layerName
                            + "\" momentum has been set but will not be applied unless the updater is set to NESTEROVS.");
        if ((momentumSchedule != null || layer.getMomentumSchedule() != null)
                        && layer.getUpdater() != Updater.NESTEROVS)
            OneTimeLogger.warn(log,"Layer \"" + layerName
                            + "\" momentum schedule has been set but will not be applied unless the updater is set to NESTEROVS.");
        if ((!Double.isNaN(adamVarDecay) || (!Double.isNaN(layer.getAdamVarDecay())))
                        && layer.getUpdater() != Updater.ADAM)
            OneTimeLogger.warn(log,"Layer \"" + layerName
                            + "\" adamVarDecay is set but will not be applied unless the updater is set to Adam.");
        if ((!Double.isNaN(adamMeanDecay) || !Double.isNaN(layer.getAdamMeanDecay()))
                        && layer.getUpdater() != Updater.ADAM)
            OneTimeLogger.warn(log,"Layer \"" + layerName
                            + "\" adamMeanDecay is set but will not be applied unless the updater is set to Adam.");
        if ((!Double.isNaN(rho) || !Double.isNaN(layer.getRho())) && layer.getUpdater() != Updater.ADADELTA)
            OneTimeLogger.warn(log,"Layer \"" + layerName
                            + "\" rho is set but will not be applied unless the updater is set to ADADELTA.");
        if ((!Double.isNaN(rmsDecay) || (!Double.isNaN(layer.getRmsDecay()))) && layer.getUpdater() != Updater.RMSPROP)
            OneTimeLogger.warn(log,"Layer \"" + layerName
                            + "\" rmsdecay is set but will not be applied unless the updater is set to RMSPROP.");


        //Set values from old (deprecated) .epsilon(), .momentum(), etc methods to the built-in updaters
        //Note that there are *layer* versions (available via the layer) and *global* versions (via the method args)
        //The layer versions take precedence over the global versions. If neither are set, we use whatever is set
        // on the IUpdater instance, which may be the default, or may be user-configured
        //Note that default values for all other parameters are set by default in the Sgd/Adam/whatever classes
        //Hence we don't need to set them here
        //Finally: we'll also set the (updater enumeration field to something sane) to avoid updater=SGD,
        // iupdater=Adam() type situations. Though the updater field isn't used, we don't want to confuse users

        IUpdater u = layer.getIUpdater();
        if (!Double.isNaN(layer.getLearningRate())) {
            //Note that for LRs, if user specifies .learningRate(x).updater(Updater.SGD) (for example), we need to set the
            // LR in the Sgd object. We can do this using the schedules method, which also works for custom updaters
            //Local layer LR set
            u.applySchedules(0, layer.getLearningRate());
        } else if (!Double.isNaN(learningRate)) {
            //Global LR set
            u.applySchedules(0, learningRate);
        }


        if (u instanceof Sgd) {
            layer.setUpdater(Updater.SGD);

        } else if (u instanceof Adam) {
            Adam a = (Adam) u;
            if (!Double.isNaN(layer.getEpsilon())) {
                //user has done legacy .epsilon(...) on the layer itself
                a.setEpsilon(layer.getEpsilon());
            } else if (!Double.isNaN(epsilon)) {
                //user has done legacy .epsilon(...) on MultiLayerNetwork or ComputationGraph
                a.setEpsilon(epsilon);
            }

            if (!Double.isNaN(layer.getAdamMeanDecay())) {
                a.setBeta1(layer.getAdamMeanDecay());
            } else if (!Double.isNaN(adamMeanDecay)) {
                a.setBeta1(adamMeanDecay);
            }

            if (!Double.isNaN(layer.getAdamVarDecay())) {
                a.setBeta2(layer.getAdamVarDecay());
            } else if (!Double.isNaN(adamVarDecay)) {
                a.setBeta2(adamVarDecay);
            }

            layer.setUpdater(Updater.ADAM);

        } else if (u instanceof AdaDelta) {
            AdaDelta a = (AdaDelta) u;

            if (!Double.isNaN(layer.getRho())) {
                a.setRho(layer.getRho());
            } else if (!Double.isNaN(rho)) {
                a.setRho(rho);
            }

            if (!Double.isNaN(layer.getEpsilon())) {
                a.setEpsilon(layer.getEpsilon());
            } else if (!Double.isNaN(epsilon)) {
                a.setEpsilon(epsilon);
            }

            layer.setUpdater(Updater.ADADELTA);

        } else if (u instanceof Nesterovs) {
            Nesterovs n = (Nesterovs) u;
            if (!Double.isNaN(layer.getMomentum())) {
                n.setMomentum(layer.getMomentum());
            } else if (!Double.isNaN(momentum)) {
                n.setMomentum(momentum);
            }

            if (layer.getMomentumSchedule() != null && !layer.getMomentumSchedule().isEmpty()) {
                n.setMomentumSchedule(layer.getMomentumSchedule());
            } else if (momentumSchedule != null && !momentumSchedule.isEmpty()) {
                n.setMomentumSchedule(momentumSchedule);
            }
            layer.setUpdater(Updater.NESTEROVS);

        } else if (u instanceof AdaGrad) {
            AdaGrad a = (AdaGrad) u;
            if (!Double.isNaN(layer.getEpsilon())) {
                a.setEpsilon(layer.getEpsilon());
            } else if (!Double.isNaN(epsilon)) {
                a.setEpsilon(epsilon);
            }

            layer.setUpdater(Updater.ADAGRAD);

        } else if (u instanceof RmsProp) {
            RmsProp r = (RmsProp) u;

            if (!Double.isNaN(layer.getEpsilon())) {
                r.setEpsilon(layer.getEpsilon());
            } else if (!Double.isNaN(epsilon)) {
                r.setEpsilon(epsilon);
            }

            if (!Double.isNaN(layer.getRmsDecay())) {
                r.setRmsDecay(layer.getRmsDecay());
            } else if (!Double.isNaN(rmsDecay)) {
                r.setRmsDecay(rmsDecay);
            }
            layer.setUpdater(Updater.RMSPROP);

        } else if (u instanceof AdaMax) {
            AdaMax a = (AdaMax) u;

            if (!Double.isNaN(layer.getEpsilon())) {
                a.setEpsilon(layer.getEpsilon());
            } else if (!Double.isNaN(epsilon)) {
                a.setEpsilon(epsilon);
            }

            if (!Double.isNaN(layer.getAdamMeanDecay())) {
                a.setBeta1(layer.getAdamMeanDecay());
            } else if (!Double.isNaN(adamMeanDecay)) {
                a.setBeta1(adamMeanDecay);
            }

            if (!Double.isNaN(layer.getAdamVarDecay())) {
                a.setBeta2(layer.getAdamVarDecay());
            } else if (!Double.isNaN(adamVarDecay)) {
                a.setBeta2(adamVarDecay);
            }
            layer.setUpdater(Updater.ADAMAX);

        } else if (u instanceof NoOp) {
            layer.setUpdater(Updater.NONE);
        } else {
            //Probably a custom updater
            layer.setUpdater(null);
        }


        //Finally: Let's set the legacy momentum, epsilon, rmsDecay fields on the layer
        //At this point, it's purely cosmetic, to avoid NaNs etc there that might confuse users
        //The *true* values are now in the IUpdater instances
        if (layer.getUpdater() != null) { //May be null with custom updaters etc
            switch (layer.getUpdater()) {
                case NESTEROVS:
                    if (Double.isNaN(momentum) && Double.isNaN(layer.getMomentum())) {
                        layer.setMomentum(Nesterovs.DEFAULT_NESTEROV_MOMENTUM);
                    } else if (Double.isNaN(layer.getMomentum()))
                        layer.setMomentum(momentum);
                    if (momentumSchedule != null && layer.getMomentumSchedule() == null)
                        layer.setMomentumSchedule(momentumSchedule);
                    else if (momentumSchedule == null && layer.getMomentumSchedule() == null)
                        layer.setMomentumSchedule(new HashMap<Integer, Double>());
                    break;
                case ADAM:
                    if (Double.isNaN(adamMeanDecay) && Double.isNaN(layer.getAdamMeanDecay())) {
                        layer.setAdamMeanDecay(Adam.DEFAULT_ADAM_BETA1_MEAN_DECAY);
                    } else if (Double.isNaN(layer.getAdamMeanDecay()))
                        layer.setAdamMeanDecay(adamMeanDecay);

                    if (Double.isNaN(adamVarDecay) && Double.isNaN(layer.getAdamVarDecay())) {
                        layer.setAdamVarDecay(Adam.DEFAULT_ADAM_BETA2_VAR_DECAY);
                    } else if (Double.isNaN(layer.getAdamVarDecay()))
                        layer.setAdamVarDecay(adamVarDecay);

                    if (Double.isNaN(epsilon) && Double.isNaN(layer.getEpsilon())) {
                        layer.setEpsilon(Adam.DEFAULT_ADAM_EPSILON);
                    } else if (Double.isNaN(layer.getEpsilon())) {
                        layer.setEpsilon(epsilon);
                    }
                    break;
                case ADADELTA:
                    if (Double.isNaN(rho) && Double.isNaN(layer.getRho())) {
                        layer.setRho(AdaDelta.DEFAULT_ADADELTA_RHO);
                    } else if (Double.isNaN(layer.getRho())) {
                        layer.setRho(rho);
                    }

                    if (Double.isNaN(epsilon) && Double.isNaN(layer.getEpsilon())) {
                        layer.setEpsilon(AdaDelta.DEFAULT_ADADELTA_EPSILON);
                    } else if (Double.isNaN(layer.getEpsilon())) {
                        layer.setEpsilon(epsilon);
                    }
                    break;
                case ADAGRAD:
                    if (Double.isNaN(epsilon) && Double.isNaN(layer.getEpsilon())) {
                        layer.setEpsilon(AdaGrad.DEFAULT_ADAGRAD_EPSILON);
                    } else if (Double.isNaN(layer.getEpsilon())) {
                        layer.setEpsilon(epsilon);
                    }
                    break;
                case RMSPROP:
                    if (Double.isNaN(rmsDecay) && Double.isNaN(layer.getRmsDecay())) {
                        layer.setRmsDecay(RmsProp.DEFAULT_RMSPROP_RMSDECAY);
                    } else if (Double.isNaN(layer.getRmsDecay()))
                        layer.setRmsDecay(rmsDecay);

                    if (Double.isNaN(epsilon) && Double.isNaN(layer.getEpsilon())) {
                        layer.setEpsilon(RmsProp.DEFAULT_RMSPROP_EPSILON);
                    } else if (Double.isNaN(layer.getEpsilon())) {
                        layer.setEpsilon(epsilon);
                    }
                    break;
                case ADAMAX:
                    if (Double.isNaN(adamMeanDecay) && Double.isNaN(layer.getAdamMeanDecay())) {
                        layer.setAdamMeanDecay(AdaMax.DEFAULT_ADAMAX_BETA1_MEAN_DECAY);
                    } else if (Double.isNaN(layer.getAdamMeanDecay()))
                        layer.setAdamMeanDecay(adamMeanDecay);

                    if (Double.isNaN(adamVarDecay) && Double.isNaN(layer.getAdamVarDecay())) {
                        layer.setAdamVarDecay(AdaMax.DEFAULT_ADAMAX_BETA2_VAR_DECAY);
                    } else if (Double.isNaN(layer.getAdamVarDecay()))
                        layer.setAdamVarDecay(adamVarDecay);

                    if (Double.isNaN(epsilon) && Double.isNaN(layer.getEpsilon())) {
                        layer.setEpsilon(AdaMax.DEFAULT_ADAMAX_EPSILON);
                    } else if (Double.isNaN(layer.getEpsilon())) {
                        layer.setEpsilon(epsilon);
                    }
            }
        }
    }

    public static void generalValidation(String layerName, Layer layer, boolean useRegularization,
                    boolean useDropConnect, Double dropOut, Double l2, Double l2Bias, Double l1, Double l1Bias,
                    Distribution dist) {
        generalValidation(layerName, layer, useRegularization, useDropConnect, dropOut == null ? 0.0 : dropOut,
                        l2 == null ? Double.NaN : l2, l2Bias == null ? Double.NaN : l2Bias,
                        l1 == null ? Double.NaN : l1, l1Bias == null ? Double.NaN : l1Bias, dist);
    }

    public static void generalValidation(String layerName, Layer layer, boolean useRegularization,
                    boolean useDropConnect, double dropOut, double l2, double l2Bias, double l1, double l1Bias,
                    Distribution dist) {

        if (layer != null) {

            if (useDropConnect && (Double.isNaN(dropOut) && (Double.isNaN(layer.getDropOut()))))
                OneTimeLogger.warn(log,"Layer \"" + layerName
                                + "\" dropConnect is set to true but dropout rate has not been added to configuration.");
            if (useDropConnect && layer.getDropOut() == 0.0)
                OneTimeLogger.warn(log,"Layer \"" + layerName + " dropConnect is set to true but dropout rate is set to 0.0");

            if (layer instanceof BaseLayer) {
                BaseLayer bLayer = (BaseLayer) layer;
                configureBaseLayer(layerName, bLayer, useRegularization, useDropConnect, dropOut, l2, l2Bias, l1,
                                l1Bias, dist);
            } else if (layer instanceof FrozenLayer && ((FrozenLayer) layer).getLayer() instanceof BaseLayer) {
                BaseLayer bLayer = (BaseLayer) ((FrozenLayer) layer).getLayer();
                configureBaseLayer(layerName, bLayer, useRegularization, useDropConnect, dropOut, l2, l2Bias, l1,
                                l1Bias, dist);
            }
        }
    }

    private static void configureBaseLayer(String layerName, BaseLayer bLayer, boolean useRegularization,
                    boolean useDropConnect, Double dropOut, Double l2, Double l2Bias, Double l1, Double l1Bias,
                    Distribution dist) {
        if (useRegularization && (Double.isNaN(l1) && Double.isNaN(bLayer.getL1()) && Double.isNaN(l2)
                        && Double.isNaN(bLayer.getL2()) && Double.isNaN(l2Bias) && Double.isNaN(l1Bias)
                        && (Double.isNaN(dropOut) || dropOut == 0.0)
                        && (Double.isNaN(bLayer.getDropOut()) || bLayer.getDropOut() == 0.0)))
            OneTimeLogger.warn(log,"Layer \"" + layerName
                            + "\" regularization is set to true but l1, l2 or dropout has not been added to configuration.");

        if (useRegularization) {
            if (!Double.isNaN(l1) && Double.isNaN(bLayer.getL1())) {
                bLayer.setL1(l1);
            }
            if (!Double.isNaN(l2) && Double.isNaN(bLayer.getL2())) {
                bLayer.setL2(l2);
            }
            if (!Double.isNaN(l1Bias) && Double.isNaN(bLayer.getL1Bias())) {
                bLayer.setL1Bias(l1Bias);
            }
            if (!Double.isNaN(l2Bias) && Double.isNaN(bLayer.getL2Bias())) {
                bLayer.setL2Bias(l2Bias);
            }
        } else if (!useRegularization && ((!Double.isNaN(l1) && l1 > 0.0)
                        || (!Double.isNaN(bLayer.getL1()) && bLayer.getL1() > 0.0) || (!Double.isNaN(l2) && l2 > 0.0)
                        || (!Double.isNaN(bLayer.getL2()) && bLayer.getL2() > 0.0)
                        || (!Double.isNaN(l1Bias) && l1Bias > 0.0)
                        || (!Double.isNaN(bLayer.getL1Bias()) && bLayer.getL1Bias() > 0.0)
                        || (!Double.isNaN(l2Bias) && l2Bias > 0.0)
                        || (!Double.isNaN(bLayer.getL2Bias()) && bLayer.getL2Bias() > 0.0))) {
            OneTimeLogger.warn(log,"Layer \"" + layerName
                            + "\" l1 or l2 has been added to configuration but useRegularization is set to false.");
        }

        if (Double.isNaN(l2) && Double.isNaN(bLayer.getL2())) {
            bLayer.setL2(0.0);
        }
        if (Double.isNaN(l1) && Double.isNaN(bLayer.getL1())) {
            bLayer.setL1(0.0);
        }
        if (Double.isNaN(l2Bias) && Double.isNaN(bLayer.getL2Bias())) {
            bLayer.setL2Bias(0.0);
        }
        if (Double.isNaN(l1Bias) && Double.isNaN(bLayer.getL1Bias())) {
            bLayer.setL1Bias(0.0);
        }


        if (bLayer.getWeightInit() == WeightInit.DISTRIBUTION) {
            if (dist != null && bLayer.getDist() == null)
                bLayer.setDist(dist);
            else if (dist == null && bLayer.getDist() == null) {
                bLayer.setDist(new NormalDistribution(0, 1));
                OneTimeLogger.warn(log,"Layer \"" + layerName
                                + "\" distribution is automatically set to normalize distribution with mean 0 and variance 1.");
            }
        } else if ((dist != null || bLayer.getDist() != null)) {
            OneTimeLogger.warn(log,"Layer \"" + layerName
                            + "\" distribution is set but will not be applied unless weight init is set to WeighInit.DISTRIBUTION.");
        }
    }
}
