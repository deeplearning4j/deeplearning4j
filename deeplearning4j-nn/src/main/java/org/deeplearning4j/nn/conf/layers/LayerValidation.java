package org.deeplearning4j.nn.conf.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.*;

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
    public static void updaterValidation(String layerName, Layer layer, Double momentum,
                    Map<Integer, Double> momentumSchedule, Double adamMeanDecay, Double adamVarDecay, Double rho,
                    Double rmsDecay, Double epsilon) {
        updaterValidation(layerName, layer, momentum == null ? Double.NaN : momentum, momentumSchedule,
                        adamMeanDecay == null ? Double.NaN : adamMeanDecay,
                        adamVarDecay == null ? Double.NaN : adamVarDecay, rho == null ? Double.NaN : rho,
                        rmsDecay == null ? Double.NaN : rmsDecay, epsilon == null ? Double.NaN : epsilon);

    }

    /**
     * Validate the updater configuration - setting the default updater values, if necessary
     */
    public static void updaterValidation(String layerName, Layer layer, double momentum,
                    Map<Integer, Double> momentumSchedule, double adamMeanDecay, double adamVarDecay, double rho,
                    double rmsDecay, double epsilon) {
        if ((!Double.isNaN(momentum) || !Double.isNaN(layer.getMomentum())) && layer.getUpdater() != Updater.NESTEROVS)
            log.warn("Layer \"" + layerName
                            + "\" momentum has been set but will not be applied unless the updater is set to NESTEROVS.");
        if ((momentumSchedule != null || layer.getMomentumSchedule() != null)
                        && layer.getUpdater() != Updater.NESTEROVS)
            log.warn("Layer \"" + layerName
                            + "\" momentum schedule has been set but will not be applied unless the updater is set to NESTEROVS.");
        if ((!Double.isNaN(adamVarDecay) || (!Double.isNaN(layer.getAdamVarDecay())))
                        && layer.getUpdater() != Updater.ADAM)
            log.warn("Layer \"" + layerName
                            + "\" adamVarDecay is set but will not be applied unless the updater is set to Adam.");
        if ((!Double.isNaN(adamMeanDecay) || !Double.isNaN(layer.getAdamMeanDecay()))
                        && layer.getUpdater() != Updater.ADAM)
            log.warn("Layer \"" + layerName
                            + "\" adamMeanDecay is set but will not be applied unless the updater is set to Adam.");
        if ((!Double.isNaN(rho) || !Double.isNaN(layer.getRho())) && layer.getUpdater() != Updater.ADADELTA)
            log.warn("Layer \"" + layerName
                            + "\" rho is set but will not be applied unless the updater is set to ADADELTA.");
        if ((!Double.isNaN(rmsDecay) || (!Double.isNaN(layer.getRmsDecay()))) && layer.getUpdater() != Updater.RMSPROP)
            log.warn("Layer \"" + layerName
                            + "\" rmsdecay is set but will not be applied unless the updater is set to RMSPROP.");

        switch (layer.getUpdater()) {
            case NESTEROVS:
                if (Double.isNaN(momentum) && Double.isNaN(layer.getMomentum())) {
                    layer.setMomentum(Nesterovs.DEFAULT_NESTEROV_MOMENTUM);
                    log.warn("Layer \"" + layerName + "\" momentum is automatically set to "
                                    + Nesterovs.DEFAULT_NESTEROV_MOMENTUM
                                    + ". Add momentum to configuration to change the value.");
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
                    log.warn("Layer \"" + layerName + "\" adamMeanDecay is automatically set to "
                                    + Adam.DEFAULT_ADAM_BETA1_MEAN_DECAY
                                    + ". Add adamVarDecay to configuration to change the value.");
                } else if (Double.isNaN(layer.getAdamMeanDecay()))
                    layer.setAdamMeanDecay(adamMeanDecay);

                if (Double.isNaN(adamVarDecay) && Double.isNaN(layer.getAdamVarDecay())) {
                    layer.setAdamVarDecay(Adam.DEFAULT_ADAM_BETA2_VAR_DECAY);
                    log.warn("Layer \"" + layerName + "\" adamVarDecay is automatically set to "
                                    + Adam.DEFAULT_ADAM_BETA2_VAR_DECAY
                                    + ". Add adamVarDecay to configuration to change the value.");
                } else if (Double.isNaN(layer.getAdamVarDecay()))
                    layer.setAdamVarDecay(adamVarDecay);

                if (Double.isNaN(epsilon) && Double.isNaN(layer.getEpsilon())) {
                    layer.setEpsilon(Adam.DEFAULT_ADAM_EPSILON);
                } else if (Double.isNaN(layer.getEpsilon())) {
                    layer.setEpsilon(epsilon);
                }
                break;
            case ADADELTA:
                if (Double.isNaN(layer.getRho()))
                    layer.setRho(rho);

                if (Double.isNaN(epsilon) && Double.isNaN(layer.getEpsilon())) {
                    layer.setEpsilon(AdaDelta.DEFAULT_ADADELTA_EPSILON);
                    log.warn("Layer \"" + layerName + "\" AdaDelta epsilon is automatically set to "
                                    + AdaDelta.DEFAULT_ADADELTA_EPSILON
                                    + ". Add epsilon to configuration to change the value.");
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
                    log.warn("Layer \"" + layerName
                                    + "\" rmsDecay is automatically set to 0.95. Add rmsDecay to configuration to change the value.");
                } else if (Double.isNaN(layer.getRmsDecay()))
                    layer.setRmsDecay(rmsDecay);

                if (Double.isNaN(epsilon) && Double.isNaN(layer.getEpsilon())) {
                    layer.setEpsilon(RmsProp.DEFAULT_RMSPROP_EPSILON);
                } else if (Double.isNaN(layer.getEpsilon())) {
                    layer.setEpsilon(epsilon);
                }

                break;
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
        if (useDropConnect && (Double.isNaN(dropOut) && (Double.isNaN(layer.getDropOut()))))
            log.warn("Layer \"" + layerName
                            + "\" dropConnect is set to true but dropout rate has not been added to configuration.");
        if (useDropConnect && dropOut == 0.0)
            log.warn("Layer \"" + layerName + " dropConnect is set to true but dropout rate is set to 0.0");
        if (useRegularization && (Double.isNaN(l1) && layer != null && Double.isNaN(layer.getL1()) && Double.isNaN(l2)
                        && Double.isNaN(layer.getL2()) && Double.isNaN(l2Bias) && Double.isNaN(l1Bias)
                        && (Double.isNaN(dropOut) || dropOut == 0.0)
                        && (Double.isNaN(layer.getDropOut()) || layer.getDropOut() == 0.0)))
            log.warn("Layer \"" + layerName
                            + "\" regularization is set to true but l1, l2 or dropout has not been added to configuration.");

        if (layer != null) {
            if (useRegularization) {
                if (!Double.isNaN(l1) && Double.isNaN(layer.getL1())) {
                    layer.setL1(l1);
                }
                if (!Double.isNaN(l2) && Double.isNaN(layer.getL2())) {
                    layer.setL2(l2);
                }
                if (!Double.isNaN(l1Bias) && Double.isNaN(layer.getL1Bias())) {
                    layer.setL1Bias(l1Bias);
                }
                if (!Double.isNaN(l2Bias) && Double.isNaN(layer.getL2Bias())) {
                    layer.setL2Bias(l2Bias);
                }
            } else if (!useRegularization && ((!Double.isNaN(l1) && l1 > 0.0)
                            || (!Double.isNaN(layer.getL1()) && layer.getL1() > 0.0) || (!Double.isNaN(l2) && l2 > 0.0)
                            || (!Double.isNaN(layer.getL2()) && layer.getL2() > 0.0)
                            || (!Double.isNaN(l1Bias) && l1Bias > 0.0)
                            || (!Double.isNaN(layer.getL1Bias()) && layer.getL1Bias() > 0.0)
                            || (!Double.isNaN(l2Bias) && l2Bias > 0.0)
                            || (!Double.isNaN(layer.getL2Bias()) && layer.getL2Bias() > 0.0))) {
                log.warn("Layer \"" + layerName
                                + "\" l1 or l2 has been added to configuration but useRegularization is set to false.");
            }

            if (Double.isNaN(l2) && Double.isNaN(layer.getL2())) {
                layer.setL2(0.0);
            }
            if (Double.isNaN(l1) && Double.isNaN(layer.getL1())) {
                layer.setL1(0.0);
            }
            if (Double.isNaN(l2Bias) && Double.isNaN(layer.getL2Bias())) {
                layer.setL2Bias(0.0);
            }
            if (Double.isNaN(l1Bias) && Double.isNaN(layer.getL1Bias())) {
                layer.setL1Bias(0.0);
            }


            if (layer.getWeightInit() == WeightInit.DISTRIBUTION) {
                if (dist != null && layer.getDist() == null)
                    layer.setDist(dist);
                else if (dist == null && layer.getDist() == null) {
                    layer.setDist(new NormalDistribution(0, 1));
                    log.warn("Layer \"" + layerName
                                    + "\" distribution is automatically set to normalize distribution with mean 0 and variance 1.");
                }
            } else if ((dist != null || layer.getDist() != null)) {
                log.warn("Layer \"" + layerName
                                + "\" distribution is set but will not be applied unless weight init is set to WeighInit.DISTRIBUTION.");
            }
        }

    }
}
