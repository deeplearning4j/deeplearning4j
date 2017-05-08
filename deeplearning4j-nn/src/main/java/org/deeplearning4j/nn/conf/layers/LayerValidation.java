package org.deeplearning4j.nn.conf.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.weights.WeightInit;
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
        updaterValidation(layerName, layer, learningRate == null ? Double.NaN : learningRate,
                momentum == null ? Double.NaN : momentum, momentumSchedule,
                        adamMeanDecay == null ? Double.NaN : adamMeanDecay,
                        adamVarDecay == null ? Double.NaN : adamVarDecay, rho == null ? Double.NaN : rho,
                        rmsDecay == null ? Double.NaN : rmsDecay, epsilon == null ? Double.NaN : epsilon);
    }

    /**
     * Validate the updater configuration - setting the default updater values, if necessary
     */
    public static void updaterValidation(String layerName, Layer layer, double learningRate, double momentum,
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


        //Set values from old (deprecated) .epsilon(), .momentum(), etc methods to the built-in updaters
        //Also set LR, where appropriate
        //Note that default values for all other parameters are set by default in the Sgd/Adam/whatever classes
        //Hence we don't need to set them here

        IUpdater u = layer.getIUpdater();

        if( u instanceof Sgd){
            Sgd sgd = (Sgd)u;
            if(!Double.isNaN(learningRate)){
                sgd.setLearningRate(learningRate);
            }

        } else if(u instanceof Adam ){
            Adam a = (Adam)u;
            if(!Double.isNaN(epsilon)){
                a.setEpsilon(epsilon);
            }
            if(!Double.isNaN(adamMeanDecay)){
                a.setBeta1(adamMeanDecay);
            }
            if(!Double.isNaN(adamVarDecay)){
                a.setBeta2(adamVarDecay);
            }
            if(!Double.isNaN(learningRate)){
                a.setLearningRate(learningRate);
            }

        } else if(u instanceof AdaDelta) {
            AdaDelta a = (AdaDelta)u;
            if(!Double.isNaN(rho)){
                a.setRho(rho);
            }
            if(!Double.isNaN(epsilon)){
                a.setEpsilon(epsilon);
            }
            //No LR for AdaDelta

        } else if(u instanceof Nesterovs ){
            Nesterovs n = (Nesterovs)u;
            if(!Double.isNaN(momentum)){
                n.setMomentum(momentum);
            }
            if(momentumSchedule != null){
                n.setMomentumSchedule(momentumSchedule);
            }
            if(!Double.isNaN(learningRate)){
                n.setLearningRate(learningRate);
            }
        } else if(u instanceof AdaGrad){
            AdaGrad a = (AdaGrad)u;
            if(!Double.isNaN(epsilon)){
                a.setEpsilon(epsilon);
            }
            if(!Double.isNaN(learningRate)){
                a.setLearningRate(learningRate);
            }

        } else if(u instanceof RmsProp){
            RmsProp r = (RmsProp)u;
            if(!Double.isNaN(epsilon)){
                r.setEpsilon(epsilon);
            }
            if(!Double.isNaN(rmsDecay)){
                r.setRmsDecay(rmsDecay);
            }
            if(!Double.isNaN(learningRate)){
                r.setLearningRate(learningRate);
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
        if (useDropConnect && (Double.isNaN(dropOut) && (Double.isNaN(layer.getDropOut()))))
            log.warn("Layer \"" + layerName
                            + "\" dropConnect is set to true but dropout rate has not been added to configuration.");
        if (useDropConnect && layer.getDropOut() == 0.0)
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
