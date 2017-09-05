package org.deeplearning4j.nn.conf.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.OneTimeLogger;
import org.nd4j.linalg.learning.config.*;

import java.util.*;

/**
 * Created by Alex on 22/02/2017.
 */
@Slf4j
public class LayerValidation {

    /**
     * Validate the updater configuration - setting the default updater values, if necessary
     */
//    public static void updaterValidation(String layerName, Layer layer, Double learningRate, Double biasLearningRate) {
    public static void updaterValidation(String layerName, Layer layer, Double learningRate, Double biasLearningRate, Double momentum,
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
                biasLearningRate == null ? Double.NaN : biasLearningRate,
                momentum == null ? Double.NaN : momentum, momentumSchedule,
                adamMeanDecay == null ? Double.NaN : adamMeanDecay,
                adamVarDecay == null ? Double.NaN : adamVarDecay, rho == null ? Double.NaN : rho,
                rmsDecay == null ? Double.NaN : rmsDecay, epsilon == null ? Double.NaN : epsilon);
    }

    /**
     * Validate the updater configuration - setting the default updater values, if necessary
     */
    public static void updaterValidation(String layerName, BaseLayer layer, double learningRate, double biasLearningRate, double momentum,
                                         Map<Integer, Double> momentumSchedule, double adamMeanDecay, double adamVarDecay, double rho,
                                         double rmsDecay, double epsilon) {
        //Set values from old (deprecated) .epsilon(), .momentum(), etc methods to the built-in updaters
        //Note that there are *layer* versions (available via the layer) and *global* versions (via the method args)
        //The layer versions take precedence over the global versions. If neither are set, we use whatever is set
        // on the IUpdater instance, which may be the default, or may be user-configured
        //Note that default values for all other parameters are set by default in the Sgd/Adam/whatever classes
        //Hence we don't need to set them here
        //Finally: we'll also set the (updater enumeration field to something sane) to avoid updater=SGD,
        // iupdater=Adam() type situations. Though the updater field isn't used, we don't want to confuse users

        IUpdater u = layer.getIUpdater();
//        if (!Double.isNaN(layer.getLearningRate())) {
//            //Note that for LRs, if user specifies .learningRate(x).updater(Updater.SGD) (for example), we need to set the
//            // LR in the Sgd object. We can do this using the schedules method, which also works for custom updaters
//            //Local layer LR set
//            setLegacyLr(u, layer.getLearningRate() );
//        } else
        if (!Double.isNaN(learningRate) && Double.isNaN(layer.getLearningRate())) {
            //Global LR set
            setLegacyLr(u, learningRate);
        }

        if(!Double.isNaN(biasLearningRate) && Double.isNaN(layer.getBiasLearningRate())){
            IUpdater ub = layer.getBiasUpdater();
            if(ub == null){
                ub = layer.getIUpdater().clone();
            }
            setLegacyLr(ub, biasLearningRate);
        }



        if (u instanceof Adam) {
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

        } else if (u instanceof Nesterovs) {
            Nesterovs n = (Nesterovs) u;
            if (!Double.isNaN(layer.getMomentum())) {
                n.setMomentum(layer.getMomentum());
            } else if (!Double.isNaN(momentum)) {
                n.setMomentum(momentum);
            }

        } else if (u instanceof AdaGrad) {
            AdaGrad a = (AdaGrad) u;
            if (!Double.isNaN(layer.getEpsilon())) {
                a.setEpsilon(layer.getEpsilon());
            } else if (!Double.isNaN(epsilon)) {
                a.setEpsilon(epsilon);
            }

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

        }
    }

    public static void setLegacyLr(IUpdater u, double lr){
        if(u instanceof AdaGrad){
            ((AdaGrad) u).setLearningRate(lr);
        } else if(u instanceof Adam){
            ((Adam) u).setLearningRate(lr);
        } else if(u instanceof AdaMax){
            ((AdaMax) u).setLearningRate(lr);
        } else if(u instanceof Nadam){
            ((Nadam) u).setLearningRate(lr);
        } else if(u instanceof Nesterovs){
            ((Nesterovs) u).setLearningRate(lr);
        } else if(u instanceof RmsProp){
            ((RmsProp) u).setLearningRate(lr);
        } else if(u instanceof Sgd){
            ((Sgd) u).setLearningRate(lr);
        }
    }

    public static Double getLr(IUpdater u){
        if(u instanceof AdaGrad){
            return ((AdaGrad) u).getLearningRate();
        } else if(u instanceof Adam){
            return ((Adam) u).getLearningRate();
        } else if(u instanceof AdaMax){
            return ((AdaMax) u).getLearningRate();
        } else if(u instanceof Nadam){
            return ((Nadam) u).getLearningRate();
        } else if(u instanceof Nesterovs){
            return ((Nesterovs) u).getLearningRate();
        } else if(u instanceof RmsProp){
            return ((RmsProp) u).getLearningRate();
        } else if(u instanceof Sgd){
            return ((Sgd) u).getLearningRate();
        }
        return null;
    }


    public static void generalValidation(String layerName, Layer layer, boolean useDropConnect, IDropout iDropOut,
                                         Double l2, Double l2Bias, Double l1, Double l1Bias,
                                         Distribution dist, List<LayerConstraint> allParamConstraints,
                                         List<LayerConstraint> weightConstraints, List<LayerConstraint> biasConstraints) {
        generalValidation(layerName, layer, useDropConnect, dropOut == null ? 0.0 : dropOut,
                        l2 == null ? Double.NaN : l2, l2Bias == null ? Double.NaN : l2Bias,
                        l1 == null ? Double.NaN : l1, l1Bias == null ? Double.NaN : l1Bias, dist, allParamConstraints, weightConstraints, biasConstraints);
    }

    public static void generalValidation(String layerName, Layer layer, boolean useDropConnect, IDropout iDropout,
                                         double l2, double l2Bias, double l1, double l1Bias,
                                         Distribution dist, List<LayerConstraint> allParamConstraints,
                                         List<LayerConstraint> weightConstraints, List<LayerConstraint> biasConstraints) {

        if (layer != null) {

//            if (useDropConnect && (Double.isNaN(dropOut) && (Double.isNaN(layer.getDropOut()))))
//                OneTimeLogger.warn(log, "Layer \"" + layerName
//                                + "\" dropConnect is set to true but dropout rate has not been added to configuration.");
//            if (useDropConnect && layer.getDropOut() == 0.0)
//                OneTimeLogger.warn(log,
//                                "Layer \"" + layerName + " dropConnect is set to true but dropout rate is set to 0.0");

            if (layer instanceof BaseLayer) {
                BaseLayer bLayer = (BaseLayer) layer;
                configureBaseLayer(layerName, bLayer, useDropConnect, iDropout, l2, l2Bias, l1,
                                l1Bias, dist);
            } else if (layer instanceof FrozenLayer && ((FrozenLayer) layer).getLayer() instanceof BaseLayer) {
                BaseLayer bLayer = (BaseLayer) ((FrozenLayer) layer).getLayer();
                configureBaseLayer(layerName, bLayer, useDropConnect, iDropout, l2, l2Bias, l1,
                                l1Bias, dist);
            }

            if(layer.getConstraints() == null || layer.constraints.isEmpty()) {
                List<LayerConstraint> allConstraints = new ArrayList<>();
                if (allParamConstraints != null && layer.initializer().paramKeys(layer).size() > 0) {
                    for (LayerConstraint c : allConstraints) {
                        LayerConstraint c2 = c.clone();
                        c2.setParams(new HashSet<>(layer.initializer().paramKeys(layer)));
                        allConstraints.add(c2);
                    }
                }

                if (weightConstraints != null && layer.initializer().weightKeys(layer).size() > 0) {
                    for (LayerConstraint c : weightConstraints) {
                        LayerConstraint c2 = c.clone();
                        c2.setParams(new HashSet<>(layer.initializer().weightKeys(layer)));
                        allConstraints.add(c2);
                    }
                }

                if (weightConstraints != null && layer.initializer().biasKeys(layer).size() > 0) {
                    for (LayerConstraint c : weightConstraints) {
                        LayerConstraint c2 = c.clone();
                        c2.setParams(new HashSet<>(layer.initializer().biasKeys(layer)));
                        allConstraints.add(c2);
                    }
                }

                layer.setConstraints(allConstraints);
            }
        }
    }

    private static void configureBaseLayer(String layerName, BaseLayer bLayer,  boolean useDropConnect,
                                           IDropout iDropout, Double l2, Double l2Bias, Double l1, Double l1Bias,
                    Distribution dist) {

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

        if(bLayer.getIDropout() == null){
            bLayer.setIDropout(iDropout);
        }


        if (bLayer.getWeightInit() == WeightInit.DISTRIBUTION) {
            if (dist != null && bLayer.getDist() == null)
                bLayer.setDist(dist);
            else if (dist == null && bLayer.getDist() == null) {
                bLayer.setDist(new NormalDistribution(0, 1));
                OneTimeLogger.warn(log, "Layer \"" + layerName
                                + "\" distribution is automatically set to normalize distribution with mean 0 and variance 1.");
            }
        } else if ((dist != null || bLayer.getDist() != null)) {
            OneTimeLogger.warn(log, "Layer \"" + layerName
                            + "\" distribution is set but will not be applied unless weight init is set to WeighInit.DISTRIBUTION.");
        }
    }
}
