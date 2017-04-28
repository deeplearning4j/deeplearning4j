package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.nd4j.linalg.learning.*;

import java.util.Objects;

/**
 * Created by Alex on 14/04/2017.
 */
public class UpdaterUtils {

    public static GradientUpdater getGradientUpdater(Layer layer, String variable) {
        org.deeplearning4j.nn.conf.Updater u = layer.conf().getLayer().getUpdaterByParam(variable);
        switch (u) {
            case SGD:
                return new org.nd4j.linalg.learning.Sgd(layer.conf().getLearningRateByParam(variable));
            case ADAM:
                return new Adam(layer.conf().getLearningRateByParam(variable),
                                layer.conf().getLayer().getAdamMeanDecay(), layer.conf().getLayer().getAdamVarDecay(),
                                layer.conf().getLayer().getEpsilon());
            case ADADELTA:
                return new AdaDelta(layer.conf().getLayer().getRho(), layer.conf().getLayer().getEpsilon());
            case NESTEROVS:
                return new Nesterovs(layer.conf().getLayer().getMomentum(),
                                layer.conf().getLearningRateByParam(variable));
            case ADAGRAD:
                return new AdaGrad(layer.conf().getLearningRateByParam(variable), layer.conf().getLayer().getEpsilon());
            case RMSPROP:
                return new org.nd4j.linalg.learning.RmsProp(layer.conf().getLearningRateByParam(variable),
                                layer.conf().getLayer().getRmsDecay(), layer.conf().getLayer().getEpsilon());
            case NONE:
                return new NoOpUpdater();
            case CUSTOM:
                throw new UnsupportedOperationException("Custom updaters: not yet implemented");
            default:
                throw new IllegalArgumentException("Unknown updater: " + u);
        }
    }

    public static int stateSizeForLayerVariable(Layer layer, String variable) {
        switch (layer.conf().getLayer().getUpdaterByParam(variable)) {
            case SGD:
            case NONE:
                return 0;

            case NESTEROVS:
            case ADAGRAD:
            case RMSPROP:
                return layer.getParam(variable).length();

            case ADAM:
            case ADADELTA:
                return 2 * layer.getParam(variable).length();

            default:
                throw new UnsupportedOperationException(
                                "Unknown updater: " + layer.conf().getLayer().getUpdaterByParam(variable));
        }
    }


    public static boolean updaterConfigurationsEquals(Layer layer1, String param1, Layer layer2, String param2) {
        org.deeplearning4j.nn.conf.layers.Layer l1 = layer1.conf().getLayer();
        org.deeplearning4j.nn.conf.layers.Layer l2 = layer2.conf().getLayer();
        org.deeplearning4j.nn.conf.Updater u1 = l1.getUpdaterByParam(param1);
        org.deeplearning4j.nn.conf.Updater u2 = l2.getUpdaterByParam(param2);
        if (u1 != u2) {
            //Different updaters
            return false;
        }

        //For updaters to be equal (and hence combinable), we require that:
        //(a) The learning rates are equal
        //(b) The learning rate *schedules* are equal
        //(c) The updater-specific configurations are equal
        //(d) If one or more of the params are pretrainable params, they are in the same layer
        //    This last point is necessary as we don't want to modify the pretrain gradient/updater state during
        //    backprop, or modify the pretrain gradient/updater state of one layer while training another

        double lr1 = layer1.conf().getLearningRateByParam(param1);
        double lr2 = layer2.conf().getLearningRateByParam(param2);
        if (lr1 != lr2) {
            return false;
        }

        if (!lrSchedulesEqual(layer1, param1, layer2, param2)) {
            return false;
        }

        boolean updaterConfigEqual;
        switch (u1) {
            case SGD: //Already checked LR and schedules
            case NONE:
                updaterConfigEqual = true;
                break;
            case ADAM:
                //Mean decay, var decay, epsilon
                updaterConfigEqual = l1.getAdamMeanDecay() == l2.getAdamMeanDecay()
                                && l1.getAdamVarDecay() == l2.getAdamVarDecay() && l1.getEpsilon() == l2.getEpsilon();
                break;
            case ADADELTA:
                updaterConfigEqual = l1.getRho() == l2.getRho() && l1.getEpsilon() == l2.getEpsilon();
                break;
            case NESTEROVS:
                updaterConfigEqual = l1.getMomentum() == l2.getMomentum()
                                && Objects.equals(l1.getMomentumSchedule(), l2.getMomentumSchedule());
                break;
            case ADAGRAD:
                updaterConfigEqual = l1.getEpsilon() == l2.getEpsilon();
                break;
            case RMSPROP:
                updaterConfigEqual = l1.getRmsDecay() == l2.getRmsDecay() && l1.getEpsilon() == l2.getEpsilon();
                break;
            case CUSTOM:
                throw new UnsupportedOperationException("Custom updaters not yet supported");
            default:
                throw new UnsupportedOperationException("Unknown updater: " + u1);
        }

        boolean isPretrainParam1 = layer1.conf().getLayer().isPretrainParam(param1);
        boolean isPretrainParam2 = layer2.conf().getLayer().isPretrainParam(param2);
        if (isPretrainParam1 || isPretrainParam2) {
            //One or both of params are pretrainable.
            if (layer1 == layer2 && isPretrainParam1 && isPretrainParam2) {
                return updaterConfigEqual;
            } else {
                //Either layers differ -> don't want to combine a pretrain updaters across layers
                //Or one is pretrain and the other isn't -> don't want to combine pretrain updaters within a layer
                return false;
            }
        }

        return updaterConfigEqual;
    }

    public static boolean lrSchedulesEqual(Layer layer1, String param1, Layer layer2, String param2) {

        LearningRatePolicy lp1 = layer1.conf().getLearningRatePolicy();
        LearningRatePolicy lp2 = layer2.conf().getLearningRatePolicy();

        if (lp1 != lp2) {
            return false;
        }

        double lr1 = layer1.conf().getLearningRateByParam(param1);
        double lr2 = layer2.conf().getLearningRateByParam(param2);
        if (lr1 != lr2) {
            return false;
        }

        double dr1 = layer1.conf().getLrPolicyDecayRate();
        double dr2 = layer2.conf().getLrPolicyDecayRate();

        boolean lrConfigEqual;
        switch (lp1) {
            case None:
                lrConfigEqual = true;
                break;
            case Exponential:
                lrConfigEqual = dr1 == dr2;
                break;
            case Inverse:
                lrConfigEqual = dr1 == dr2 && layer1.conf().getLrPolicyPower() == layer2.conf().getLrPolicyPower();
                break;
            case Poly:
                lrConfigEqual = layer1.conf().getLrPolicyPower() == layer2.conf().getLrPolicyPower();
                break;
            case Sigmoid:
                lrConfigEqual = dr1 == dr2 && layer1.conf().getLrPolicySteps() == layer2.conf().getLrPolicySteps();
                break;
            case Step:
                lrConfigEqual = dr1 == dr2 && layer1.conf().getLrPolicySteps() == layer2.conf().getLrPolicySteps();
                break;
            case TorchStep:
                lrConfigEqual = layer1.conf().getLrPolicyPower() == layer2.conf().getLrPolicyPower();
                break;
            case Schedule:
                lrConfigEqual = Objects.equals(layer1.conf().getLayer().getLearningRateSchedule(),
                                layer2.conf().getLayer().getLearningRateSchedule());
                break;
            case Score:
                //TODO - might be ok sometimes??
                lrConfigEqual = false;
                break;
            default:
                throw new UnsupportedOperationException("Unknown learning rate schedule: " + lp1);
        }

        return lrConfigEqual;
    }

}
