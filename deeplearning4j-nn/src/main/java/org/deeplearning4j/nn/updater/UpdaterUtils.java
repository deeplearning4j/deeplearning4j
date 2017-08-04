package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.nd4j.linalg.learning.*;
import org.nd4j.linalg.learning.config.*;

import java.util.Objects;

/**
 * Created by Alex on 14/04/2017.
 */
public class UpdaterUtils {


    public static boolean updaterConfigurationsEquals(Layer layer1, String param1, Layer layer2, String param2) {
        org.deeplearning4j.nn.conf.layers.Layer l1 = layer1.conf().getLayer();
        org.deeplearning4j.nn.conf.layers.Layer l2 = layer2.conf().getLayer();
        IUpdater u1 = l1.getIUpdaterByParam(param1);
        IUpdater u2 = l2.getIUpdaterByParam(param2);
        if (!u1.equals(u2)) {
            //Different updaters or different config
            return false;
        }
        //For updaters to be equal (and hence combinable), we require that:
        //(a) The updater-specific configurations are equal (inc. LR)
        //(b) The learning rate *schedules* are equal
        //(c) If one or more of the params are pretrainable params, they are in the same layer
        //    This last point is necessary as we don't want to modify the pretrain gradient/updater state during
        //    backprop, or modify the pretrain gradient/updater state of one layer while training another
        if (!lrSchedulesEqual(layer1, param1, layer2, param2)) {
            return false;
        }

        boolean isPretrainParam1 = layer1.conf().getLayer().isPretrainParam(param1);
        boolean isPretrainParam2 = layer2.conf().getLayer().isPretrainParam(param2);
        if (isPretrainParam1 || isPretrainParam2) {
            //One or both of params are pretrainable.
            //Either layers differ -> don't want to combine a pretrain updaters across layers
            //Or one is pretrain and the other isn't -> don't want to combine pretrain updaters within a layer
            return layer1 == layer2 && isPretrainParam1 && isPretrainParam2;
        }

        return true;
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
                BaseLayer bl1 = (BaseLayer) layer1.conf().getLayer();
                BaseLayer bl2 = (BaseLayer) layer2.conf().getLayer();
                lrConfigEqual = Objects.equals(bl1.getLearningRateSchedule(), bl2.getLearningRateSchedule());
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
