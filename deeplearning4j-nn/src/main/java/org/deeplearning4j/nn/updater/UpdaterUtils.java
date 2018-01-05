package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.learning.config.IUpdater;

/**
 * Created by Alex on 14/04/2017.
 */
public class UpdaterUtils {


    public static boolean updaterConfigurationsEquals(Layer layer1, String param1, Layer layer2, String param2) {
        org.deeplearning4j.nn.conf.layers.Layer l1 = layer1.conf().getLayer();
        org.deeplearning4j.nn.conf.layers.Layer l2 = layer2.conf().getLayer();
        IUpdater u1 = l1.getUpdaterByParam(param1);
        IUpdater u2 = l2.getUpdaterByParam(param2);

        //For updaters to be equal (and hence combinable), we require that:
        //(a) The updater-specific configurations are equal (inc. LR, LR/momentum schedules etc)
        //(b) If one or more of the params are pretrainable params, they are in the same layer
        //    This last point is necessary as we don't want to modify the pretrain gradient/updater state during
        //    backprop, or modify the pretrain gradient/updater state of one layer while training another
        if (!u1.equals(u2)) {
            //Different updaters or different config
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
}
