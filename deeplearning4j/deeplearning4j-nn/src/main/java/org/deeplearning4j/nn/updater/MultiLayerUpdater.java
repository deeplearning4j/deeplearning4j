package org.deeplearning4j.nn.updater;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;

/**
 * MultiLayerUpdater: Gradient updater for MultiLayerNetworks.
 * Expects backprop gradients for all layers to be in single Gradient object,
 * keyed by "0_b", "1_w" etc., as per MultiLayerNetwork.backward()
 *
 * @author Alex Black
 */
@Getter
@Slf4j
public class MultiLayerUpdater extends BaseMultiLayerUpdater<MultiLayerNetwork> {

    public MultiLayerUpdater(MultiLayerNetwork network) {
        this(network, null);
    }

    public MultiLayerUpdater(MultiLayerNetwork network, INDArray updaterState) {
        super(network, updaterState);

        layersByName = new HashMap<>();
        Layer[] l = network.getLayers();
        for (int i = 0; i < l.length; i++) {
            layersByName.put(String.valueOf(i), l[i]);
        }
    }

    @Override
    protected Layer[] getOrderedLayers() {
        return network.getLayers();
    }

    @Override
    protected INDArray getFlattenedGradientsView() {
        if (network.getFlattenedGradients() == null) {
            network.initGradientsView();
        }
        return network.getFlattenedGradients();
    }

    @Override
    protected INDArray getParams() {
        return network.params();
    }

    @Override
    protected boolean isMiniBatch() {
        return network.conf().isMiniBatch();
    }

    @Override
    public Updater clone() {
        return new MultiLayerUpdater(network, null);
    }
}
