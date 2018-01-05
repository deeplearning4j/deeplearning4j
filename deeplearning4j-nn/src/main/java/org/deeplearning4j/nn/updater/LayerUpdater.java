package org.deeplearning4j.nn.updater;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;

/**
 * Updater for a single layer, excluding MultiLayerNetwork (which also implements the Layer interface)
 *
 * @author Alex Black
 */
@Slf4j
public class LayerUpdater extends BaseMultiLayerUpdater<Layer> {

    public LayerUpdater(Layer layer) {
        this(layer, null);
    }

    public LayerUpdater(Layer layer, INDArray updaterState) {
        super(layer, updaterState);
        if (layer instanceof MultiLayerNetwork) {
            throw new UnsupportedOperationException("Cannot use LayerUpdater for a MultiLayerNetwork");
        }

        layersByName = new HashMap<>();
        layersByName.put(layer.conf().getLayer().getLayerName(), layer);
    }

    @Override
    protected Layer[] getOrderedLayers() {
        return new Layer[] {network};
    }

    @Override
    protected INDArray getFlattenedGradientsView() {
        return network.getGradientsViewArray();
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
    protected boolean isSingleLayerUpdater() {
        return true;
    }
}
