package org.deeplearning4j.nn.updater;

import com.google.common.base.Preconditions;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Updater for a single layer, excluding MultiLayerNetwork (which also implements the Layer interface)
 *
 * @author Alex Black
 */
@Slf4j
public class LayerUpdater extends BaseMultiLayerUpdater<Layer> {

    private final Layer[] layerArr;

    public LayerUpdater(Layer layer) {
        this(layer, null);
    }

    public LayerUpdater(Layer layer, INDArray updaterState){
        super(layer, updaterState);
        if(layer instanceof MultiLayerNetwork){
            throw new UnsupportedOperationException("Cannot use LayerUpdater for a MultiLayerNetwork");
        }

        this.layerArr = new Layer[]{layer};
    }

    @Override
    protected Layer[] getOrderedLayers() {
        return layerArr;
    }

    @Override
    protected INDArray getFlattenedGradientsView() {
        return network.getGradientsViewArray();
    }

    @Override
    protected boolean isMiniBatch() {
        return network.conf().isMiniBatch();
    }
}
