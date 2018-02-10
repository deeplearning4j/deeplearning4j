package org.deeplearning4j.arbiter.layers.fixed;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.layers.LayerSpace;
import org.deeplearning4j.nn.conf.layers.Layer;

@AllArgsConstructor
public class FixedLayerSpace<T extends Layer> extends LayerSpace<T> {

    protected T layer;

    @Override
    public T getValue(double[] parameterValues) {
        return (T)layer.clone();
    }
}
