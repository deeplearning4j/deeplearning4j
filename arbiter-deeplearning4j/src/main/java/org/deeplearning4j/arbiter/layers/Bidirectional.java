package org.deeplearning4j.arbiter.layers;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.List;

/**
 * Bidirectional layer wrapper. Can be used wrap an existing layer space, in the same way that
 * {@link org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional} wraps a DL4J layer
 *
 * @author Alex Black
 */
@NoArgsConstructor  //JSON
@Data
public class Bidirectional extends LayerSpace<Layer> {

    protected LayerSpace<?> layerSpace;

    public Bidirectional(LayerSpace<?> layerSpace){
        this.layerSpace = layerSpace;
    }

    @Override
    public Layer getValue(double[] parameterValues) {
        Layer underlying = layerSpace.getValue(parameterValues);
        return new org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional(underlying);
    }

    @Override
    public int numParameters() {
        return layerSpace.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return layerSpace.collectLeaves();
    }

    @Override
    public boolean isLeaf() {
        return layerSpace.isLeaf();
    }

    @Override
    public void setIndices(int... indices) {
        layerSpace.setIndices(indices);
    }
}
