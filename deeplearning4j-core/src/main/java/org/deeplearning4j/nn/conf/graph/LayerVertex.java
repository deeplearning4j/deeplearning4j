package org.deeplearning4j.nn.conf.graph;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.factory.LayerFactories;

/**LayerVertex is a GraphVertex with a Layer configuration (and, optionally preprocessor) in it
 *
 */
@AllArgsConstructor @NoArgsConstructor @Data
public class LayerVertex extends GraphVertex {

    private NeuralNetConfiguration layerConf;
    private InputPreProcessor preProcessor;

    @Override
    public GraphVertex clone() {
        return new LayerVertex(layerConf.clone(),(preProcessor != null ? preProcessor.clone() : null));
    }

    @Override
    public boolean equals(Object o) {
        if(!(o instanceof LayerVertex)) return false;
        LayerVertex lv = (LayerVertex)o;
        if(!layerConf.equals(lv.layerConf)) return false;
        if(preProcessor == null && lv.preProcessor != null || preProcessor != null && lv.preProcessor == null) return false;
        return preProcessor == null || preProcessor.equals(lv.preProcessor);
    }

    @Override
    public int hashCode() {
        return layerConf.hashCode() ^ (preProcessor != null ? preProcessor.hashCode() : 0);
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx) {
        return new org.deeplearning4j.nn.graph.vertex.impl.LayerVertex(
                graph, name, idx,
                LayerFactories.getFactory(layerConf).create(layerConf, null, idx),
                preProcessor);
    }
}
