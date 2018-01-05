package org.deeplearning4j.nn.updater.graph;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.updater.BaseMultiLayerUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;

/**
 * Gradient updater for ComputationGraph. Most of the functionality is shared with
 * {@link org.deeplearning4j.nn.updater.MultiLayerUpdater} via {@link BaseMultiLayerUpdater}
 *
 * @author Alex Black
 */
public class ComputationGraphUpdater extends BaseMultiLayerUpdater<ComputationGraph> {

    protected Layer[] orderedLayers;

    public ComputationGraphUpdater(ComputationGraph graph) {
        this(graph, null);
    }

    public ComputationGraphUpdater(ComputationGraph graph, INDArray updaterState) {
        super(graph, updaterState);

        layersByName = new HashMap<>();
        Layer[] layers = getOrderedLayers();
        for (Layer l : layers) {
            layersByName.put(l.conf().getLayer().getLayerName(), l);
        }
    }

    @Override
    protected Layer[] getOrderedLayers() {
        if (orderedLayers != null) {
            return orderedLayers;
        }
        GraphVertex[] vertices = network.getVertices();

        //In CompGraph: we need to know topological ordering, so we know how parameters are laid out in the 1d view arrays
        int[] topologicalOrdering = network.topologicalSortOrder();

        Layer[] out = new Layer[network.getNumLayers()];

        int j = 0;
        for (int i = 0; i < topologicalOrdering.length; i++) {
            GraphVertex currentVertex = vertices[topologicalOrdering[i]];
            if (!currentVertex.hasLayer()) {
                continue;
            }

            out[j++] = currentVertex.getLayer();
        }

        orderedLayers = out;
        return orderedLayers;
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
}
