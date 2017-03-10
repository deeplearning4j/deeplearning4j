package org.deeplearning4j.nn.updater.graph;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Gradient updater for ComputationGraph.<br>
 * Note: ComputationGraph does not implement the Layer interface (due to multiple in/out etc), hence ComputationGraphUpdater
 * can't be defined as an {@link Updater}.
 *
 * @author Alex Black
 */
public class ComputationGraphUpdater implements Serializable, Cloneable {

    private final Updater[] layerUpdaters;
    private final Map<String, Integer> layerUpdatersMap;
    private INDArray viewArray;

    public ComputationGraphUpdater(ComputationGraph graph) {
        layerUpdaters = new Updater[graph.getNumLayers()];
        layerUpdatersMap = new HashMap<>();

        int i = 0;
        int updaterStateSize = 0;
        for (Layer layer : graph.getLayers()) {
            Updater u = UpdaterCreator.getUpdater(layer);
            layerUpdaters[i] = u;
            layerUpdatersMap.put(layer.conf().getLayer().getLayerName(), i);
            updaterStateSize += layerUpdaters[i].stateSizeForLayer(layer);
            i++;
        }

        //Initialize the updater state
        if (updaterStateSize > 0) {
            //May be 0 if all SGD updaters, for example
            viewArray = Nd4j.createUninitialized(new int[] {1, updaterStateSize}, Nd4j.order());
        }
        int soFar = 0;
        i = 0;
        for (Layer layer : graph.getLayers()) {
            int thisSize = layerUpdaters[i].stateSizeForLayer(layer);
            if (thisSize == 0) {
                i++;
                continue;
            }
            INDArray view = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + thisSize));
            layerUpdaters[i++].setStateViewArray(layer, view, true);
            soFar += thisSize;
        }
    }

    public ComputationGraphUpdater(ComputationGraph graph, INDArray updaterState) {
        layerUpdatersMap = new HashMap<>();
        Layer[] layers = graph.getLayers();
        layerUpdaters = new Updater[layers.length];

        int updaterStateSize = 0;
        for (int i = 0; i < layers.length; i++) {
            layerUpdaters[i] = UpdaterCreator.getUpdater(layers[i]);
            updaterStateSize += layerUpdaters[i].stateSizeForLayer(layers[i]);
            layerUpdatersMap.put(layers[i].conf().getLayer().getLayerName(), i);
        }

        if (updaterState != null) {
            if (updaterState.length() != updaterStateSize) {
                throw new IllegalStateException("Expected updater state with size " + updaterStateSize + ", got size "
                                + updaterState.length());
            }
            //Assign subsets to the various updaters, without initializing (overwriting) the layer values
            this.viewArray = updaterState;
            int soFar = 0;
            for (int i = 0; i < layers.length; i++) {
                int thisSize = layerUpdaters[i].stateSizeForLayer(layers[i]);
                if (thisSize == 0)
                    continue;
                INDArray view = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + thisSize));
                layerUpdaters[i].setStateViewArray(layers[i], view, false);
                soFar += thisSize;
            }
        } else if (updaterStateSize != 0) {
            //Updater state size is non-zero, but we didn't get an array...
            throw new IllegalStateException(
                            "Expected updater state with size " + updaterStateSize + ", got null input");
        }
    }

    private ComputationGraphUpdater(int size, Map<String, Integer> layerUpdatersMap) {
        layerUpdaters = new Updater[size];
        this.layerUpdatersMap = layerUpdatersMap;
    }

    private ComputationGraphUpdater(ComputationGraphUpdater updater) {
        layerUpdaters = new Updater[updater.layerUpdaters.length];
        for (int i = 0; i < layerUpdaters.length; i++)
            layerUpdaters[i] = updater.layerUpdaters[i].clone();
        layerUpdatersMap = new HashMap<>(updater.layerUpdatersMap);
    }

    @Override
    public ComputationGraphUpdater clone() {
        return new ComputationGraphUpdater(this);
    }

    /**
     * Update the gradients for the given ComputationGraph
     */
    public void update(ComputationGraph graph, Gradient gradient, int iteration, int batchSize) {
        Map<String, Gradient> layerGradients = new HashMap<>();

        for (Map.Entry<String, INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
            String key = gradientPair.getKey();
            int idx = key.lastIndexOf('_');
            if (idx == -1)
                throw new IllegalStateException(
                                "Invalid key: ComputationGraph Gradient key does not have layer separator: \"" + key
                                                + "\"");

            String layerName = key.substring(0, idx);

            Gradient g = layerGradients.get(layerName);
            if (g == null) {
                g = new DefaultGradient();
                layerGradients.put(layerName, g);
            }

            String newKey = key.substring(idx + 1);
            g.setGradientFor(newKey, gradientPair.getValue());
        }

        for (Map.Entry<String, Gradient> entry : layerGradients.entrySet()) {
            String layerName = entry.getKey();
            int updaterIdx = layerUpdatersMap.get(layerName);
            layerUpdaters[updaterIdx].update(graph.getLayer(layerName), entry.getValue(), iteration, batchSize);


            //Gradients may be replaced by BaseUpdater.update()
            for (Map.Entry<String, INDArray> entry2 : layerGradients.get(layerName).gradientForVariable().entrySet()) {
                gradient.setGradientFor(entry.getKey() + "_" + entry2.getKey(), entry2.getValue());
            }
        }
    }


    public void setStateViewArray(INDArray viewArray) {
        if (this.viewArray.length() != viewArray.length())
            throw new IllegalStateException("Invalid input: view arrays differ in length. " + "Expected length "
                            + this.viewArray.length() + ", got length " + viewArray.length());
        this.viewArray.assign(viewArray);
    }


    public INDArray getStateViewArray() {
        return viewArray;
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof ComputationGraphUpdater))
            return false;
        return layerUpdatersMap.equals(((ComputationGraphUpdater) other).layerUpdatersMap);
    }

    @Override
    public int hashCode() {
        return layerUpdatersMap.hashCode();
    }
}
