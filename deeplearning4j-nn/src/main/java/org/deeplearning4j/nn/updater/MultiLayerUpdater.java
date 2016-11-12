package org.deeplearning4j.nn.updater;

import com.google.common.base.Preconditions;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Map;

/**
 * MultiLayerUpdater: Gradient updater for MultiLayerNetworks.
 * Expects backprop gradients for all layers to be in single Gradient object,
 * keyed by "0_b", "1_w" etc., as per MultiLayerNetwork.backward()
 */
@EqualsAndHashCode @Getter
public class MultiLayerUpdater implements Updater {
    private final Updater[] layerUpdaters;
    private INDArray viewArray;

    public MultiLayerUpdater(MultiLayerNetwork network) {
        Layer[] layers = network.getLayers();
        for(int i = 0; i < layers.length; i++) {
           while(layers[i] == null)
               layers = network.getLayers();
        }
        layerUpdaters = new Updater[layers.length];

        int updaterStateSize = 0;
        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            Preconditions.checkNotNull(layer);
            layerUpdaters[i] = UpdaterCreator.getUpdater(layer);
            updaterStateSize += layerUpdaters[i].stateSizeForLayer(layer);
        }

        //Initialize the updater state:
        if(updaterStateSize > 0) {
            //May be 0 if all SGD updaters, for example
            viewArray = Nd4j.createUninitialized(new int[]{1, updaterStateSize}, Nd4j.order());
        }
        int soFar = 0;
        for (int i = 0; i < layers.length; i++) {
            int thisSize = layerUpdaters[i].stateSizeForLayer(layers[i]);
            if(thisSize == 0) continue;
            INDArray view = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar,soFar+thisSize));
            layerUpdaters[i].setStateViewArray(layers[i], view, true);
            soFar += thisSize;
        }
    }

    public MultiLayerUpdater(MultiLayerNetwork network, INDArray updaterState){
        Layer[] layers = network.getLayers();
        layerUpdaters = new Updater[layers.length];

        int updaterStateSize = 0;
        for (int i = 0; i < layers.length; i++) {
            layerUpdaters[i] = UpdaterCreator.getUpdater(layers[i]);
            updaterStateSize += layerUpdaters[i].stateSizeForLayer(layers[i]);
        }

        if(updaterState != null){
            if(updaterState.length() != updaterStateSize){
                throw new IllegalStateException("Expected updater state with size " + updaterStateSize + ", got size " + updaterState.length());
            }
            //Assign subsets to the various updaters, without initializing (overwriting) the layer values
            this.viewArray = updaterState;
            int soFar = 0;
            for (int i = 0; i < layers.length; i++) {
                int thisSize = layerUpdaters[i].stateSizeForLayer(layers[i]);
                if(thisSize == 0) continue;
                INDArray view = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar,soFar+thisSize));
                layerUpdaters[i].setStateViewArray(layers[i], view, false);
                soFar += thisSize;
            }
        } else if(updaterStateSize != 0){
            //Updater state size is non-zero, but we didn't get an array...
            throw new IllegalStateException("Expected updater state with size " + updaterStateSize + ", got null input");
        }
    }

    @Override
    public void setStateViewArray(Layer layer, INDArray viewArray, boolean initialize){
        if(this.viewArray.length() != viewArray.length()) throw new IllegalStateException("Invalid input: view arrays differ in length. " +
                "Expected length " + this.viewArray.length() + ", got length " + viewArray.length());
        this.viewArray.assign(viewArray);
    }

    @Override
    public INDArray getStateViewArray(){
        return viewArray;
    }

    @Override
    public int stateSizeForLayer(Layer layer) {
        if(!(layer instanceof MultiLayerNetwork)) throw new IllegalArgumentException("Expected MultiLayerNetwork");

        return viewArray.length();
    }

    @Override
    public void update(Layer layer, Gradient gradient, int iteration, int batchSize) {
        MultiLayerNetwork mln = (MultiLayerNetwork) layer;

        Gradient[] layerGradients = new Gradient[layerUpdaters.length];
        for (int i = 0; i < layerGradients.length; i++)
            layerGradients[i] = new DefaultGradient();

        for (Map.Entry<String, INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
            String key = gradientPair.getKey();
            int idx = key.indexOf('_');
            if (idx == -1)
                throw new IllegalStateException("Invalid key: MuliLayerNetwork Gradient key does not have layer separator: \"" + key + "\"");
            int layerIdx = Integer.parseInt(key.substring(0, idx));

            String newKey = key.substring(idx + 1);
            layerGradients[layerIdx].gradientForVariable().put(newKey, gradientPair.getValue());
        }

        for (int i = 0; i < layerUpdaters.length; i++) {
            layerUpdaters[i].update(mln.getLayer(i), layerGradients[i], iteration, batchSize);
        }
    }

    @Override
    public Updater clone() {
        throw new UnsupportedOperationException("Not yet implemented");
//        return new MultiLayerUpdater(this);
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof MultiLayerUpdater)) return false;

        MultiLayerUpdater multiLayerUpdater = (MultiLayerUpdater) other;
        if (layerUpdaters.length != multiLayerUpdater.layerUpdaters.length) return false;

        for (int i = 0; i < layerUpdaters.length; i++) {
            if (!layerUpdaters[i].equals(multiLayerUpdater.layerUpdaters[i])) return false;
        }
        return true;
    }
}
