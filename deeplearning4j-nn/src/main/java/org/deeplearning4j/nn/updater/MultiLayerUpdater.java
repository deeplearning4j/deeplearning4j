package org.deeplearning4j.nn.updater;

import com.google.common.base.Preconditions;
import lombok.*;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * MultiLayerUpdater: Gradient updater for MultiLayerNetworks.
 * Expects backprop gradients for all layers to be in single Gradient object,
 * keyed by "0_b", "1_w" etc., as per MultiLayerNetwork.backward()
 */
@EqualsAndHashCode
@Getter
@Slf4j
public class MultiLayerUpdater implements Updater {

    private final List<UpdaterBlock> updaterBlocks;
    private final Updater[] layerUpdaters;
    private INDArray viewArray;

//    public MultiLayerUpdater(MultiLayerNetwork network) {
//        Layer[] layers = network.getLayers();
//        for (int i = 0; i < layers.length; i++) {
//            while (layers[i] == null)
//                layers = network.getLayers();
//        }
//        layerUpdaters = new Updater[layers.length];
//
//        int updaterStateSize = 0;
//        for (int i = 0; i < layers.length; i++) {
//            Layer layer = layers[i];
//            Preconditions.checkNotNull(layer);
//            layerUpdaters[i] = UpdaterCreator.getUpdater(layer);
//            updaterStateSize += layerUpdaters[i].stateSizeForLayer(layer);
//        }
//
//        //Initialize the updater state:
//        if (updaterStateSize > 0) {
//            //May be 0 if all SGD updaters, for example
//            viewArray = Nd4j.createUninitialized(new int[] {1, updaterStateSize}, Nd4j.order());
//        }
//        int soFar = 0;
//        for (int i = 0; i < layers.length; i++) {
//            int thisSize = layerUpdaters[i].stateSizeForLayer(layers[i]);
//            if (thisSize == 0)
//                continue;
//            INDArray view = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + thisSize));
//            layerUpdaters[i].setStateViewArray(layers[i], view, true);
//            soFar += thisSize;
//        }
//    }

    public MultiLayerUpdater(MultiLayerNetwork network) {
        Layer[] layers = network.getLayers();

        int updaterStateSize = 0;
        //Iterate through layers, and variables for each layer.
        //While the updater configuration is the same: combine
        Layer lastLayer = null;
        String lastVariable = null;
        UpdaterBlock currentBlock = null;
        updaterBlocks = new ArrayList<>();
        int currentParamOffset = 0;
        int currentUpdaterOffset = 0;

        INDArray paramsView = network.params();
        INDArray gradientView = network.getFlattenedGradients();
        int paramsViewSoFar = 0;
        for( int i=0; i<layers.length; i++ ){
            Map<String,INDArray> layerParamTable = layers[i].paramTable();
            List<String> variables = new ArrayList<>(layerParamTable.keySet());    //Is a set, but iteration order should be fixed per layer as it's a from a LinkedHashSet
            for( int j=0; j<variables.size(); j++ ){
                String var = variables.get(j);
                int paramSizeThisVariable = layerParamTable.get(var).length();
                int updaterStateSizeThisVariable = UpdaterUtils.stateSizeForLayerVariable(layers[i], var);

                INDArray gradientViewSubset = null;
                INDArray paramsViewSubset = null;
                if(paramSizeThisVariable > 0) {
                    paramsViewSubset = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(paramsViewSoFar, paramsViewSoFar + paramSizeThisVariable));
                    gradientViewSubset = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(paramsViewSoFar, paramsViewSoFar + paramSizeThisVariable));
                }

                //First: decide whether to add to the existing updater block, or create a new one
                if(currentBlock == null || !UpdaterUtils.updaterConfigurationsEquals(lastLayer, lastVariable, layers[i], var)){
                    List<UpdaterBlock.VarState> list = new ArrayList<>();
                    list.add(new UpdaterBlock.VarState(layers[i], var, paramsViewSubset, gradientViewSubset));
                    currentBlock = new UpdaterBlock(currentParamOffset, currentParamOffset+paramSizeThisVariable,
                            currentUpdaterOffset, currentUpdaterOffset+updaterStateSizeThisVariable, list);

                    updaterBlocks.add(currentBlock);
                } else {
                    //Add to existing updater block
                    currentBlock.setParamOffsetEnd( currentBlock.getParamOffsetEnd() + paramSizeThisVariable);
                    currentBlock.setUpdaterViewOffsetEnd( currentBlock.getUpdaterViewOffsetEnd() + updaterStateSizeThisVariable);
                    currentBlock.getLayersAndVariablesInBlock().add(
                            new UpdaterBlock.VarState(layers[i], var, paramsViewSubset, gradientViewSubset));
                }

                lastLayer = layers[i];
                lastVariable = variables.get(j);
                updaterStateSize += updaterStateSizeThisVariable;
                paramsViewSoFar += paramSizeThisVariable;
            }
        }


        layerUpdaters = new Updater[layers.length];

        //Initialize the updater state:
        if (updaterStateSize > 0) {
            //May be 0 if all SGD updaters, for example
            viewArray = Nd4j.createUninitialized(new int[] {1, updaterStateSize}, Nd4j.order());
        }

        //Set up the updaters, for the updater blocks:

        int updaterViewSoFar = 0;
        paramsViewSoFar = 0;
        for( int i=0; i<updaterBlocks.size(); i++ ){
            UpdaterBlock ub = updaterBlocks.get(i);

            int viewStateSize = ub.getUpdaterViewOffsetEnd() - ub.getUpdaterViewOffsetStart();
            int gradSize = ub.getParamOffsetEnd() - ub.getParamOffsetStart();

            if(viewStateSize > 0) {
                INDArray updaterViewSubset = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(updaterViewSoFar, updaterViewSoFar + viewStateSize));
                ub.setUpdaterView(updaterViewSubset);
            }

            if(gradSize > 0) {
                INDArray gradientViewSubset = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(paramsViewSoFar, paramsViewSoFar + gradSize));
                ub.setGradientView(gradientViewSubset);
            }

            updaterViewSoFar += viewStateSize;
            paramsViewSoFar += gradSize;
        }

//        int updaterStateSize = 0;
//        for (int i = 0; i < layers.length; i++) {
//            Layer layer = layers[i];
//            Preconditions.checkNotNull(layer);
//            layerUpdaters[i] = UpdaterCreator.getUpdater(layer);
//            updaterStateSize += layerUpdaters[i].stateSizeForLayer(layer);
//        }

//        int soFar = 0;
//        for (int i = 0; i < layers.length; i++) {
//            int thisSize = layerUpdaters[i].stateSizeForLayer(layers[i]);
//            if (thisSize == 0)
//                continue;
//            INDArray view = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + thisSize));
//            layerUpdaters[i].setStateViewArray(layers[i], view, true);
//            soFar += thisSize;
//        }

        System.out.println();
    }



    public MultiLayerUpdater(MultiLayerNetwork network, INDArray updaterState) {
        throw new UnsupportedOperationException("Not yet re-implemented");
//        Layer[] layers = network.getLayers();
//        layerUpdaters = new Updater[layers.length];
//
//        int updaterStateSize = 0;
//        for (int i = 0; i < layers.length; i++) {
//            layerUpdaters[i] = UpdaterCreator.getUpdater(layers[i]);
//            updaterStateSize += layerUpdaters[i].stateSizeForLayer(layers[i]);
//        }
//
//        if (updaterState != null) {
//            if (updaterState.length() != updaterStateSize) {
//                throw new IllegalStateException("Expected updater state with size " + updaterStateSize + ", got size "
//                                + updaterState.length());
//            }
//            //Assign subsets to the various updaters, without initializing (overwriting) the layer values
//            this.viewArray = updaterState;
//            int soFar = 0;
//            for (int i = 0; i < layers.length; i++) {
//                int thisSize = layerUpdaters[i].stateSizeForLayer(layers[i]);
//                if (thisSize == 0)
//                    continue;
//                INDArray view = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + thisSize));
//                layerUpdaters[i].setStateViewArray(layers[i], view, false);
//                soFar += thisSize;
//            }
//        } else if (updaterStateSize != 0) {
//            //Updater state size is non-zero, but we didn't get an array...
//            throw new IllegalStateException(
//                            "Expected updater state with size " + updaterStateSize + ", got null input");
//        }
    }

    @Override
    public void setStateViewArray(Layer layer, INDArray viewArray, boolean initialize) {
        if (this.viewArray.length() != viewArray.length())
            throw new IllegalStateException("Invalid input: view arrays differ in length. " + "Expected length "
                            + this.viewArray.length() + ", got length " + viewArray.length());
        this.viewArray.assign(viewArray);
    }

    @Override
    public INDArray getStateViewArray() {
        return viewArray;
    }

    @Override
    public int stateSizeForLayer(Layer layer) {
        if (!(layer instanceof MultiLayerNetwork))
            throw new IllegalArgumentException("Expected MultiLayerNetwork");

        return viewArray.length();
    }

    @Override
    public void update(Layer layer, Gradient gradient, int iteration, int batchSize) {
        MultiLayerNetwork mln = (MultiLayerNetwork) layer;

        for(UpdaterBlock ub : updaterBlocks){
            ub.update(iteration, batchSize);
        }

//        Gradient[] layerGradients = new Gradient[layerUpdaters.length];
//
//
//        for (int i = 0; i < layerGradients.length; i++)
//            layerGradients[i] = new DefaultGradient();
//
//
//        for (Map.Entry<String, INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
//            String key = gradientPair.getKey();
//            int idx = key.indexOf('_');
//            if (idx == -1)
//                throw new IllegalStateException(
//                        "Invalid key: MuliLayerNetwork Gradient key does not have layer separator: \"" + key
//                                + "\"");
//            int layerIdx = Integer.parseInt(key.substring(0, idx));
//
//            String newKey = key.substring(idx + 1);
//            layerGradients[layerIdx].gradientForVariable().put(newKey, gradientPair.getValue());
//        }
//
//        //First: update using the UpdaterBlocks
//
//        for (int i = 0; i < layerUpdaters.length; i++) {
//            layerUpdaters[i].update(mln.getLayer(i), layerGradients[i], iteration, batchSize);
//        }
    }

    @Override
    public Updater clone() {
        throw new UnsupportedOperationException("Not yet implemented");
        //        return new MultiLayerUpdater(this);
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof MultiLayerUpdater))
            return false;

        MultiLayerUpdater multiLayerUpdater = (MultiLayerUpdater) other;
        if (layerUpdaters.length != multiLayerUpdater.layerUpdaters.length)
            return false;

        for (int i = 0; i < layerUpdaters.length; i++) {
            if (!layerUpdaters[i].equals(multiLayerUpdater.layerUpdaters[i]))
                return false;
        }
        return true;
    }
}
