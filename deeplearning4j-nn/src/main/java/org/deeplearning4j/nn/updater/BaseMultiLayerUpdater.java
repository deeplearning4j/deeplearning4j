package org.deeplearning4j.nn.updater;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Norm2;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * MultiLayerUpdater: Gradient updater for MultiLayerNetworks.
 * Expects backprop gradients for all layers to be in single Gradient object,
 * keyed by "0_b", "1_w" etc., as per MultiLayerNetwork.backward()
 */
@Getter
public abstract class BaseMultiLayerUpdater<T extends Model> implements Updater {

    protected final T network;
    protected Map<String,Layer> layersByName;
    protected final List<UpdaterBlock> updaterBlocks;
    protected INDArray updaterStateViewArray;

    public BaseMultiLayerUpdater(T network) {
        this(network, null);
    }

    public BaseMultiLayerUpdater(T network, INDArray updaterState) {
        this.network = network;
        Layer[] layers = getOrderedLayers();

        int updaterStateSize = 0;
        //Iterate through layers, and variables for each layer.
        //While the updater configuration is the same: combine into one op, rather than doing a lot of smaller
        // (yet identical) ops.
        Layer lastLayer = null;
        String lastVariable = null;
        UpdaterBlock currentBlock = null;
        updaterBlocks = new ArrayList<>();
        int currentParamOffset = 0;
        int currentUpdaterOffset = 0;

        INDArray paramsView = network.params();
        INDArray gradientView = getFlattenedGradientsView();
        int paramsViewSoFar = 0;
        for( int i=0; i<layers.length; i++ ){
            Map<String,INDArray> layerParamTable = layers[i].paramTable();
            List<String> variables = new ArrayList<>(layerParamTable.keySet());    //Is from a set, but iteration order should be fixed per layer as it's a from a LinkedHashSet
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
                    //Create a new block
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

        //Initialize the updater state, if required
        boolean updaterRequiresInit = false;
        if(updaterState != null){
            updaterStateViewArray = updaterState;
            updaterRequiresInit = false;
        } else if (updaterStateSize > 0) {
            //May be 0 if all SGD updaters, for example
            updaterStateViewArray = Nd4j.createUninitialized(new int[] {1, updaterStateSize}, Nd4j.order());
            updaterRequiresInit = true;
        }

        //Create and set up the updaters, for the updater blocks:
        int updaterViewSoFar = 0;
        paramsViewSoFar = 0;
        for( int i=0; i<updaterBlocks.size(); i++ ){
            UpdaterBlock ub = updaterBlocks.get(i);

            int viewStateSize = ub.getUpdaterViewOffsetEnd() - ub.getUpdaterViewOffsetStart();
            int gradSize = ub.getParamOffsetEnd() - ub.getParamOffsetStart();

            if(viewStateSize > 0) {
                INDArray updaterViewSubset = updaterStateViewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(updaterViewSoFar, updaterViewSoFar + viewStateSize));
                ub.setUpdaterView(updaterViewSubset);
                ub.setUpdaterViewRequiresInitialization(updaterRequiresInit);
            }

            if(gradSize > 0) {
                INDArray gradientViewSubset = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(paramsViewSoFar, paramsViewSoFar + gradSize));
                ub.setGradientView(gradientViewSubset);
            }

            updaterViewSoFar += viewStateSize;
            paramsViewSoFar += gradSize;
        }
    }

    protected abstract Layer[] getOrderedLayers();

    protected abstract INDArray getFlattenedGradientsView();

    protected abstract boolean isMiniBatch();

    public void setStateViewArray(INDArray viewArray){
        if (this.updaterStateViewArray.length() != viewArray.length())
            throw new IllegalStateException("Invalid input: view arrays differ in length. " + "Expected length "
                    + this.updaterStateViewArray.length() + ", got length " + viewArray.length());
        this.updaterStateViewArray.assign(viewArray);
    }

    @Override
    public void setStateViewArray(Layer layer, INDArray viewArray, boolean initialize) {
        this.setStateViewArray(viewArray);
    }

    @Override
    public INDArray getStateViewArray() {
        return updaterStateViewArray;
    }

    @Override
    public int stateSizeForLayer(Layer layer) {
        if (!(layer instanceof MultiLayerNetwork))
            throw new IllegalArgumentException("Expected MultiLayerNetwork");

        return updaterStateViewArray.length();
    }

    @Override
    public void update(Layer layer, Gradient gradient, int iteration, int batchSize) {
        update((T)layer, gradient, iteration, batchSize);
    }

    public void update(T model, Gradient gradient, int iteration, int batchSize ){

        //Split up the gradients on a per-layer basis, for pre-apply
        Map<String, Gradient> layerGradients = new HashMap<>();
        for (Map.Entry<String, INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
            String key = gradientPair.getKey();
            int idx = key.lastIndexOf('_');
            if (idx == -1)
                throw new IllegalStateException(
                        "Invalid key: Gradient key does not have layer separator: \"" + key+ "\"");
            String layerName = key.substring(0, idx);

            Gradient g = layerGradients.get(layerName);
            if (g == null) {
                g = new DefaultGradient();
                layerGradients.put(layerName, g);
            }

            String newKey = key.substring(idx + 1);
            g.setGradientFor(newKey, gradientPair.getValue());
        }

        //PRE apply (gradient clipping, etc): done on a per-layer basis
        for (Map.Entry<String, Gradient> entry : layerGradients.entrySet()) {
            String layerName = entry.getKey();
            Layer layer = layersByName.get(layerName);

            preApply(layer, layerGradients.get(layerName), iteration);
        }


        //Apply the updaters in blocks. This also applies LR and momentum schedules, L1 and L2
        for(UpdaterBlock ub : updaterBlocks){
            ub.update(iteration);
        }

        //Divide by minibatch size if necessary
        if(isMiniBatch()){
            getFlattenedGradientsView().divi(batchSize);
        }
    }

    public void preApply(Layer layer, Gradient gradient, int iteration) {

        GradientNormalization normalization = layer.conf().getLayer().getGradientNormalization();
        if (normalization == null || normalization == GradientNormalization.None || layer.conf().isPretrain())
            return; //no op

        final double threshold = layer.conf().getLayer().getGradientNormalizationThreshold();

        switch (normalization) {
            case RenormalizeL2PerLayer:
                double sumSquares = 0.0;
                for (INDArray g : gradient.gradientForVariable().values()) {
                    double l2 = g.norm2Number().doubleValue();
                    //l2 norm: sqrt(sum_i g_i^2)
                    sumSquares += l2 * l2;
                }
                double layerL2 = FastMath.sqrt(sumSquares);
                for (INDArray g : gradient.gradientForVariable().values()) {
                    g.divi(layerL2);
                }
                break;
            case RenormalizeL2PerParamType:
                for (INDArray g : gradient.gradientForVariable().values()) {
                    double l2 = Nd4j.getExecutioner().execAndReturn(new Norm2(g)).getFinalResult().doubleValue();
                    g.divi(l2);
                }
                break;
            case ClipElementWiseAbsoluteValue:
                for (INDArray g : gradient.gradientForVariable().values()) {
                    BooleanIndexing.replaceWhere(g, threshold, Conditions.greaterThan(threshold));
                    BooleanIndexing.replaceWhere(g, -threshold, Conditions.lessThan(-threshold));
                }
                break;
            case ClipL2PerLayer:
                double sumSquares2 = 0.0;
                for (INDArray g : gradient.gradientForVariable().values()) {
                    double l2 = Nd4j.getExecutioner().execAndReturn(new Norm2(g)).getFinalResult().doubleValue();
                    //l2 norm: sqrt(sum_i g_i^2)
                    sumSquares2 += l2 * l2;
                }
                double layerL22 = FastMath.sqrt(sumSquares2);
                if (layerL22 > threshold) {
                    double scalingFactor = threshold / layerL22; // g = g / l2 * threshold ->
                    for (INDArray g : gradient.gradientForVariable().values()) {
                        g.muli(scalingFactor);
                    }
                }
                break;
            case ClipL2PerParamType:
                for (INDArray g : gradient.gradientForVariable().values()) {
                    double l2 = g.norm2Number().doubleValue();
                    if (l2 > threshold) {
                        double scalingFactor = l2 / threshold;
                        g.divi(scalingFactor);
                    }
                }
                break;
            default:
                throw new RuntimeException(
                        "Unknown (or not implemented) gradient normalization strategy: " + normalization);
        }
    }
}
