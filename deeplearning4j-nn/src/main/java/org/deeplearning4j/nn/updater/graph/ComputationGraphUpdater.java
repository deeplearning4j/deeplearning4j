package org.deeplearning4j.nn.updater.graph;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.updater.UpdaterBlock;
import org.deeplearning4j.nn.updater.UpdaterUtils;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Norm2;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Gradient updater for ComputationGraph.<br>
 * Note: ComputationGraph does not implement the Layer interface (due to multiple in/out etc), hence ComputationGraphUpdater
 * can't be defined as an {@link Updater}.
 *
 * @author Alex Black
 */
public class ComputationGraphUpdater implements Serializable {

    private final List<UpdaterBlock> updaterBlocks;
    private INDArray updaterStateViewArray;

    public ComputationGraphUpdater(ComputationGraph graph) {
        this(graph, null);
    }

    public ComputationGraphUpdater(ComputationGraph graph, INDArray updaterState) {
        Layer[] layers = graph.getLayers();
        GraphVertex[] vertices = graph.getVertices();

        //In CompGraph: we need to know topological ordering, so we know how parameters are laid out in the 1d view arrays
        int[] topologicalOrdering = graph.topologicalSortOrder();

        int updaterStateSize = 0;
        //Iterate through layers, and variables for each layer.
        //While the updater configuration is the same: combine
        Layer lastLayer = null;
        String lastVariable = null;
        UpdaterBlock currentBlock = null;
        updaterBlocks = new ArrayList<>();
        int currentParamOffset = 0;
        int currentUpdaterOffset = 0;

        INDArray paramsView = graph.params();
        INDArray gradientView = graph.getFlattenedGradients();
        int paramsViewSoFar = 0;

        for( int i=0; i<topologicalOrdering.length; i++ ){
            GraphVertex currentVertex = vertices[topologicalOrdering[i]];
            if(!currentVertex.hasLayer()){
                continue;
            }

            Layer currentLayer = currentVertex.getLayer();
            Map<String,INDArray> layerParamTable = currentLayer.paramTable();
            List<String> variables = new ArrayList<>(layerParamTable.keySet());    //Is from a set, but iteration order should be fixed per layer as it's a from a LinkedHashSet

            for( int j=0; j<variables.size(); j++ ){
                String var = variables.get(j);
                int paramSizeThisVariable = layerParamTable.get(var).length();
                int updaterStateSizeThisVariable = UpdaterUtils.stateSizeForLayerVariable(currentLayer, var);

                INDArray gradientViewSubset = null;
                INDArray paramsViewSubset = null;
                if(paramSizeThisVariable > 0) {
                    paramsViewSubset = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(paramsViewSoFar, paramsViewSoFar + paramSizeThisVariable));
                    gradientViewSubset = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(paramsViewSoFar, paramsViewSoFar + paramSizeThisVariable));
                }

                //First: decide whether to add to the existing updater block, or create a new one
                if(currentBlock == null || !UpdaterUtils.updaterConfigurationsEquals(lastLayer, lastVariable, currentLayer, var)){
                    List<UpdaterBlock.VarState> list = new ArrayList<>();
                    list.add(new UpdaterBlock.VarState(currentLayer, var, paramsViewSubset, gradientViewSubset));
                    currentBlock = new UpdaterBlock(currentParamOffset, currentParamOffset+paramSizeThisVariable,
                            currentUpdaterOffset, currentUpdaterOffset+updaterStateSizeThisVariable, list);

                    updaterBlocks.add(currentBlock);
                } else {
                    //Add to existing updater block
                    currentBlock.setParamOffsetEnd( currentBlock.getParamOffsetEnd() + paramSizeThisVariable);
                    currentBlock.setUpdaterViewOffsetEnd( currentBlock.getUpdaterViewOffsetEnd() + updaterStateSizeThisVariable);
                    currentBlock.getLayersAndVariablesInBlock().add(
                            new UpdaterBlock.VarState(currentLayer, var, paramsViewSubset, gradientViewSubset));
                }

                lastLayer = currentLayer;
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

//        System.out.println("Updater state array: " + Arrays.toString(updaterStateViewArray.shape()));
//        System.out.println("Parameters array shape: " + Arrays.toString(paramsView.shape()));

        //Set up the updaters, for each updater block:
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

    public void update(ComputationGraph graph, Gradient gradient, int iteration, int batchSize) {
        Map<String, Gradient> layerGradients = new HashMap<>();


        for (Map.Entry<String, INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
            String key = gradientPair.getKey();
            int idx = key.lastIndexOf('_');
            if (idx == -1)
                throw new IllegalStateException(
                        "Invalid key: MuliLayerNetwork Gradient key does not have layer separator: \"" + key
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

        //PRE apply (gradient clipping, etc): done on a per-layer basis
        for (Map.Entry<String, Gradient> entry : layerGradients.entrySet()) {
            String layerName = entry.getKey();
            Layer layer = graph.getLayer(layerName);

            preApply(layer, layerGradients.get(layerName), iteration);
        }


        //Apply the updaters in blocks
        for(UpdaterBlock ub : updaterBlocks){
            ub.update(iteration);
        }

        if(graph.conf().isMiniBatch()){
            graph.getFlattenedGradients().divi(batchSize);
        }
    }


    public void setStateViewArray(INDArray viewArray) {
        if (this.updaterStateViewArray.length() != viewArray.length())
            throw new IllegalStateException("Invalid input: view arrays differ in length. " + "Expected length "
                            + this.updaterStateViewArray.length() + ", got length " + viewArray.length());
        this.updaterStateViewArray.assign(viewArray);
    }


    public INDArray getStateViewArray() {
        return updaterStateViewArray;
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof ComputationGraphUpdater))
            return false;
//        return layerUpdatersMap.equals(((ComputationGraphUpdater) other).layerUpdatersMap);
        throw new UnsupportedOperationException("Not yet re-implemented");
    }

    @Override
    public int hashCode() {
//        return layerUpdatersMap.hashCode();
        throw new UnsupportedOperationException("Not yet re-implemented");
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
