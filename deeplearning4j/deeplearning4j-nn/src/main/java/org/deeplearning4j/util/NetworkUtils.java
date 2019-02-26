/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.util;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Trainable;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.deeplearning4j.nn.updater.UpdaterBlock;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.*;

@Slf4j
public class NetworkUtils {

    private NetworkUtils() {
    }

    /**
     * Convert a MultiLayerNetwork to a ComputationGraph
     *
     * @return ComputationGraph equivalent to this network (including parameters and updater state)
     */
    public static ComputationGraph toComputationGraph(MultiLayerNetwork net) {

        //We rely heavily here on the fact that the topological sort order - and hence the layout of parameters - is
        // by definition the identical for a MLN and "single stack" computation graph. This also has to hold
        // for the updater state...

        ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
                .graphBuilder();

        MultiLayerConfiguration origConf = net.getLayerWiseConfigurations().clone();


        int layerIdx = 0;
        String lastLayer = "in";
        b.addInputs("in");
        for (NeuralNetConfiguration c : origConf.getConfs()) {
            String currLayer = String.valueOf(layerIdx);

            InputPreProcessor preproc = origConf.getInputPreProcess(layerIdx);
            b.addLayer(currLayer, c.getLayer(), preproc, lastLayer);

            lastLayer = currLayer;
            layerIdx++;
        }
        b.setOutputs(lastLayer);

        ComputationGraphConfiguration conf = b.build();

        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        cg.setParams(net.params());

        //Also copy across updater state:
        INDArray updaterState = net.getUpdater().getStateViewArray();
        if (updaterState != null) {
            cg.getUpdater().getUpdaterStateViewArray()
                    .assign(updaterState);
        }

        return cg;
    }

    /**
     * Set the learning rate for all layers in the network to the specified value. Note that if any learning rate
     * schedules are currently present, these will be removed in favor of the new (fixed) learning rate.<br>
     * <br>
     * <b>Note</b>: <i>This method not free from a performance point of view</i>: a proper learning rate schedule
     * should be used in preference to calling this method at every iteration.
     *
     * @param net   Network to set the LR for
     * @param newLr New learning rate for all layers
     */
    public static void setLearningRate(MultiLayerNetwork net, double newLr) {
        setLearningRate(net, newLr, null);
    }

    private static void setLearningRate(MultiLayerNetwork net, double newLr, ISchedule lrSchedule) {
        int nLayers = net.getnLayers();
        for (int i = 0; i < nLayers; i++) {
            setLearningRate(net, i, newLr, lrSchedule, false);
        }
        refreshUpdater(net);
    }

    private static void setLearningRate(MultiLayerNetwork net, int layerNumber, double newLr, ISchedule newLrSchedule, boolean refreshUpdater) {

        Layer l = net.getLayer(layerNumber).conf().getLayer();
        if (l instanceof BaseLayer) {
            BaseLayer bl = (BaseLayer) l;
            IUpdater u = bl.getIUpdater();
            if (u != null && u.hasLearningRate()) {
                if (newLrSchedule != null) {
                    u.setLrAndSchedule(Double.NaN, newLrSchedule);
                } else {
                    u.setLrAndSchedule(newLr, null);
                }
            }

            //Need to refresh the updater - if we change the LR (or schedule) we may rebuild the updater blocks, which are
            // built by creating blocks of params with the same configuration
            if (refreshUpdater) {
                refreshUpdater(net);
            }
        }
    }

    private static void refreshUpdater(MultiLayerNetwork net) {
        INDArray origUpdaterState = net.getUpdater().getStateViewArray();
        MultiLayerUpdater origUpdater = (MultiLayerUpdater) net.getUpdater();
        net.setUpdater(null);
        MultiLayerUpdater newUpdater = (MultiLayerUpdater) net.getUpdater();
        INDArray newUpdaterState = rebuildUpdaterStateArray(origUpdaterState, origUpdater.getUpdaterBlocks(), newUpdater.getUpdaterBlocks());
        newUpdater.setStateViewArray(newUpdaterState);
    }

    /**
     * Set the learning rate schedule for all layers in the network to the specified schedule.
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note that the iteration/epoch counts will <i>not</i> be reset. Use {@link MultiLayerConfiguration#setIterationCount(int)}
     * and {@link MultiLayerConfiguration#setEpochCount(int)} if this is required
     *
     * @param newLrSchedule New learning rate schedule for all layers
     */
    public static void setLearningRate(MultiLayerNetwork net, ISchedule newLrSchedule) {
        setLearningRate(net, Double.NaN, newLrSchedule);
    }

    /**
     * Set the learning rate for a single layer in the network to the specified value. Note that if any learning rate
     * schedules are currently present, these will be removed in favor of the new (fixed) learning rate.<br>
     * <br>
     * <b>Note</b>: <i>This method not free from a performance point of view</i>: a proper learning rate schedule
     * should be used in preference to calling this method at every iteration. Note also that
     * {@link #setLearningRate(MultiLayerNetwork, double)} should also be used in preference, when all layers need to be set to a new LR
     *
     * @param layerNumber Number of the layer to set the LR for
     * @param newLr       New learning rate for a single layers
     */
    public static void setLearningRate(MultiLayerNetwork net, int layerNumber, double newLr) {
        setLearningRate(net, layerNumber, newLr, null, true);
    }

    /**
     * Set the learning rate schedule for a single layer in the network to the specified value.<br>
     * Note also that {@link #setLearningRate(MultiLayerNetwork, ISchedule)} should also be used in preference, when all layers need
     * to be set to a new LR schedule.<br>
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note also that the iteration/epoch counts will <i>not</i> be reset. Use {@link MultiLayerConfiguration#setIterationCount(int)}
     * and {@link MultiLayerConfiguration#setEpochCount(int)} if this is required
     *
     * @param layerNumber Number of the layer to set the LR schedule for
     * @param lrSchedule  New learning rate for a single layer
     */
    public static void setLearningRate(MultiLayerNetwork net, int layerNumber, ISchedule lrSchedule) {
        setLearningRate(net, layerNumber, Double.NaN, lrSchedule, true);
    }

    /**
     * Get the current learning rate, for the specified layer, fromthe network.
     * Note: If the layer has no learning rate (no parameters, or an updater without a learning rate) then null is returned
     *
     * @param net         Network
     * @param layerNumber Layer number to get the learning rate for
     * @return Learning rate for the specified layer, or null
     */
    public static Double getLearningRate(MultiLayerNetwork net, int layerNumber) {
        Layer l = net.getLayer(layerNumber).conf().getLayer();
        int iter = net.getIterationCount();
        int epoch = net.getEpochCount();
        if (l instanceof BaseLayer) {
            BaseLayer bl = (BaseLayer) l;
            IUpdater u = bl.getIUpdater();
            if (u != null && u.hasLearningRate()) {
                double d = u.getLearningRate(iter, epoch);
                if (Double.isNaN(d)) {
                    return null;
                }
                return d;
            }
            return null;
        }
        return null;
    }

    /**
     * Set the learning rate for all layers in the network to the specified value. Note that if any learning rate
     * schedules are currently present, these will be removed in favor of the new (fixed) learning rate.<br>
     * <br>
     * <b>Note</b>: <i>This method not free from a performance point of view</i>: a proper learning rate schedule
     * should be used in preference to calling this method at every iteration.
     *
     * @param net   Network to set the LR for
     * @param newLr New learning rate for all layers
     */
    public static void setLearningRate(ComputationGraph net, double newLr) {
        setLearningRate(net, newLr, null);
    }

    private static void setLearningRate(ComputationGraph net, double newLr, ISchedule lrSchedule) {
        org.deeplearning4j.nn.api.Layer[] layers = net.getLayers();
        for (int i = 0; i < layers.length; i++) {
            setLearningRate(net, layers[i].conf().getLayer().getLayerName(), newLr, lrSchedule, false);
        }
        refreshUpdater(net);
    }

    private static void setLearningRate(ComputationGraph net, String layerName, double newLr, ISchedule newLrSchedule, boolean refreshUpdater) {

        Layer l = net.getLayer(layerName).conf().getLayer();
        if (l instanceof BaseLayer) {
            BaseLayer bl = (BaseLayer) l;
            IUpdater u = bl.getIUpdater();
            if (u != null && u.hasLearningRate()) {
                if (newLrSchedule != null) {
                    u.setLrAndSchedule(Double.NaN, newLrSchedule);
                } else {
                    u.setLrAndSchedule(newLr, null);
                }
            }

            //Need to refresh the updater - if we change the LR (or schedule) we may rebuild the updater blocks, which are
            // built by creating blocks of params with the same configuration
            if (refreshUpdater) {
                refreshUpdater(net);
            }
        }
    }

    private static void refreshUpdater(ComputationGraph net) {
        INDArray origUpdaterState = net.getUpdater().getStateViewArray();
        ComputationGraphUpdater uOrig = net.getUpdater();
        net.setUpdater(null);
        ComputationGraphUpdater uNew = net.getUpdater();
        INDArray newUpdaterState = rebuildUpdaterStateArray(origUpdaterState, uOrig.getUpdaterBlocks(), uNew.getUpdaterBlocks());
        uNew.setStateViewArray(newUpdaterState);
    }

    /**
     * Set the learning rate schedule for all layers in the network to the specified schedule.
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note that the iteration/epoch counts will <i>not</i> be reset. Use {@link ComputationGraphConfiguration#setIterationCount(int)}
     * and {@link ComputationGraphConfiguration#setEpochCount(int)} if this is required
     *
     * @param newLrSchedule New learning rate schedule for all layers
     */
    public static void setLearningRate(ComputationGraph net, ISchedule newLrSchedule) {
        setLearningRate(net, Double.NaN, newLrSchedule);
    }

    /**
     * Set the learning rate for a single layer in the network to the specified value. Note that if any learning rate
     * schedules are currently present, these will be removed in favor of the new (fixed) learning rate.<br>
     * <br>
     * <b>Note</b>: <i>This method not free from a performance point of view</i>: a proper learning rate schedule
     * should be used in preference to calling this method at every iteration. Note also that
     * {@link #setLearningRate(ComputationGraph, double)} should also be used in preference, when all layers need to be set to a new LR
     *
     * @param layerName Name of the layer to set the LR for
     * @param newLr     New learning rate for a single layers
     */
    public static void setLearningRate(ComputationGraph net, String layerName, double newLr) {
        setLearningRate(net, layerName, newLr, null, true);
    }

    /**
     * Set the learning rate schedule for a single layer in the network to the specified value.<br>
     * Note also that {@link #setLearningRate(ComputationGraph, ISchedule)} should also be used in preference, when all
     * layers need to be set to a new LR schedule.<br>
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note also that the iteration/epoch counts will <i>not</i> be reset. Use {@link ComputationGraphConfiguration#setIterationCount(int)}
     * and {@link ComputationGraphConfiguration#setEpochCount(int)} if this is required
     *
     * @param layerName  Name of the layer to set the LR schedule for
     * @param lrSchedule New learning rate for a single layer
     */
    public static void setLearningRate(ComputationGraph net, String layerName, ISchedule lrSchedule) {
        setLearningRate(net, layerName, Double.NaN, lrSchedule, true);
    }

    /**
     * Get the current learning rate, for the specified layer, from the network.
     * Note: If the layer has no learning rate (no parameters, or an updater without a learning rate) then null is returned
     *
     * @param net       Network
     * @param layerName Layer name to get the learning rate for
     * @return Learning rate for the specified layer, or null
     */
    public static Double getLearningRate(ComputationGraph net, String layerName) {
        Layer l = net.getLayer(layerName).conf().getLayer();
        int iter = net.getConfiguration().getIterationCount();
        int epoch = net.getConfiguration().getEpochCount();
        if (l instanceof BaseLayer) {
            BaseLayer bl = (BaseLayer) l;
            IUpdater u = bl.getIUpdater();
            if (u != null && u.hasLearningRate()) {
                double d = u.getLearningRate(iter, epoch);
                if (Double.isNaN(d)) {
                    return null;
                }
                return d;
            }
            return null;
        }
        return null;
    }

    /**
     * Currently supports {@link MultiLayerNetwork} and {@link ComputationGraph} models.
     * Pull requests to support additional <code>org.deeplearning4j</code> models are welcome.
     *
     * @param model Model to use
     * @param input Inputs to the model
     * @return output Outputs of the model
     * @see org.deeplearning4j.nn.graph.ComputationGraph#outputSingle(INDArray...)
     * @see org.deeplearning4j.nn.multilayer.MultiLayerNetwork#output(INDArray)
     */
    public static INDArray output(Model model, INDArray input) {

        if (model instanceof MultiLayerNetwork) {
            final MultiLayerNetwork multiLayerNetwork = (MultiLayerNetwork) model;
            final INDArray output = multiLayerNetwork.output(input);
            return output;
        }

        if (model instanceof ComputationGraph) {
            final ComputationGraph computationGraph = (ComputationGraph) model;
            final INDArray output = computationGraph.outputSingle(input);
            return output;
        }

        final String message;
        if (model.getClass().getName().startsWith("org.deeplearning4j")) {
            message = model.getClass().getName() + " models are not yet supported and " +
                    "pull requests are welcome: https://github.com/deeplearning4j/deeplearning4j";
        } else {
            message = model.getClass().getName() + " models are unsupported.";
        }

        throw new UnsupportedOperationException(message);
    }

    /**
     * Remove any instances of the specified type from the list.
     * This includes any subtypes.
     * @param list   List. May be null
     * @param remove Type of objects to remove
     */
    public static void removeInstances(List<?> list, Class<?> remove) {
        removeInstancesWithWarning(list, remove, null);
    }

    public static void removeInstancesWithWarning(List<?> list, Class<?> remove, String warning){
        if(list == null || list.isEmpty())
            return;
        Iterator<?> iter = list.iterator();
        while(iter.hasNext()){
            Object o = iter.next();
            if(remove.isAssignableFrom(o.getClass())){
                if(warning != null) {
                    log.warn(warning);
                }
                iter.remove();
            }
        }
    }


    /**
     * Rebuild the updater state after a learning rate change.
     * With updaters like Adam, they have 2 components... m and v array, for a total updater state size of 2*numParams.
     * Because we combine across parameters and layers where possible (smaller number of larger operations -> more efficient)
     * we can sometimes need to rearrange the updater state array.
     * For example, if the original updater state for Adam is organized like [mParam1, mParam2, vParam1, vParam2] in one block
     * and we change the learning rate for one of the layers, param 1 and param2 now belong to different updater blocks.
     * Consequently, we need to rearrange the updater state to be like [mParam1][vParam1] in block 1, [mParam2][vParam2] in block 2
     *
     * @param origUpdaterState Original updater state view array
     * @param orig             Original updater blocks
     * @param newUpdater       New updater blocks
     * @return New state view array
     */
    protected static INDArray rebuildUpdaterStateArray(INDArray origUpdaterState, List<UpdaterBlock> orig, List<UpdaterBlock> newUpdater){
        if(origUpdaterState == null)
            return origUpdaterState;

        //First: check if there has been any change in the updater blocks to warrant rearranging the updater state view array
        if(orig.size() == newUpdater.size()){
            boolean allEq = true;
            for( int i=0; i<orig.size(); i++ ){
                UpdaterBlock ub1 = orig.get(i);
                UpdaterBlock ub2 = newUpdater.get(i);
                if(!ub1.getLayersAndVariablesInBlock().equals(ub2.getLayersAndVariablesInBlock())){
                    allEq = false;
                    break;
                }
            }
            if(allEq){
                return origUpdaterState;
            }
        }

        Map<String,List<INDArray>> stateViewsPerParam = new HashMap<>();
        for(UpdaterBlock ub : orig){
            List<UpdaterBlock.ParamState> params = ub.getLayersAndVariablesInBlock();
            int blockPStart = ub.getParamOffsetStart();
            int blockPEnd = ub.getParamOffsetEnd();

            int blockUStart = ub.getUpdaterViewOffsetStart();
            int blockUEnd = ub.getUpdaterViewOffsetEnd();

            int paramsMultiplier = (blockUEnd-blockUStart)/(blockPEnd-blockPStart);     //Updater state length should be exactly 0, 1, 2 or 3x number of params

            INDArray updaterView = ub.getUpdaterView();
            long nParamsInBlock = blockPEnd - blockPStart;

            long soFar = 0;
            for( int sub=0; sub<paramsMultiplier; sub++) {
                //subsetUpdaterView: [m0, m1, m2] etc
                INDArray subsetUpdaterView = updaterView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + paramsMultiplier * nParamsInBlock));

                long offsetWithinSub = 0;
                for (UpdaterBlock.ParamState ps : params) {
                    int idx = getId(ps.getLayer());
                    String paramName = idx + "_" + ps.getParamName();
                    INDArray pv = ps.getParamView();
                    long nParamsThisParam = pv.length();

                    INDArray currSplit = subsetUpdaterView.get(NDArrayIndex.point(0), NDArrayIndex.interval(offsetWithinSub, offsetWithinSub + nParamsThisParam));
                    if(!stateViewsPerParam.containsKey(paramName))
                        stateViewsPerParam.put(paramName, new ArrayList<INDArray>());
                    stateViewsPerParam.get(paramName).add(currSplit);
                    offsetWithinSub += nParamsThisParam;
                }

                soFar += nParamsInBlock;
            }
        }

        //Now that we've got updater state per param, we need to reconstruct it in an order suitable for the new updater blocks...
        List<INDArray> toConcat = new ArrayList<>();
        for(UpdaterBlock ub : newUpdater){
            List<UpdaterBlock.ParamState> ps = ub.getLayersAndVariablesInBlock();
            int idx = getId(ps.get(0).getLayer());
            String firstParam = idx + "_" + ps.get(0).getParamName();
            int size = stateViewsPerParam.get(firstParam).size();
            //For multiple params in the one block, we want to order like [a0, b0, c0][a1,b1,c1]
            for( int i=0; i<size; i++ ){
                for(UpdaterBlock.ParamState p : ps) {
                    idx = getId(p.getLayer());
                    String paramName = idx + "_" + p.getParamName();
                    INDArray arr = stateViewsPerParam.get(paramName).get(i);
                    toConcat.add(arr);
                }
            }
        }
        INDArray newUpdaterState = Nd4j.hstack(toConcat);
        Preconditions.checkState(newUpdaterState.rank() == 2, "Expected rank 2");
        Preconditions.checkState(origUpdaterState.length() == newUpdaterState.length(), "Updater state array lengths should be equal: got %s s. %s",
                origUpdaterState.length(), newUpdaterState.length());
        return newUpdaterState;
    }


    private static int getId(Trainable trainable){
        if(trainable instanceof GraphVertex){
            GraphVertex gv = (GraphVertex)trainable;
            return gv.getVertexIndex();
        } else {
            org.deeplearning4j.nn.api.Layer l = (org.deeplearning4j.nn.api.Layer)trainable;
            return l.getIndex();
        }
    }

}
