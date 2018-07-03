package org.deeplearning4j.util;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.schedule.ISchedule;

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

    private static void setLearningRate(MultiLayerNetwork net, double newLr, ISchedule lrSchedule) {
        int nLayers = net.getnLayers();
        for (int i = 0; i < nLayers; i++) {
            setLearningRate(net, i, newLr, lrSchedule, false);
        }
        refreshUpdater(net);
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

    /**
     * Get the current learning rate, for the specified layer, fromthe network.
     * Note: If the layer has no learning rate (no parameters, or an updater without a learning rate) then null is returned
     * @param net           Network
     * @param layerNumber   Layer number to get the learning rate for
     * @return Learning rate for the specified layer, or null
     */
    public static Double getLearningRate(MultiLayerNetwork net, int layerNumber){
        Layer l = net.getLayer(layerNumber).conf().getLayer();
        int iter = net.getIterationCount();
        int epoch = net.getEpochCount();
        if (l instanceof BaseLayer) {
            BaseLayer bl = (BaseLayer) l;
            IUpdater u = bl.getIUpdater();
            if (u != null && u.hasLearningRate()) {
                double d = u.getLearningRate(iter, epoch);
                if(Double.isNaN(d)){
                    return null;
                }
                return d;
            }
            return null;
        }
        return null;
    }

    private static void refreshUpdater(MultiLayerNetwork net) {
        INDArray origUpdaterState = net.getUpdater().getStateViewArray();
        net.setUpdater(null);
        MultiLayerUpdater u = (MultiLayerUpdater) net.getUpdater();
        u.setStateViewArray(origUpdaterState);
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

    private static void setLearningRate(ComputationGraph net, double newLr, ISchedule lrSchedule) {
        org.deeplearning4j.nn.api.Layer[] layers = net.getLayers();
        for (int i = 0; i < layers.length; i++) {
            setLearningRate(net, layers[i].conf().getLayer().getLayerName(), newLr, lrSchedule, false);
        }
        refreshUpdater(net);
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

    /**
     * Get the current learning rate, for the specified layer, from the network.
     * Note: If the layer has no learning rate (no parameters, or an updater without a learning rate) then null is returned
     * @param net        Network
     * @param layerName  Layer name to get the learning rate for
     * @return Learning rate for the specified layer, or null
     */
    public static Double getLearningRate(ComputationGraph net, String layerName){
        Layer l = net.getLayer(layerName).conf().getLayer();
        int iter = net.getConfiguration().getIterationCount();
        int epoch = net.getConfiguration().getEpochCount();
        if (l instanceof BaseLayer) {
            BaseLayer bl = (BaseLayer) l;
            IUpdater u = bl.getIUpdater();
            if (u != null && u.hasLearningRate()) {
                double d = u.getLearningRate(iter, epoch);
                if(Double.isNaN(d)){
                    return null;
                }
                return d;
            }
            return null;
        }
        return null;
    }

    private static void refreshUpdater(ComputationGraph net) {
        INDArray origUpdaterState = net.getUpdater().getStateViewArray();
        net.setUpdater(null);
        ComputationGraphUpdater u = net.getUpdater();
        u.setStateViewArray(origUpdaterState);
    }

  /**
   * Currently supports {@link MultiLayerNetwork} and {@link ComputationGraph} models.
   * Pull requests to support additional <code>org.deeplearning4j</code> models are welcome.
   *
   * @param model Model to use
   * @param input Inputs to the model
   * @return output Outputs of the model
   *
   * @see org.deeplearning4j.nn.graph.ComputationGraph#outputSingle(INDArray...)
   * @see org.deeplearning4j.nn.multilayer.MultiLayerNetwork#output(INDArray)
   */
  public static INDArray output(Model model, INDArray input) {

    if (model instanceof MultiLayerNetwork) {
      final MultiLayerNetwork multiLayerNetwork = (MultiLayerNetwork)model;
      final INDArray output = multiLayerNetwork.output(input);
      return output;
    }

    if (model instanceof ComputationGraph) {
      final ComputationGraph computationGraph = (ComputationGraph)model;
      final INDArray output = computationGraph.outputSingle(input);
      return output;
    }

    final String message;
    if (model.getClass().getName().startsWith("org.deeplearning4j")) {
      message = model.getClass().getName()+" models are not yet supported and " +
        "pull requests are welcome: https://github.com/deeplearning4j/deeplearning4j";
    } else {
      message = model.getClass().getName()+" models are unsupported.";
    }

    throw new UnsupportedOperationException(message);
  }

}
