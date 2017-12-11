package org.deeplearning4j.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.schedule.ISchedule;

public class NetworkUtils {

    private NetworkUtils(){ }

    /**
     * Convert a MultiLayerNetwork to a ComputationGraph
     *
     * @return ComputationGraph equivalent to this network (including parameters and updater state)
     */
    public static ComputationGraph toComputationGraph(MultiLayerNetwork net){

        //We rely heavily here on the fact that the topological sort order - and hence the layout of parameters - is
        // by definition the identical for a MLN and "single stack" computation graph. This also has to hold
        // for the updater state...

        ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
                .graphBuilder();

        MultiLayerConfiguration origConf = net.getLayerWiseConfigurations().clone();


        int layerIdx = 0;
        String lastLayer = "in";
        b.addInputs("in");
        for(NeuralNetConfiguration c : origConf.getConfs()){
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
        if(updaterState != null) {
            cg.getUpdater().getUpdaterStateViewArray()
                    .assign(updaterState);
        }

        return cg;
    }

    public  static void setLearningRate(MultiLayerNetwork net, double newLr) {
        setLearningRate(net, newLr, null);
    }

    public  static void setLearningRate(MultiLayerNetwork net, ISchedule newLrSchedule) {
        setLearningRate(net, Double.NaN, newLrSchedule);
    }

    private static void setLearningRate(MultiLayerNetwork net, double newLr, ISchedule lrSchedule){
        int nLayers = net.getnLayers();
        for( int i=0; i<nLayers; i++ ){
            setLearningRate(net, i, newLr, lrSchedule, false);
        }
        refreshUpdater(net);
    }

    public static void setLearningRate(MultiLayerNetwork net, int layerNumber, double newLr){
        setLearningRate(net, layerNumber, newLr, null, true);
    }

    private static void setLearningRate(MultiLayerNetwork net, int layerNumber, double newLr, ISchedule newLrSchedule, boolean refreshUpdater){

        Layer l = net.getLayer(layerNumber).conf().getLayer();
        if(l instanceof BaseLayer){
            BaseLayer bl = (BaseLayer)l;
            IUpdater u = bl.getIUpdater();
            if(u != null && u.hasLearningRate()){
                if(newLrSchedule != null){
                    u.setLrAndSchedule(Double.NaN, newLrSchedule);
                } else {
                    u.setLrAndSchedule(newLr, null);
                }
            }

            //Need to refresh the updater - if we change the LR (or schedule) we may rebuild the updater blocks, which are
            // built by creating blocks of params with the same configuration
            if(refreshUpdater){
                refreshUpdater(net);
            }
        }
    }

    private static void refreshUpdater(MultiLayerNetwork net){
        INDArray origUpdaterState = net.getUpdater().getStateViewArray();
        net.setUpdater(null);
        MultiLayerUpdater u = (MultiLayerUpdater) net.getUpdater();
        u.setStateViewArray(origUpdaterState);
    }
}
