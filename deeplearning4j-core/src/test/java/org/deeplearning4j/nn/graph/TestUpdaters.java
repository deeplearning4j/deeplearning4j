package org.deeplearning4j.nn.graph;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.LayerUpdater;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 05/08/2016.
 */
public class TestUpdaters {

    @Test
    public void testUpdaters() throws Exception {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .graphBuilder()
                .addInputs("input") // 40x40x1
                .addLayer("l0_cnn", new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})/*.nIn(1)*/.nOut(100).build(), "input") // out: 40x40x100
                .addLayer("l1_max", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}, new int[]{2,2}, new int[]{1, 1}).build(), "l0_cnn") // 21x21x100
                .addLayer("l2_cnn", new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1})/*.nIn(100)*/.nOut(200).build(), "l1_max") // 11x11x200
                .addLayer("l3_max", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2,2}, new int[]{1, 1}).build(), "l2_cnn") // 6x6x200
                .addLayer("l4_fc", new DenseLayer.Builder()/*.nIn(6*6*200)*/.nOut(1024).build(), "l3_max") // output: 1x1x1024
                .addLayer("l5_out", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        /*.nIn(1024)*/.nOut(10).activation("softmax").build(), "l4_fc")
                .setOutputs("l5_out")
                .backprop(true).pretrain(false)
                .setInputTypes(InputType.convolutional(40,40,1))
                .build();

        //First: check that the nIns are set properly...
        Map<String,GraphVertex> map = conf.getVertices();
        LayerVertex l0_cnn = (LayerVertex)map.get("l0_cnn");
        LayerVertex l2_cnn = (LayerVertex)map.get("l2_cnn");
        LayerVertex l4_fc = (LayerVertex)map.get("l4_fc");
        LayerVertex l5_out = (LayerVertex)map.get("l5_out");

        assertEquals(1, ((FeedForwardLayer)l0_cnn.getLayerConf().getLayer()).getNIn());
        assertEquals(100, ((FeedForwardLayer)l2_cnn.getLayerConf().getLayer()).getNIn());
        assertEquals(6*6*200, ((FeedForwardLayer)l4_fc.getLayerConf().getLayer()).getNIn());
        assertEquals(1024, ((FeedForwardLayer)l5_out.getLayerConf().getLayer()).getNIn());


        //Check updaters state:
        ComputationGraph g = new ComputationGraph(conf);
        g.init();
        g.initGradientsView();

        ComputationGraphUpdater updater = g.getUpdater();

        //First: get the updaters array
        Field layerUpdatersField = updater.getClass().getDeclaredField("layerUpdaters");
        layerUpdatersField.setAccessible(true);
        org.deeplearning4j.nn.api.Updater[] layerUpdaters = (org.deeplearning4j.nn.api.Updater[])layerUpdatersField.get(updater);

        //And get the map between names and updater indexes
        Field layerUpdatersMapField = updater.getClass().getDeclaredField("layerUpdatersMap");
        layerUpdatersMapField.setAccessible(true);
        Map<String,Integer> layerUpdatersMap = (Map<String,Integer>)layerUpdatersMapField.get(updater);


        //Go through each layer; check that the updater state size matches the parameters size
        org.deeplearning4j.nn.api.Layer[] layers = g.getLayers();
        for(org.deeplearning4j.nn.api.Layer l : layers){
            String layerName = l.conf().getLayer().getLayerName();
            int nParams = l.numParams();
            Map<String,INDArray> paramTable = l.paramTable();


            Map<String,Integer> parameterSizeCounts = new LinkedHashMap<>();
            for(Map.Entry<String,INDArray> e : paramTable.entrySet()){
                parameterSizeCounts.put(e.getKey(), e.getValue().length());
            }

            int updaterIdx = layerUpdatersMap.get(layerName);
            org.deeplearning4j.nn.api.Updater u = layerUpdaters[updaterIdx];

            LayerUpdater lu = (LayerUpdater) u;

            Field updaterForVariableField = LayerUpdater.class.getDeclaredField("updaterForVariable");
            updaterForVariableField.setAccessible(true);
            Map<String,GradientUpdater> updaterForVariable = (Map<String,GradientUpdater>)updaterForVariableField.get(lu);
            Map<String,Integer> updaterStateSizeCounts = new HashMap<>();
            for(Map.Entry<String,GradientUpdater> entry : updaterForVariable.entrySet()){
                GradientUpdater gu = entry.getValue();
                Nesterovs nesterovs = (Nesterovs)gu;
                INDArray v = nesterovs.getV();
                int length = (v == null ? -1 : v.length());
                updaterStateSizeCounts.put(entry.getKey(), length);
            }

            //Check subsampling layers:
            if(l.numParams() == 0){
                assertEquals(0, updaterForVariable.size());
            }

            System.out.println(layerName + "\t" + nParams + "\t" + parameterSizeCounts + "\t Updater size: " + updaterStateSizeCounts);

            //Now, with nesterov updater: 1 history value per parameter
            for(String s : parameterSizeCounts.keySet()){
                int paramSize = parameterSizeCounts.get(s);
                int updaterSize = updaterStateSizeCounts.get(s);

                assertEquals(layerName+"/"+s, paramSize, updaterSize);
            }

        }

        INDArray in = Nd4j.create(2,1,40,40);   //minibatch, depth, height, width
        INDArray l = Nd4j.create(2,10);

        DataSet ds = new DataSet(in,l);

        g.fit(ds);
    }


}
