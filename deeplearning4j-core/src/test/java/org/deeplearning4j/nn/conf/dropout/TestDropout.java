package org.deeplearning4j.nn.conf.dropout;

import lombok.Data;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Triple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TestDropout {

    @Test
    public void testBasicConfig(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dropOut(0.6)
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).dropOut(0.7).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).dropOut(new AlphaDropout(0.5)).build())
                .build();

        assertEquals(new Dropout(0.6), conf.getConf(0).getLayer().getIDropout());
        assertEquals(new Dropout(0.7), conf.getConf(1).getLayer().getIDropout());
        assertEquals(new AlphaDropout(0.5), conf.getConf(2).getLayer().getIDropout());


        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .dropOut(0.6)
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in")
                .addLayer("1", new DenseLayer.Builder().nIn(10).nOut(10).dropOut(0.7).build(), "0")
                .addLayer("2", new DenseLayer.Builder().nIn(10).nOut(10).dropOut(new AlphaDropout(0.5)).build(), "1")
                .setOutputs("2")
                .build();

        assertEquals(new Dropout(0.6), ((LayerVertex)conf2.getVertices().get("0")).getLayerConf().getLayer().getIDropout());
        assertEquals(new Dropout(0.7), ((LayerVertex)conf2.getVertices().get("1")).getLayerConf().getLayer().getIDropout());
        assertEquals(new AlphaDropout(0.5), ((LayerVertex)conf2.getVertices().get("2")).getLayerConf().getLayer().getIDropout());
    }

    @Test
    public void testCalls(){

        CustomDropout d1 = new CustomDropout();
        CustomDropout d2 = new CustomDropout();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(3).dropOut(d1).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).dropOut(d2).nIn(3).nOut(3).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        List<DataSet> l = new ArrayList<>();
        l.add(new DataSet(Nd4j.rand(5,4), Nd4j.rand(5,3)));
        l.add(new DataSet(Nd4j.rand(5,4), Nd4j.rand(5,3)));
        l.add(new DataSet(Nd4j.rand(5,4), Nd4j.rand(5,3)));

        DataSetIterator iter = new ExistingDataSetIterator(l);

        net.fit(iter);
        net.fit(iter);

        List<Triple<Integer,Integer,Boolean>> expList = Arrays.asList(
                new Triple<>(0, 0, false),
                new Triple<>(1, 0, false),
                new Triple<>(2, 0, false),
                new Triple<>(0, 1, false),
                new Triple<>(1, 1, false),
                new Triple<>(2, 1, false));

        assertEquals(expList, d1.getAllCalls());
        assertEquals(expList, d2.getAllCalls());


        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .dropOut(0.6)
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(4).nOut(3).dropOut(d1).build(), "in")
                .addLayer("1", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).dropOut(d2).nIn(3).nOut(3).build(), "0")
                .setOutputs("1")
                .build();

    }

    @Data
    private static class CustomDropout implements IDropout{

        private List<Triple<Integer,Integer,Boolean>> allCalls = new ArrayList<>();

        @Override
        public INDArray applyDropout(INDArray inputActivations, int iteration, int epoch, boolean inPlace) {
            allCalls.add(new Triple<>(iteration, epoch, inPlace));
            return inputActivations;
        }

        @Override
        public IDropout clone() {
            throw new UnsupportedOperationException();
        }
    }

    @Test
    public void testValues(){

    }

}
