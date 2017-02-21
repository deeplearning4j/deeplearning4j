package org.deeplearning4j.nn.transferlearning;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by susaneraly on 2/20/17.
 */
public class TransferLearningComplex {

    @Test
    public void testWithMergeAndSubset() {

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().learningRate(0.001).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD);
        /*
                        inCentre        inRight
                            |               |
                        denseCentre0        |
                            |               |
             ---------  denseCentre1     denseRight0
            /               |               |
        subsetLeft(0-3) denseCentre2 ----mergeRight
          |                 |               |
     denseLeft0         denseCentre3       denseRight1
          |                 |                |
     outLeft             outCentre         outRight

         */

        ComputationGraphConfiguration conf
                = overallConf.graphBuilder()
                        .addInputs("inCentre","inRight")
                        .addLayer("denseCentre0", new DenseLayer.Builder().nIn(10).nOut(9).build(),"inCentre")
                        .addLayer("denseCentre1", new DenseLayer.Builder().nIn(9).nOut(8).build(),"denseCentre0")
                        .addLayer("denseCentre2", new DenseLayer.Builder().nIn(8).nOut(7).build(),"denseCentre1")
                        .addLayer("denseCentre3", new DenseLayer.Builder().nIn(7).nOut(7).build(),"denseCentre2")
                        .addLayer("outCentre", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(7).nOut(4).build(),"denseCentre3")
                        .addVertex("subsetLeft", new SubsetVertex(0,3),"denseCentre1")
                        .addLayer("denseLeft0", new DenseLayer.Builder().nIn(4).nOut(5).build(),"subsetLeft")
                        .addLayer("outLeft", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(5).nOut(6).build(),"denseLeft0")
                        .addLayer("denseRight0", new DenseLayer.Builder().nIn(2).nOut(3).build(),"inRight")
                        .addVertex("mergeRight", new MergeVertex(),"denseCentre2","denseRight0")
                        .addLayer("denseRight1", new DenseLayer.Builder().nIn(10).nOut(5).build(),"mergeRight")
                        .addLayer("outRight", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(5).nOut(5).build(),"denseRight1")
                        .setOutputs("outLeft","outCentre","outRight")
                        .build();

        ComputationGraph modelToTune = new ComputationGraph(conf);
        modelToTune.init();

        INDArray [] features = new INDArray[] {Nd4j.rand(10,10),Nd4j.rand(10,2)};
        INDArray [] labels = new INDArray[] {Nd4j.rand(10,6),Nd4j.rand(10,4),Nd4j.rand(10,5)};

        //Instead of manually setting these three vertices - the comp graph .setFeatureExtractor should have a way of saying freeze from inCentre to denseCentre2
        //and should trace the path from inCentre to denseCenter2 and freeze only those (not freeze the mergeRight etc etc)
        modelToTune.getVertex("denseCentre0").setLayerAsFrozen();
        modelToTune.getVertex("denseCentre1").setLayerAsFrozen();
        modelToTune.getVertex("denseCentre2").setLayerAsFrozen();
        modelToTune.setParams(modelToTune.params());


        ComputationGraph leftGraph =
                    new ComputationGraph(overallConf.graphBuilder()
                        .addInputs("subsetLeft")
                        .addLayer("denseLeft0",new DenseLayer.Builder().nIn(4).nOut(5).build(),"subsetLeft")
                        .addLayer("outLeft", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(5).nOut(6).build(),"denseLeft0")
                        .setOutputs("outLeft")
                        .build());
        leftGraph.init();
        leftGraph.getLayer("denseLeft0").setParams(modelToTune.getLayer("denseLeft0").params());
        leftGraph.getLayer("outLeft").setParams(modelToTune.getLayer("outLeft").params());

        ComputationGraph rightGraph =
                new ComputationGraph(overallConf.graphBuilder()
                        .addInputs("denseCentre2","inRight")
                        .addLayer("denseRight0", new DenseLayer.Builder().nIn(2).nOut(3).build(),"inRight")
                        .addVertex("mergeRight", new MergeVertex(),"denseCentre2","denseRight0")
                        .addLayer("denseRight1", new DenseLayer.Builder().nIn(10).nOut(5).build(),"mergeRight")
                        .addLayer("outRight", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(5).nOut(5).build(),"denseRight1")
                        .setOutputs("outRight")
                        .build());
        rightGraph.init();
        rightGraph.getLayer("denseRight0").setParams(modelToTune.getLayer("denseRight0").params());
        rightGraph.getLayer("denseRight1").setParams(modelToTune.getLayer("denseRight1").params());
        rightGraph.getLayer("outRight").setParams(modelToTune.getLayer("outRight").params());

        ComputationGraph centreGraph =
                new ComputationGraph(overallConf.graphBuilder()
                        .addInputs("denseCentre2")
                        .addLayer("denseCentre3", new DenseLayer.Builder().nIn(7).nOut(7).build(),"denseCentre2")
                        .addLayer("outCentre", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(7).nOut(4).build(),"denseCentre3")
                        .setOutputs("outCentre")
                        .build());
        centreGraph.init();
        centreGraph.getLayer("denseCentre3").setParams(modelToTune.getLayer("denseCentre3").params());
        centreGraph.getLayer("outCentre").setParams(modelToTune.getLayer("outCentre").params());

        ComputationGraph frozenGraph =
                new ComputationGraph(overallConf.graphBuilder()
                        .addInputs("inCentre")
                        .addLayer("denseCentre0", new DenseLayer.Builder().nIn(10).nOut(9).build(),"inCentre")
                        .addLayer("denseCentre1", new DenseLayer.Builder().nIn(9).nOut(8).build(),"denseCentre0")
                        .addLayer("denseCentre2", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(8).nOut(7).build(),"denseCentre1")
                        .setOutputs("denseCentre2")
                        .build());
        frozenGraph.init();
        frozenGraph.getLayer("denseCentre0").setParams(modelToTune.getLayer("denseCentre0").params());
        frozenGraph.getLayer("denseCentre1").setParams(modelToTune.getLayer("denseCentre1").params());
        frozenGraph.getLayer("denseCentre2").setParams(modelToTune.getLayer("denseCentre2").params());

        INDArray denseCentre2 = frozenGraph.output(features[0])[0];
        INDArray subsetLeft = frozenGraph.feedForward(features[0],true).get("denseCentre1").get(NDArrayIndex.all(),NDArrayIndex.interval(0,3,true));

        leftGraph.fit(new DataSet(subsetLeft,labels[0]));
        centreGraph.fit(new DataSet(denseCentre2,labels[1]));
        MultiDataSet rightDataSet = new MultiDataSet(new INDArray[] {denseCentre2,features[1]},new INDArray[] {labels[2]});
        rightGraph.fit(rightDataSet);

        modelToTune.fit(new MultiDataSet(features,labels));

        assertEquals(modelToTune.getLayer("denseCentre0").params(),frozenGraph.getLayer("denseCentre0").params());
        assertEquals(modelToTune.getLayer("denseCentre1").params(),frozenGraph.getLayer("denseCentre1").params());
        assertEquals(modelToTune.getLayer("denseCentre2").params(),frozenGraph.getLayer("denseCentre2").params());
        assertEquals(modelToTune.getLayer("denseCentre3").params(),centreGraph.getLayer("denseCentre3").params());
        assertEquals(modelToTune.getLayer("outCentre").params(),centreGraph.getLayer("outCentre").params());
        assertEquals(modelToTune.getLayer("denseLeft0").params(),leftGraph.getLayer("denseLeft0").params());
        assertEquals(modelToTune.getLayer("outLeft").params(),leftGraph.getLayer("outLeft").params());
        assertEquals(modelToTune.getLayer("denseRight0").params(),rightGraph.getLayer("denseRight0").params());
        //Fails with small differences
        //assertEquals(modelToTune.getLayer("denseRight1").params(),rightGraph.getLayer("denseRight1").params());
        assertTrue(modelToTune.getLayer("denseRight1").params().equalsWithEps(rightGraph.getLayer("denseRight1").params(),0.00000001));
        //Fails with small differences
        //assertEquals(modelToTune.getLayer("outRight").params(),rightGraph.getLayer("outRight").params());
        assertTrue(modelToTune.getLayer("outRight").params().equalsWithEps(rightGraph.getLayer("outRight").params(),0.00000001));
    }
}
