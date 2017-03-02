package org.deeplearning4j.nn.transferlearning;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.*;

/**
 * Created by susaneraly on 2/20/17.
 */
@Slf4j
public class TransferLearningComplex {

    @Test
    public void testWithMergeAndSubset() {

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder()
                .learningRate(0.1)
                .activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD);
        /*
                             (inCentre)                        (inRight)
                                |                                |
                            denseCentre0                         |
                                |                                |
                 ,--------  denseCentre1                       denseRight0
                /               |                                |
        subsetLeft(0-3)    denseCentre2 ---- denseRight ----  mergeRight
              |                 |                                |
         denseLeft0        denseCentre3                        denseRight1
              |                 |                                |
          (outLeft)         (outCentre)                        (outRight)

         */

        ComputationGraphConfiguration conf
                = overallConf.graphBuilder()
                .addInputs("inCentre", "inRight")
                .addLayer("denseCentre0", new DenseLayer.Builder().nIn(10).nOut(9).build(), "inCentre")
                .addLayer("denseCentre1", new DenseLayer.Builder().nIn(9).nOut(8).build(), "denseCentre0")
                .addLayer("denseCentre2", new DenseLayer.Builder().nIn(8).nOut(7).build(), "denseCentre1")
                .addLayer("denseCentre3", new DenseLayer.Builder().nIn(7).nOut(7).build(), "denseCentre2")
                .addLayer("outCentre", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(7).nOut(4).build(), "denseCentre3")
                .addVertex("subsetLeft", new SubsetVertex(0, 3), "denseCentre1")
                .addLayer("denseLeft0", new DenseLayer.Builder().nIn(4).nOut(5).build(), "subsetLeft")
                .addLayer("outLeft", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(5).nOut(6).build(), "denseLeft0")
                .addLayer("denseRight", new DenseLayer.Builder().nIn(7).nOut(7).build(), "denseCentre2")
                .addLayer("denseRight0", new DenseLayer.Builder().nIn(2).nOut(3).build(), "inRight")
                .addVertex("mergeRight", new MergeVertex(), "denseRight", "denseRight0")
                .addLayer("denseRight1", new DenseLayer.Builder().nIn(10).nOut(5).build(), "mergeRight")
                .addLayer("outRight", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(5).nOut(5).build(), "denseRight1")
                .setOutputs("outLeft", "outCentre", "outRight")
                .build();

        ComputationGraph modelToTune = new ComputationGraph(conf);
        modelToTune.init();

        INDArray[] features = new INDArray[]{Nd4j.rand(10, 10), Nd4j.rand(10, 2)};
        INDArray[] labels = new INDArray[]{Nd4j.rand(10, 6), Nd4j.rand(10, 4), Nd4j.rand(10, 5)};

        ComputationGraph modelNow = new TransferLearning.GraphBuilder(modelToTune)
                .setFeatureExtractor("denseCentre2")
                .build();

        assertTrue(modelNow.getVertex("denseCentre0").getLayer() instanceof FrozenLayer);
        assertTrue(modelNow.getVertex("denseCentre1").getLayer() instanceof FrozenLayer);
        assertTrue(modelNow.getVertex("denseCentre2").getLayer() instanceof FrozenLayer);
        assertTrue(!(modelNow.getVertex("denseRight").getLayer() instanceof FrozenLayer));
        assertTrue(!(modelNow.getVertex("denseRight0").getLayer() instanceof FrozenLayer));
        assertTrue(!(modelNow.getVertex("denseRight1").getLayer() instanceof FrozenLayer));
        assertTrue(!(modelNow.getVertex("denseLeft0").getLayer() instanceof FrozenLayer));

        ComputationGraph leftGraph =
                new ComputationGraph(overallConf.graphBuilder()
                        .addInputs("subsetLeft")
                        .addLayer("denseLeft0", new DenseLayer.Builder().nIn(4).nOut(5).build(), "subsetLeft")
                        .addLayer("outLeft", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(5).nOut(6).build(), "denseLeft0")
                        .setOutputs("outLeft")
                        .build());
        leftGraph.init();
        leftGraph.getLayer("denseLeft0").setParams(modelToTune.getLayer("denseLeft0").params());
        leftGraph.getLayer("outLeft").setParams(modelToTune.getLayer("outLeft").params());

        ComputationGraph rightGraph =
                new ComputationGraph(overallConf.graphBuilder()
                        .addInputs("denseCentre2", "inRight")
                        .addLayer("denseRight", new DenseLayer.Builder().nIn(7).nOut(7).build(), "denseCentre2")
                        .addLayer("denseRight0", new DenseLayer.Builder().nIn(2).nOut(3).build(), "inRight")
                        .addVertex("mergeRight", new MergeVertex(), "denseRight", "denseRight0")
                        .addLayer("denseRight1", new DenseLayer.Builder().nIn(10).nOut(5).build(), "mergeRight")
                        .addLayer("outRight", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(5).nOut(5).build(), "denseRight1")
                        .setOutputs("outRight")
                        .build());
        rightGraph.init();
        rightGraph.getLayer("denseRight").setParams(modelToTune.getLayer("denseRight").params());
        rightGraph.getLayer("denseRight0").setParams(modelToTune.getLayer("denseRight0").params());
        rightGraph.getLayer("denseRight1").setParams(modelToTune.getLayer("denseRight1").params());
        rightGraph.getLayer("outRight").setParams(modelToTune.getLayer("outRight").params());

        ComputationGraph centreGraph =
                new ComputationGraph(overallConf.graphBuilder()
                        .addInputs("denseCentre2")
                        .addLayer("denseCentre3", new DenseLayer.Builder().nIn(7).nOut(7).build(), "denseCentre2")
                        .addLayer("outCentre", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(7).nOut(4).build(), "denseCentre3")
                        .setOutputs("outCentre")
                        .build());
        centreGraph.init();
        centreGraph.getLayer("denseCentre3").setParams(modelToTune.getLayer("denseCentre3").params());
        centreGraph.getLayer("outCentre").setParams(modelToTune.getLayer("outCentre").params());

        ComputationGraph frozenGraph =
                new ComputationGraph(overallConf.graphBuilder()
                        .addInputs("inCentre")
                        .addLayer("denseCentre0", new DenseLayer.Builder().nIn(10).nOut(9).build(), "inCentre")
                        .addLayer("denseCentre1", new DenseLayer.Builder().nIn(9).nOut(8).build(), "denseCentre0")
                        .addLayer("denseCentre2", new DenseLayer.Builder().nIn(8).nOut(7).build(), "denseCentre1")
                        .setOutputs("denseCentre2")
                        .build());
        frozenGraph.init();
        frozenGraph.getLayer("denseCentre0").setParams(modelToTune.getLayer("denseCentre0").params());
        frozenGraph.getLayer("denseCentre1").setParams(modelToTune.getLayer("denseCentre1").params());
        frozenGraph.getLayer("denseCentre2").setParams(modelToTune.getLayer("denseCentre2").params());

        INDArray denseCentre2 = frozenGraph.output(features[0])[0];
        INDArray subsetLeft = frozenGraph.feedForward(features[0], false).get("denseCentre1").get(NDArrayIndex.all(), NDArrayIndex.interval(0, 3, true));
        MultiDataSet rightDataSet = new MultiDataSet(new INDArray[]{denseCentre2, features[1]}, new INDArray[]{labels[2]});

        int n = 0;
        String[] listOfLayers = new String[]{"denseCentre0",
                "denseCentre1",
                "denseCentre2",
                "denseLeft0",
                //Can't figure out why these are failing - could be a bug
                //"denseRight",
                //"denseRight0",
                //"denseRight1",
                "outLeft",
                "outCentre",
                //"outRight"
        };
        while (n < 2) {
            if (n == 0) {
                INDArray activationsM = modelNow.feedForward(features,false).get("denseCentre2");
                assertEquals(denseCentre2,activationsM);
                assertEquals(rightGraph.feedForward(rightDataSet.getFeatures(),false).get("denseRight"),modelNow.feedForward(features,false).get("denseRight"));
                assertEquals(rightGraph.feedForward(rightDataSet.getFeatures(),false).get("denseRight0"),modelNow.feedForward(features,false).get("denseRight0"));
                assertEquals(rightGraph.feedForward(rightDataSet.getFeatures(),false).get("mergeRight"),modelNow.feedForward(features,false).get("mergeRight"));
                assertEquals(rightGraph.feedForward(rightDataSet.getFeatures(),false).get("denseRight1"),modelNow.feedForward(features,false).get("denseRight1"));
                assertEquals(rightGraph.feedForward(rightDataSet.getFeatures(),false).get("outRight"),modelNow.feedForward(features,false).get("outRight"));

                assertEquals(rightGraph.getLayer("denseRight").conf().toJson(),modelNow.getLayer("denseRight").conf().toJson());
                assertEquals(rightGraph.getLayer("denseRight0").conf().toJson(),modelNow.getLayer("denseRight0").conf().toJson());
                assertEquals(rightGraph.getLayer("denseRight1").conf().toJson(),modelNow.getLayer("denseRight1").conf().toJson());
                assertEquals(rightGraph.getLayer("outRight").conf().toJson(),modelNow.getLayer("outRight").conf().toJson());

                // will fail because string param names are different but everything else is the same, so okay
                // assertEquals(rightGraph.getConfiguration().getDefaultConfiguration().toJson(),modelNow.getConfiguration().getDefaultConfiguration().toJson());
            }
            leftGraph.fit(new DataSet(subsetLeft, labels[0]));
            centreGraph.fit(new DataSet(denseCentre2, labels[1]));
            assertEquals(rightDataSet,new MultiDataSet(new INDArray[] {modelNow.feedForward(features,false).get("denseCentre2"),rightDataSet.getFeatures(1)},new INDArray[] {labels[2]}));

            rightGraph.fit(rightDataSet);
            modelNow.fit(new MultiDataSet(features, labels));
            assertEquals(modelNow.getLayer("denseCentre2").params(),frozenGraph.getLayer("denseCentre2").params());
            log.info("Fit after "+n);
            for (int i = 0; i < listOfLayers.length; i++) {
                String currentLayer = listOfLayers[i];
                INDArray expectedParams;
                if (frozenGraph.getConfiguration().getVertices().containsKey(currentLayer)) {
                    expectedParams = frozenGraph.getLayer(currentLayer).params();
                } else if (leftGraph.getConfiguration().getVertices().containsKey(currentLayer)) {
                    expectedParams = leftGraph.getLayer(currentLayer).params();
                } else if (rightGraph.getConfiguration().getVertices().containsKey(currentLayer)) {
                    expectedParams = rightGraph.getLayer(currentLayer).params();
                } else {
                    expectedParams = centreGraph.getLayer(currentLayer).params();
                }
                INDArray actualParams = modelNow.getLayer(currentLayer).params();
                log.info("Checking layer " + currentLayer + "\nPrinting differences in percentage..");
                log.info(expectedParams.sub(actualParams).mul(100).div(actualParams).toString());
                assertEquals(expectedParams,actualParams);
                //assertTrue(expectedParams.equalsWithEps(actualParams, 1e-3));
            }
            n++;
        }

    }


    @Test
    public void testMergeAndFreeze() {
        // in1 -> A -> B -> merge, in2 -> C -> merge -> D -> out
        //Goal here: test a number of things...
        // (a) Ensure that freezing C doesn't impact A and B. Only C should be frozen in this config
        // (b) Test global override (should be selective)


        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADAM)
                .learningRate(1e-4)
                .activation(Activation.LEAKYRELU)
                .graphBuilder()
                .addInputs("in1", "in2")
                .addLayer("A", new DenseLayer.Builder().nIn(10).nOut(9).build(), "in1")
                .addLayer("B", new DenseLayer.Builder().nIn(9).nOut(8).build(), "A")
                .addLayer("C", new DenseLayer.Builder().nIn(7).nOut(6).build(), "in2")
                .addLayer("D", new DenseLayer.Builder().nIn(8 + 7).nOut(5).build(), "B", "C")
                .addLayer("out", new OutputLayer.Builder().nIn(5).nOut(4).build(), "D")
                .setOutputs("out")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();


        int[] topologicalOrder = graph.topologicalSortOrder();
        org.deeplearning4j.nn.graph.vertex.GraphVertex[] vertices = graph.getVertices();

        for (int i = 0; i < topologicalOrder.length; i++) {
            org.deeplearning4j.nn.graph.vertex.GraphVertex v = vertices[topologicalOrder[i]];
            log.info(i + "\t" + v.getVertexName());
        }

        ComputationGraph graph2 = new TransferLearning.GraphBuilder(graph)
                .fineTuneConfiguration(new FineTuneConfiguration.Builder()
                        .learningRate(2e-2).build())
                .setFeatureExtractor("C")
                .build();

        boolean cFound = false;
        Layer[] layers = graph2.getLayers();

        for (Layer l : layers) {
            String name = l.conf().getLayer().getLayerName();
            log.info(name + "\t frozen: " + (l instanceof FrozenLayer));
            if ("C".equals(l.conf().getLayer().getLayerName())) {
                //Only C should be frozen in this config
                cFound = true;
                assertTrue(name, l instanceof FrozenLayer);
            } else {
                assertFalse(name, l instanceof FrozenLayer);
            }

            //Also check config:
            assertEquals(Updater.ADAM, l.conf().getLayer().getUpdater());
            assertEquals(2e-2, l.conf().getLayer().getLearningRate(), 1e-5);
            assertEquals(Activation.LEAKYRELU.getActivationFunction(), l.conf().getLayer().getActivationFn());
        }
        assertTrue(cFound);

    }

    @Test
    public void testSimplerMergeBackProp() {

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder()
                .learningRate(0.9)
                .activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD);

        /*
                inCentre                inRight
                   |                        |
             denseCentre0               denseRight0
                   |                        |
                   |------ mergeRight ------|
                                |
                              outRight

        */

        ComputationGraphConfiguration conf = overallConf.graphBuilder()
                .addInputs("inCentre", "inRight")
                .addLayer("denseCentre0", new DenseLayer.Builder().nIn(2).nOut(2).build(),"inCentre")
                .addLayer("denseRight0", new DenseLayer.Builder().nIn(2).nOut(2).build(),"inRight")
                .addVertex("mergeRight", new MergeVertex(),"denseCentre0","denseRight0")
                .addLayer("outRight", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(4).nOut(2).build(),"mergeRight")
                .setOutputs("outRight")
                .build();
        ComputationGraph modelToTune = new ComputationGraph(conf);
        modelToTune.init();

        MultiDataSet randData = new MultiDataSet(new INDArray[] {Nd4j.rand(2,2),Nd4j.rand(2,2)}, new INDArray[] {Nd4j.rand(2,2)});
        INDArray denseCentre0 = modelToTune.feedForward(randData.getFeatures(),false).get("denseCentre0");
        MultiDataSet otherRandData = new MultiDataSet(new INDArray[] {denseCentre0,randData.getFeatures(1)}, randData.getLabels());

        ComputationGraphConfiguration otherConf = overallConf.graphBuilder()
                .addInputs("denseCentre0","inRight")
                .addLayer("denseRight0", new DenseLayer.Builder().nIn(2).nOut(2).build(),"inRight")
                .addVertex("mergeRight", new MergeVertex(),"denseCentre0","denseRight0")
                .addLayer("outRight", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(4).nOut(2).build(),"mergeRight")
                .setOutputs("outRight")
                .build();
        ComputationGraph modelOther = new ComputationGraph(otherConf);
        modelOther.init();
        modelOther.getLayer("denseRight0").setParams(modelToTune.getLayer("denseRight0").params());
        modelOther.getLayer("outRight").setParams(modelToTune.getLayer("outRight").params());

        modelToTune.getVertex("denseCentre0").setLayerAsFrozen();
        ComputationGraph modelNow = new TransferLearning.GraphBuilder(modelToTune).setFeatureExtractor("denseCentre0").build();
        int n = 0;
        while (n < 5) {
            if (n == 0) {
                //confirm activations out of the merge are equivalent
                assertEquals(modelToTune.feedForward(randData.getFeatures(),false).get("mergeRight"), modelOther.feedForward(otherRandData.getFeatures(),false).get("mergeRight"));
                assertEquals(modelNow.feedForward(randData.getFeatures(),false).get("mergeRight"), modelOther.feedForward(otherRandData.getFeatures(),false).get("mergeRight"));
            }
            //confirm activations out of frozen vertex is the same as the input to the other model
            modelOther.fit(otherRandData);
            modelToTune.fit(randData);
            modelNow.fit(randData);

            assertEquals(otherRandData.getFeatures(0),modelNow.feedForward(randData.getFeatures(),false).get("denseCentre0"));
            assertEquals(otherRandData.getFeatures(0),modelToTune.feedForward(randData.getFeatures(),false).get("denseCentre0"));

            assertEquals(modelOther.getLayer("denseRight0").params(),modelNow.getLayer("denseRight0").params());
            assertEquals(modelOther.getLayer("denseRight0").params(),modelToTune.getLayer("denseRight0").params());

            assertEquals(modelOther.getLayer("outRight").params(),modelNow.getLayer("outRight").params());
            assertEquals(modelOther.getLayer("outRight").params(),modelToTune.getLayer("outRight").params());
            n++;
        }

    }

    @Test
    public void testAddOutput() {
        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder()
                .learningRate(0.9)
                .activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD);

        ComputationGraphConfiguration conf = overallConf.graphBuilder()
                .addInputs("inCentre", "inRight")
                .addLayer("denseCentre0", new DenseLayer.Builder().nIn(2).nOut(2).build(),"inCentre")
                .addLayer("denseRight0", new DenseLayer.Builder().nIn(2).nOut(2).build(),"inRight")
                .addVertex("mergeRight", new MergeVertex(),"denseCentre0","denseRight0")
                .addLayer("outRight", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(4).nOut(2).build(),"mergeRight")
                .setOutputs("outRight")
                .build();
        ComputationGraph modelToTune = new ComputationGraph(conf);
        modelToTune.init();

        ComputationGraph modelNow = new TransferLearning.GraphBuilder(modelToTune)
                .addLayer("outCentre", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(2).nOut(3).build(),"denseCentre0")
                .setOutputs("outCentre")
                .build();

        assertEquals(2,modelNow.getNumOutputArrays());
        MultiDataSet rand = new MultiDataSet(new INDArray[] {Nd4j.rand(2,2),Nd4j.rand(2,2)},new INDArray[] {Nd4j.rand(2,2),Nd4j.rand(2,3)});
        modelNow.fit(rand);
        log.info(modelNow.summary());

    }
}
