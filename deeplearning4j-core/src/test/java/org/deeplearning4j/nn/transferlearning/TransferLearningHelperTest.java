package org.deeplearning4j.nn.transferlearning;

import lombok.extern.slf4j.Slf4j;
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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;

/**
 * Created by susaneraly on 2/24/17.
 */
@Slf4j
public class TransferLearningHelperTest {

    @Test
    public void tesUnfrozenSubset() {

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder()
                .learningRate(0.1)
                .seed(124)
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
        modelToTune.getVertex("denseCentre0").setLayerAsFrozen();
        modelToTune.getVertex("denseCentre1").setLayerAsFrozen();
        modelToTune.getVertex("denseCentre2").setLayerAsFrozen();

        TransferLearningHelper helper = new TransferLearningHelper(modelToTune);

        ComputationGraph modelSubset = helper.unfrozenGraph();

        ComputationGraphConfiguration expectedConf
                = overallConf.graphBuilder()
                .addInputs("denseCentre1", "denseCentre2", "inRight") //inputs are in sorted order
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
        ComputationGraph expectedModel = new ComputationGraph(expectedConf);
        expectedModel.init();
        assertEquals(expectedConf.toJson(), modelSubset.getConfiguration().toJson());
    }

    @Test
    public void testFitUnFrozen() {

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder()
                .learningRate(0.1)
                .seed(124)
                .activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD);

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
        modelToTune.getVertex("denseCentre0").setLayerAsFrozen();
        modelToTune.getVertex("denseCentre1").setLayerAsFrozen();
        modelToTune.getVertex("denseCentre2").setLayerAsFrozen();

        INDArray inRight = Nd4j.rand(10, 2);
        INDArray inCentre = Nd4j.rand(10, 10);
        INDArray outLeft = Nd4j.rand(10, 6);
        INDArray outRight = Nd4j.rand(10, 5);
        INDArray outCentre = Nd4j.rand(10, 4);
        MultiDataSet origData = new MultiDataSet(new INDArray[]{inCentre, inRight}, new INDArray[]{outLeft, outCentre, outRight});
        ComputationGraph modelIdentical = modelToTune.clone();

        TransferLearningHelper helper = new TransferLearningHelper(modelToTune);
        MultiDataSet featurizedDataSet = helper.featurize(origData);

        modelIdentical.fit(origData);
        helper.fitFeaturized(featurizedDataSet);

        assertEquals(modelIdentical.getLayer("denseCentre0").params(), modelToTune.getLayer("denseCentre0").params());
        assertEquals(modelIdentical.getLayer("denseCentre1").params(), modelToTune.getLayer("denseCentre1").params());
        assertEquals(modelIdentical.getLayer("denseCentre2").params(), modelToTune.getLayer("denseCentre2").params());
        assertEquals(modelIdentical.getLayer("denseCentre3").params(), modelToTune.getLayer("denseCentre3").params());
        assertEquals(modelIdentical.getLayer("outCentre").params(), modelToTune.getLayer("outCentre").params());
        //assertEquals(modelIdentical.getLayer("denseRight0").params(),modelToTune.getLayer("denseRight0").params());
        //assertEquals(modelIdentical.getLayer("denseRight1").params(),modelToTune.getLayer("denseRight1").params());
        //assertEquals(modelIdentical.getLayer("outRight").params(),modelToTune.getLayer("outRight").params());
        assertEquals(modelIdentical.getLayer("denseLeft0").params(), modelToTune.getLayer("denseLeft0").params());
        assertEquals(modelIdentical.getLayer("outLeft").params(), modelToTune.getLayer("outLeft").params());

        log.info(modelIdentical.summary());
        log.info(helper.unfrozenGraph().summary());

    }
}
