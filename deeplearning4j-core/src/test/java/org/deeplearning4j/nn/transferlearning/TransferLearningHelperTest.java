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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;

/**
 * Created by susaneraly on 2/24/17.
 */
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
                .addInputs("denseCentre1","denseCentre2", "inRight") //inputs are in sorted order
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
        assertEquals(expectedConf.toJson(),modelSubset.getConfiguration().toJson());
    }
}
