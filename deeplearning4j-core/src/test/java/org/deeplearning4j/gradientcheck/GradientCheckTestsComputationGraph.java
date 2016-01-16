package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertTrue;

public class GradientCheckTestsComputationGraph {

    public static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;

    @Before
    public void before(){
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        NDArrayFactory factory = Nd4j.factory();
        factory.setDType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testBasicIris(){
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
                .updater(Updater.NONE).learningRate(1.0)
                .graphBuilder()
                .addInputs("input")
                .addLayer("firstLayer", new DenseLayer.Builder().nIn(4).nOut(5).activation("tanh").build(), "input")
                .addLayer("outputLayer", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax").nIn(5).nOut(3).build(), "firstLayer")
                .setOutputs("outputLayer")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Nd4j.getRandom().setSeed(12345);
        int nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(1,nParams);
        graph.setParams(newParams);

        DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        if( PRINT_RESULTS ){
            System.out.println("testBasicIris()" );
            for( int j=0; j<graph.getNumLayers(); j++ ) System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[]{input}, new INDArray[]{labels});

        String msg = "testBasicIris()";
        assertTrue(msg,gradOK);
    }

}
