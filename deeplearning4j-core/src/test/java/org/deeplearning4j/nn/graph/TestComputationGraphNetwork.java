package org.deeplearning4j.nn.graph;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.canova.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.*;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class TestComputationGraphNetwork {

    private static ComputationGraphConfiguration getIrisGraphConfiguration(){
        return new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .addInputs("input")
                .addLayer("firstLayer", new DenseLayer.Builder().nIn(4).nOut(5).build(), "input")
                .addLayer("outputLayer", new OutputLayer.Builder().nIn(5).nOut(3).build(), "firstLayer")
                .setOutputs("outputLayer")
                .pretrain(false).backprop(true)
                .build();
    }

    private static MultiLayerConfiguration getIrisMLNConfiguration(){
        return new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(5).build())
                .layer(1, new OutputLayer.Builder().nIn(5).nOut(3).build())
                .pretrain(false).backprop(true)
                .build();
    }

    private static int getNumParams(){
        //Number of parameters for both iris models
        return (4*5+5) + (5*3+3);
    }

    @Test
    public void testConfigurationBasic(){

        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();

        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        //Get topological sort order
        int[] order = graph.topologicalSortOrder();
        int[] expOrder = new int[]{0,1,2};
        assertArrayEquals(expOrder,order);  //Only one valid order: 0 (input) -> 1 (firstlayer) -> 2 (outputlayer)

        INDArray params = graph.params();
        assertNotNull(params);

        int nParams = getNumParams();
        assertEquals(nParams,params.length());

        INDArray arr = Nd4j.linspace(0,nParams,nParams);
        assertEquals(nParams,arr.length());

        graph.setParams(arr);
        params = graph.params();
        assertEquals(arr,params);

        //Number of inputs and outputs:
        assertEquals(1,graph.getNumInputArrays());
        assertEquals(1,graph.getNumOutputArrays());
    }

    @Test
    public void testForwardBasicIris(){

        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlc = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        DataSetIterator iris = new IrisDataSetIterator(150,150);
        DataSet ds = iris.next();

        graph.setInput(0, ds.getFeatureMatrix());
        Map<String,INDArray> activations = graph.feedForward(false);
        assertEquals(3,activations.size()); //2 layers + 1 input node
        assertTrue(activations.containsKey("input"));
        assertTrue(activations.containsKey("firstLayer"));
        assertTrue(activations.containsKey("outputLayer"));

        //Now: set parameters of both networks to be identical. Then feedforward, and check we get the same outputs
        Nd4j.getRandom().setSeed(12345);
        int nParams = getNumParams();
        INDArray params = Nd4j.rand(1, nParams);
        graph.setParams(params.dup());
        net.setParams(params.dup());

        List<INDArray> mlnAct = net.feedForward(ds.getFeatureMatrix(),false);
        activations = graph.feedForward(ds.getFeatureMatrix(), false);

        assertEquals(mlnAct.get(0),activations.get("input"));
        assertEquals(mlnAct.get(1),activations.get("firstLayer"));
        assertEquals(mlnAct.get(2),activations.get("outputLayer"));
    }

    @Test
    public void testBackwardIrisBasic(){
        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlc = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        DataSetIterator iris = new IrisDataSetIterator(150,150);
        DataSet ds = iris.next();

        //Now: set parameters of both networks to be identical. Then feedforward, and check we get the same outputs
        Nd4j.getRandom().setSeed(12345);
        int nParams = (4*5+5) + (5*3+3);
        INDArray params = Nd4j.rand(1, nParams);
        graph.setParams(params.dup());
        net.setParams(params.dup());

        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();
        graph.setInput(0, input.dup());
        graph.setLabel(0, labels.dup());

        net.setInput(input.dup());
        net.setLabels(labels.dup());

        //Compute gradients
        net.computeGradientAndScore();
        Pair<Gradient,Double> netGradScore = net.gradientAndScore();

        graph.computeGradientAndScore();
        Pair<Gradient,Double> graphGradScore = graph.gradientAndScore();

        assertEquals(netGradScore.getSecond(), graphGradScore.getSecond(), 1e-3);

        //Compare gradients
        Gradient netGrad = netGradScore.getFirst();
        Gradient graphGrad = graphGradScore.getFirst();

        assertNotNull(graphGrad);
        assertEquals(netGrad.gradientForVariable().size(), graphGrad.gradientForVariable().size());

        assertEquals(netGrad.getGradientFor("0_W"), graphGrad.getGradientFor("firstLayer_W"));
        assertEquals(netGrad.getGradientFor("0_b"),graphGrad.getGradientFor("firstLayer_b"));
        assertEquals(netGrad.getGradientFor("1_W"), graphGrad.getGradientFor("outputLayer_W"));
        assertEquals(netGrad.getGradientFor("1_b"),graphGrad.getGradientFor("outputLayer_b"));
    }

    @Test
    public void testIrisFit(){

        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlnConfig = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlnConfig);
        net.init();

        Nd4j.getRandom().setSeed(12345);
        int nParams = getNumParams();
        INDArray params = Nd4j.rand(1,nParams);

        graph.setParams(params.dup());
        net.setParams(params.dup());


        DataSetIterator iris = new IrisDataSetIterator(75,150);

        net.fit(iris);
        iris.reset();

        graph.fit(iris);

        //Check that parameters are equal for both models after fitting:
        INDArray paramsMLN = net.params();
        INDArray paramsGraph = graph.params();

        assertNotEquals(params,paramsGraph);
        assertEquals(paramsMLN,paramsGraph);
    }

    @Test
    public void testIrisFitMultiDataSetIterator() throws Exception {

        RecordReader rr = new CSVRecordReader(0,",");
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

        MultiDataSetIterator iter = new RecordReaderMultiDataSetIterator.Builder(10)
                .addReader("iris",rr)
                .addInput("iris",0,3)
                .addOutputOneHot("iris",4,3)
                .build();

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.1)
                .graphBuilder()
                .addInputs("in")
                .addLayer("dense",new DenseLayer.Builder().nIn(4).nOut(2).build(),"in")
                .addLayer("out",new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(2).nOut(3).build(),"dense")
                .setOutputs("out")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph cg = new ComputationGraph(config);
        cg.init();

        cg.fit(iter);


        rr.reset();
        iter = new RecordReaderMultiDataSetIterator.Builder(10)
                .addReader("iris",rr)
                .addInput("iris",0,3)
                .addOutputOneHot("iris",4,3)
                .build();
        while(iter.hasNext()){
            cg.fit(iter.next());
        }
    }

    @Test
    public void testCloning(){
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        ComputationGraph g2 = graph.clone();

        DataSetIterator iris = new IrisDataSetIterator(150,150);
        INDArray in = iris.next().getFeatureMatrix();
        Map<String,INDArray> activations = graph.feedForward(in, false);
        Map<String,INDArray> activations2 = g2.feedForward(in,false);
        assertEquals(activations,activations2);
    }

    @Test
    public void testScoringDataSet(){
        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlc = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        DataSetIterator iris = new IrisDataSetIterator(150,150);
        DataSet ds = iris.next();

        //Now: set parameters of both networks to be identical. Then feedforward, and check we get the same score
        Nd4j.getRandom().setSeed(12345);
        int nParams = getNumParams();
        INDArray params = Nd4j.rand(1, nParams);
        graph.setParams(params.dup());
        net.setParams(params.dup());

        double scoreMLN = net.score(ds, false);
        double scoreCG = graph.score(ds, false);

        assertEquals(scoreMLN,scoreCG,1e-4);
    }

    @Test
    public void testPreprocessorAddition(){
        //Also check that nIns are set automatically
        //First: check FF -> RNN
        ComputationGraphConfiguration conf1 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .setInputTypes(InputType.feedForward(5))
                .addLayer("rnn", new GravesLSTM.Builder().nOut(5).build(), "in")
                .addLayer("out", new RnnOutputLayer.Builder().nOut(5).build(), "rnn")
                .setOutputs("out")
                .build();

        assertEquals(5, ((FeedForwardLayer)((LayerVertex) conf1.getVertices().get("rnn")).getLayerConf().getLayer()).getNIn());
        assertEquals(5, ((FeedForwardLayer)((LayerVertex) conf1.getVertices().get("out")).getLayerConf().getLayer()).getNIn());

        LayerVertex lv1 = (LayerVertex) conf1.getVertices().get("rnn");
        assertTrue(lv1.getPreProcessor() instanceof FeedForwardToRnnPreProcessor);
        LayerVertex lv2 = (LayerVertex) conf1.getVertices().get("out");
        assertNull(lv2.getPreProcessor());

        //Check RNN -> FF -> RNN
        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .setInputTypes(InputType.recurrent(5))
                .addLayer("ff", new DenseLayer.Builder().nOut(5).build(), "in")
                .addLayer("out",new RnnOutputLayer.Builder().nOut(5).build(),"ff")
                .setOutputs("out")
                .build();

        assertEquals(5, ((FeedForwardLayer)((LayerVertex) conf2.getVertices().get("ff")).getLayerConf().getLayer()).getNIn());
        assertEquals(5, ((FeedForwardLayer)((LayerVertex) conf2.getVertices().get("out")).getLayerConf().getLayer()).getNIn());

        lv1 = (LayerVertex) conf2.getVertices().get("ff");
        assertTrue(lv1.getPreProcessor() instanceof RnnToFeedForwardPreProcessor);
        lv2 = (LayerVertex) conf2.getVertices().get("out");
        assertTrue(lv2.getPreProcessor() instanceof FeedForwardToRnnPreProcessor);

        //CNN -> Dense
        ComputationGraphConfiguration conf3 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .setInputTypes(InputType.convolutional(1,28,28))
                .addLayer("cnn",new ConvolutionLayer.Builder().kernelSize(2,2).padding(0,0).stride(2,2).nOut(3).build(),"in")   //(28-2+0)/2+1 = 14
                .addLayer("pool", new SubsamplingLayer.Builder().kernelSize(2, 2).padding(0, 0).stride(2, 2).build(), "cnn")   //(14-2+0)/2+1=7
                .addLayer("dense", new DenseLayer.Builder().nOut(10).build(), "pool")
                .addLayer("out", new OutputLayer.Builder().nIn(10).nOut(5).build(), "dense")
                .setOutputs("out")
                .build();
            //Check preprocessors:
        lv1 = (LayerVertex) conf3.getVertices().get("cnn");
        assertNotNull(lv1.getPreProcessor());
        FeedForwardToCnnPreProcessor ffcnn = (FeedForwardToCnnPreProcessor)lv1.getPreProcessor();
        assertEquals(1, ffcnn.getNumChannels());
        assertEquals(28, ffcnn.getInputHeight());
        assertEquals(28, ffcnn.getInputWidth());
        lv2 = (LayerVertex) conf3.getVertices().get("pool");
        assertNull(lv2.getPreProcessor());
        LayerVertex lv3 = (LayerVertex) conf3.getVertices().get("dense");
        assertTrue(lv3.getPreProcessor() instanceof CnnToFeedForwardPreProcessor);
        CnnToFeedForwardPreProcessor proc = (CnnToFeedForwardPreProcessor) lv3.getPreProcessor();
        assertEquals(3, proc.getNumChannels());
        assertEquals(7, proc.getInputHeight());
        assertEquals(7, proc.getInputWidth());
        LayerVertex lv4 = (LayerVertex) conf3.getVertices().get("out");
        assertNull(lv4.getPreProcessor());
            //Check nIns:
        assertEquals(7*7*3,((FeedForwardLayer)lv3.getLayerConf().getLayer()).getNIn());

        //CNN->Dense, RNN->Dense, Dense->RNN
        ComputationGraphConfiguration conf4 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("inCNN", "inRNN")
                .setInputTypes(InputType.convolutional(1, 28, 28), InputType.recurrent(5))
                .addLayer("cnn", new ConvolutionLayer.Builder().kernelSize(2, 2).padding(0, 0).stride(2, 2).nOut(3).build(), "inCNN")   //(28-2+0)/2+1 = 14
                .addLayer("pool", new SubsamplingLayer.Builder().kernelSize(2, 2).padding(0, 0).stride(2, 2).build(), "cnn")   //(14-2+0)/2+1=7
                .addLayer("dense", new DenseLayer.Builder().nOut(10).build(), "pool")
                .addLayer("dense2", new DenseLayer.Builder().nOut(10).build(), "inRNN")
                .addVertex("merge", new MergeVertex(), "dense", "dense2")
                .addLayer("out", new RnnOutputLayer.Builder().nOut(5).build(), "merge")
                .setOutputs("out")
                .build();

        //Check preprocessors:
        lv1 = (LayerVertex) conf4.getVertices().get("cnn");
        assertNotNull(lv1.getPreProcessor());
        ffcnn = (FeedForwardToCnnPreProcessor)lv1.getPreProcessor();
        assertEquals(1, ffcnn.getNumChannels());
        assertEquals(28, ffcnn.getInputHeight());
        assertEquals(28, ffcnn.getInputWidth());
        lv2 = (LayerVertex) conf4.getVertices().get("pool");
        assertNull(lv2.getPreProcessor());
        lv3 = (LayerVertex) conf4.getVertices().get("dense");
        assertTrue(lv3.getPreProcessor() instanceof CnnToFeedForwardPreProcessor);
        proc = (CnnToFeedForwardPreProcessor) lv3.getPreProcessor();
        assertEquals(3, proc.getNumChannels());
        assertEquals(7, proc.getInputHeight());
        assertEquals(7, proc.getInputWidth());
        lv4 = (LayerVertex) conf4.getVertices().get("dense2");
        assertTrue(lv4.getPreProcessor() instanceof RnnToFeedForwardPreProcessor);
        LayerVertex lv5 = (LayerVertex) conf4.getVertices().get("out");
        assertTrue(lv5.getPreProcessor() instanceof FeedForwardToRnnPreProcessor);
            //Check nIns:
        assertEquals(7 * 7 * 3, ((FeedForwardLayer) lv3.getLayerConf().getLayer()).getNIn());
        assertEquals(5, ((FeedForwardLayer)lv4.getLayerConf().getLayer()).getNIn());
        assertEquals(10, ((FeedForwardLayer)lv5.getLayerConf().getLayer()).getNIn());



        //Input to 2 CNN layers:
        ComputationGraphConfiguration conf5 = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(1,28,28))
                .addLayer("cnn_1", new ConvolutionLayer.Builder(2,2).stride(2, 2).nIn(1).nOut(3).build(), "input")
                .addLayer("cnn_2", new ConvolutionLayer.Builder(4,4).stride(2,2).padding(1,1).nIn(1).nOut(3).build(), "input")
                .addLayer("max_1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).build(), "cnn_1", "cnn_2")
                .addLayer("output", new OutputLayer.Builder().nOut(10).build(), "max_1")    //.nIn(7 * 7 * 6)
                .setOutputs("output")
                .pretrain(false).backprop(true)
                .build();
        lv1 = (LayerVertex) conf5.getVertices().get("cnn_1");
        assertTrue(lv1.getPreProcessor() instanceof FeedForwardToCnnPreProcessor);
        ffcnn = (FeedForwardToCnnPreProcessor)lv1.getPreProcessor();
        assertEquals(1, ffcnn.getNumChannels());
        assertEquals(28, ffcnn.getInputWidth());
        assertEquals(28, ffcnn.getInputHeight());

        lv2 = (LayerVertex) conf5.getVertices().get("cnn_2");
        assertTrue(lv2.getPreProcessor() instanceof FeedForwardToCnnPreProcessor);
        FeedForwardToCnnPreProcessor ffcnn2 = (FeedForwardToCnnPreProcessor)lv2.getPreProcessor();
        assertEquals(1, ffcnn2.getNumChannels());
        assertEquals(28, ffcnn2.getInputWidth());
        assertEquals(28, ffcnn2.getInputHeight());

        assertNull(((LayerVertex) conf5.getVertices().get("max_1")).getPreProcessor());

        lv3 = (LayerVertex) conf5.getVertices().get("output");
        assertTrue(lv3.getPreProcessor() instanceof CnnToFeedForwardPreProcessor);
        CnnToFeedForwardPreProcessor cnnff = (CnnToFeedForwardPreProcessor)lv3.getPreProcessor();
        assertEquals(6, cnnff.getNumChannels());
        assertEquals(7, cnnff.getInputHeight());
        assertEquals(7, cnnff.getInputWidth());
    }

    @Test
    public void testCompGraphUnderscores(){
        //Problem: underscores in names could be problematic for ComputationGraphUpdater, HistogramIterationListener

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .addInputs("input")
                .addLayer("first_layer", new DenseLayer.Builder().nIn(4).nOut(5).build(), "input")
                .addLayer("output_layer", new OutputLayer.Builder().nIn(5).nOut(3).build(), "first_layer")
                .setOutputs("output_layer")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        DataSetIterator iris = new IrisDataSetIterator(10,150);
        while(iris.hasNext()){
            net.fit(iris.next());
        }
    }

    @Test
    public void testPreTraining(){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(100)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.SGD).learningRate(1e-6)
                .regularization(true)
                .l2(2e-4)
                .graphBuilder()
                .addInputs("in")
                .addLayer("layer0", new RBM.Builder(RBM.HiddenUnit.GAUSSIAN, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
                        .activation("tanh")
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build(),"in")
                .addLayer("layer1", new RBM.Builder(RBM.HiddenUnit.GAUSSIAN, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
                        .activation("tanh")
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build(),"in")
                .addLayer("layer2", new RBM.Builder(RBM.HiddenUnit.GAUSSIAN, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(3).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
                        .activation("tanh")
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build(),"layer1")
                .addLayer("out", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3+3).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
                        .activation("softmax").build(), "layer0","layer2")
                .setOutputs("out")
                .pretrain(true).backprop(false)
                .build();


        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator iter = new IrisDataSetIterator(10, 150);
        net.fit(iter);
    }

}
