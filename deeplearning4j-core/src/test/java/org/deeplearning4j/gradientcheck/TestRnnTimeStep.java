package org.deeplearning4j.gradientcheck;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestRnnTimeStep {

//    public static final ILogger LOG = LoggerFactory.create(RecurrentSequenceEmbedding.class);
    public static final Logger LOG = LoggerFactory.getLogger(TestRnnTimeStep.class);

    /** recurrent neural network */
    private final MultiLayerNetwork net;
    private final int nIn;
    private final int nOut;
    public AtomicLong mid;

    public TestRnnTimeStep(int nIn, int nOut) {
        this.nIn = nIn;
        this.nOut = nOut; //nIn, nOut can also be inferred from the net
        this.net = TestRnnTimeStep.buildRecurrentNetwork(nIn, nOut);
        this.mid = new AtomicLong();
    }

    public static MultiLayerNetwork buildRecurrentNetwork(int nIn, int nOut) {

        int lstmLayerSize = 100;
        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.2)
                .momentum(0.5)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .list(2)
                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayerSize)
                        .updater(Updater.RMSPROP)
                        .activation("hardtanh").weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.08, 0.08)).build())
                .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(nOut)
                        .updater(Updater.RMSPROP)
                        .activation("hardtanh").weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.08, 0.08)).build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net_ = new MultiLayerNetwork(conf);
        net_.init();
        return net_;
    }

    /** makes prediction on a time-series data using the recurrent-network given input and recurrent
     * input and returns all the intermediate embeddings including the final embedding */
    public Map<String, INDArray>[] getEmbedding(INDArray input, final Map<String, INDArray>[] rnnState) {

		/* CHECK The Code below for whether vectors are being copied or being referenced. If referenced
		 * then there might be bugs. */
        final int numLayers = this.net.getnLayers();
        INDArray result;

        MultiLayerNetwork mynet = this.net;

        //set the rnn activations
        if(rnnState != null) {
            for(int layer = 0; layer < rnnState.length; layer++) {
                mynet.rnnSetPreviousState(layer, rnnState[layer]);
            }
        } else {
            mynet.rnnClearPreviousState();
        }

        long start = System.currentTimeMillis();
        //do feedforwarding
        result = mynet.rnnTimeStep(input);
        long end = System.currentTimeMillis();

        //get the new activations
        @SuppressWarnings("unchecked")
        Map<String, INDArray>[] newRNNState = new HashMap[numLayers];

        for(int i = 0; i < numLayers; i++) {
            newRNNState[i] = mynet.rnnGetPreviousState(i);
        }

        this.mid.addAndGet(end - start);

        INDArray embedding = Nd4j.zeros(this.nOut);
        for(int i = 0; i < this.nOut; i++)
            embedding.putScalar(i, result.getDouble(new int[]{0, i, 0}));

        return newRNNState;
    }

    public static void main(String[] args) throws Exception {

        int nIn = 61;
        int nOut = 25;
        TestRnnTimeStep test = new TestRnnTimeStep(nIn, nOut);

        //create random inputs
//        for(int i=0; i<=400; i++) {
        int nTests = 5000;
        int nSteps = 3;
//        for(int i=0; i<=nTests; i++) {
//            Map<String, INDArray>[] rnnState = null;
//
//            for(int step=0; step<nSteps; step++) {
//
//                INDArray input = Nd4j.zeros(new int[]{1, nIn, 1});
//                for(int j=0; j<nIn; j++) {
//                    input.putScalar(new int[]{0, j, 0}, Math.random());
//                }
//
//                rnnState = test.getEmbedding(input, rnnState);
//            }
//        }

        INDArray input = Nd4j.rand(new int[]{1, nIn, 1}); //Nd4j.zeros(new int[]{1, nIn, 1});
//        for(int j=0; j<nIn; j++) {
//            input.putScalar(new int[]{0, j, 0}, Math.random());
//        }
        for(int i=0; i<=nTests; i++) {
            Map<String, INDArray>[] rnnState = null;
            for(int step=0; step<nSteps; step++) {
                rnnState = test.getEmbedding(input, rnnState);
            }
        }

        long time = test.mid.get();
        System.out.println("Total Time taken by rnn time step is "+time+" ms");
        System.out.println("Average Time taken by rnn time step is "+time/(double)(nTests*nSteps)+" ms");
    }
}