package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.TimeDistributed;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class TestTimeDistributed extends BaseDL4JTest {

    private RNNFormat rnnDataFormat;

    public TestTimeDistributed(RNNFormat rnnDataFormat){
        this.rnnDataFormat = rnnDataFormat;
    }
    @Parameterized.Parameters
    public static Object[] params(){
        return RNNFormat.values();
    }
    @Test
    public void testTimeDistributed(){
        for(WorkspaceMode wsm : new WorkspaceMode[]{WorkspaceMode.ENABLED, WorkspaceMode.NONE}) {

            MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder()
                    .trainingWorkspaceMode(wsm)
                    .inferenceWorkspaceMode(wsm)
                    .seed(12345)
                    .updater(new Adam(0.1))
                    .list()
                    .layer(new LSTM.Builder().nIn(3).nOut(3).dataFormat(rnnDataFormat).build())
                    .layer(new DenseLayer.Builder().nIn(3).nOut(3).activation(Activation.TANH).build())
                    .layer(new RnnOutputLayer.Builder().nIn(3).nOut(3).activation(Activation.SOFTMAX).dataFormat(rnnDataFormat)
                            .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                    .setInputType(InputType.recurrent(3, rnnDataFormat))
                    .build();

            MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                    .trainingWorkspaceMode(wsm)
                    .inferenceWorkspaceMode(wsm)
                    .seed(12345)
                    .updater(new Adam(0.1))
                    .list()
                    .layer(new LSTM.Builder().nIn(3).nOut(3).dataFormat(rnnDataFormat).build())
                    .layer(new TimeDistributed(new DenseLayer.Builder().nIn(3).nOut(3).activation(Activation.TANH).build(), rnnDataFormat))
                    .layer(new RnnOutputLayer.Builder().nIn(3).nOut(3).activation(Activation.SOFTMAX).dataFormat(rnnDataFormat)
                            .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                    .setInputType(InputType.recurrent(3, rnnDataFormat))
                    .build();

            MultiLayerNetwork net1 = new MultiLayerNetwork(conf1);
            MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
            net1.init();
            net2.init();

            for( int mb : new int[]{1, 5}) {
                for(char inLabelOrder : new char[]{'c', 'f'}) {
                    INDArray in = Nd4j.rand(DataType.FLOAT, mb, 3, 5).dup(inLabelOrder);
                    if (rnnDataFormat == RNNFormat.NWC){
                        in = in.permute(0, 2, 1);
                    }
                    INDArray out1 = net1.output(in);
                    INDArray out2 = net2.output(in);
                    assertEquals(out1, out2);

                    INDArray labels ;
                    if (rnnDataFormat == RNNFormat.NCW) {
                        labels = TestUtils.randomOneHotTimeSeries(mb, 3, 5).dup(inLabelOrder);
                    }else{
                        labels = TestUtils.randomOneHotTimeSeries(mb, 5, 3).dup(inLabelOrder);
                    }



                    DataSet ds = new DataSet(in, labels);
                    net1.fit(ds);
                    net2.fit(ds);

                    assertEquals(net1.params(), net2.params());

                    MultiLayerNetwork net3 = TestUtils.testModelSerialization(net2);
                    out2 = net2.output(in);
                    INDArray out3 = net3.output(in);

                    assertEquals(out2, out3);
                }
            }
        }
    }
}
