package org.deeplearning4j.nn.dtypes;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

import static org.junit.Assert.*;

public class DTypeTests extends BaseDL4JTest {

    @Test
    public void testMultiLayerNetworkTypeConversion(){

        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder().activation(Activation.TANH).nIn(10).nOut(10).build())
                .layer(new DenseLayer.Builder().activation(Activation.TANH).nIn(10).nOut(10).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray inD = Nd4j.rand(DataType.DOUBLE, 1, 10);
        INDArray lD = Nd4j.create(DataType.DOUBLE, 1,10);
        net.fit(inD, lD);

        INDArray outDouble = net.output(inD);
        net.setInput(inD);
        net.setLabels(lD);
        net.computeGradientAndScore();
        double scoreDouble = net.score();
        INDArray grads = net.getFlattenedGradients();
        INDArray u = net.getUpdater().getStateViewArray();

        MultiLayerNetwork netFloat = net.convertDataType(DataType.FLOAT);
        netFloat.initGradientsView();
        assertEquals(DataType.FLOAT, netFloat.params().dataType());
        assertEquals(DataType.FLOAT, netFloat.getFlattenedGradients().dataType());
        assertEquals(DataType.FLOAT, netFloat.getUpdater(true).getStateViewArray().dataType());
        INDArray inF = inD.castTo(DataType.FLOAT);
        INDArray lF = lD.castTo(DataType.FLOAT);
        INDArray outFloat = netFloat.output(inF);
        netFloat.setInput(inF);
        netFloat.setLabels(lF);
        netFloat.computeGradientAndScore();
        double scoreFloat = netFloat.score();
        INDArray gradsFloat = netFloat.getFlattenedGradients();
        INDArray uFloat = netFloat.getUpdater().getStateViewArray();

        assertEquals(scoreDouble, scoreFloat, 1e-6);
        assertEquals(outDouble.castTo(DataType.FLOAT), outFloat);
        assertEquals(grads.castTo(DataType.FLOAT), gradsFloat);
        INDArray uCast = u.castTo(DataType.FLOAT);
        assertTrue(uCast.equalsWithEps(uFloat, 1e-4));

        MultiLayerNetwork netFP16 = net.convertDataType(DataType.HALF);
        netFP16.initGradientsView();
        assertEquals(DataType.HALF, netFP16.params().dataType());
        assertEquals(DataType.HALF, netFP16.getFlattenedGradients().dataType());
        assertEquals(DataType.HALF, netFP16.getUpdater(true).getStateViewArray().dataType());

        INDArray inH = inD.castTo(DataType.HALF);
        INDArray lH = lD.castTo(DataType.HALF);
        INDArray outHalf = netFP16.output(inH);
        netFP16.setInput(inH);
        netFP16.setLabels(lH);
        netFP16.computeGradientAndScore();
        double scoreHalf = netFP16.score();
        INDArray gradsHalf = netFP16.getFlattenedGradients();
        INDArray uHalf = netFP16.getUpdater().getStateViewArray();

        assertEquals(scoreDouble, scoreHalf, 1e-4);
        boolean outHalfEq = outDouble.castTo(DataType.HALF).equalsWithEps(outHalf, 1e-3);
        assertTrue(outHalfEq);
        boolean gradsHalfEq = grads.castTo(DataType.HALF).equalsWithEps(gradsHalf, 1e-3);
        assertTrue(gradsHalfEq);
        INDArray uHalfCast = u.castTo(DataType.HALF);
        assertTrue(uHalfCast.equalsWithEps(uHalf, 1e-4));
    }


    @Test
    public void testDtypesModelVsGlobalDtype(){
        for(DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}){
            for(DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}){
                for( int outputLayer=0; outputLayer<3; outputLayer++ ) {

                    String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", outputLayer=" + outputLayer;

                    Nd4j.setDefaultDataTypes(networkDtype, networkDtype);

                    Layer ol;
                    switch (outputLayer){
                        case 0:
                            ol = new OutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            break;
                        case 1:
                            ol = new LossLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            break;
                        case 2:
                            ol = new CenterLossOutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            break;
                        default:
                            throw new RuntimeException();
                    }

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .convolutionMode(ConvolutionMode.Same)
                            .updater(new Adam(1e-2))
                            .list()
                            .layer(new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(3).activation(Activation.TANH).build())
                            .layer(new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.AVG).kernelSize(3, 3).stride(2, 2).build())
                            .layer(new ConvolutionLayer.Builder().kernelSize(2, 2).stride(2, 2).nOut(3).activation(Activation.TANH).build())
                            .layer(new BatchNormalization.Builder().build())
                            .layer(new DenseLayer.Builder().nOut(10).activation(Activation.SIGMOID).build())
                            .layer(ol)
                            .setInputType(InputType.convolutional(28, 28, 1))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    Nd4j.setDefaultDataTypes(globalDtype, globalDtype);

                    net.initGradientsView();
                    assertEquals(msg, networkDtype, net.params().dataType());
                    assertEquals(msg, networkDtype, net.getFlattenedGradients().dataType());
                    assertEquals(msg, networkDtype, net.getUpdater(true).getStateViewArray().dataType());

                    INDArray in = Nd4j.rand(networkDtype, 2, 1, 28, 28);
                    INDArray label = TestUtils.randomOneHot(2, 10).castTo(networkDtype);

                    INDArray out = net.output(in);
                    assertEquals(msg, networkDtype, out.dataType());
                    List<INDArray> ff = net.feedForward(in);
                    for (int i = 0; i < ff.size(); i++) {
                        assertEquals(msg, networkDtype, ff.get(i).dataType());
                    }

                    net.setInput(in);
                    net.setLabels(label);
                    net.computeGradientAndScore();

                    net.fit(new DataSet(in, label));
                }
            }
        }
    }

    @Test
    public void testDtypeLeverage(){

        for(DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            for (DataType arrayDType : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                Nd4j.setDefaultDataTypes(globalDtype, globalDtype);

                WorkspaceConfiguration configOuter = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.NONE).build();
                WorkspaceConfiguration configInner = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.NONE).build();

                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configOuter, "ws")) {
                    INDArray arr = Nd4j.create(arrayDType, 3, 4);
                    try (MemoryWorkspace wsInner = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configOuter, "wsInner")) {
                        INDArray leveraged = arr.leverageTo("ws");
                        assertTrue(leveraged.isAttached());
                        assertEquals(arrayDType, leveraged.dataType());

                        INDArray detached = leveraged.detach();
                        assertFalse(detached.isAttached());
                        assertEquals(arrayDType, detached.dataType());
                    }
                }
            }
        }
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

}
