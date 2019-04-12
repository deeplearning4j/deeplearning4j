package org.deeplearning4j.nn.dtypes;

import com.google.common.collect.ImmutableSet;
import com.google.common.reflect.ClassPath;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.AlphaDropout;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.dropout.GaussianNoise;
import org.deeplearning4j.nn.conf.dropout.SpatialDropout;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.layers.util.MaskLayer;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.layers.util.IdentityLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.AfterClass;
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

import java.io.IOException;
import java.lang.reflect.Modifier;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.*;

@Slf4j
public class DTypeTests extends BaseDL4JTest {

    protected static Set<Class<?>> seenLayers = new HashSet<>();
    protected static Set<Class<?>> seenPreprocs = new HashSet<>();
    protected static Set<Class<?>> seenVertices = new HashSet<>();

    @AfterClass
    public static void after(){
        ImmutableSet<ClassPath.ClassInfo> info;
        try {
            //Dependency note: this ClassPath class was added in Guava 14
            info = com.google.common.reflect.ClassPath.from(DTypeTests.class.getClassLoader())
                    .getTopLevelClassesRecursive("org.deeplearning4j");
        } catch (IOException e){
            //Should never happen
            throw new RuntimeException(e);
        }

        System.out.println("CLASS INFO SIZE: " + info.size());

        Set<Class<?>> layerClasses = new HashSet<>();
        Set<Class<?>> preprocClasses = new HashSet<>();
        Set<Class<?>> vertexClasses = new HashSet<>();
        for(ClassPath.ClassInfo ci : info){
            Class<?> clazz;
            try{
                clazz = Class.forName(ci.getName());
            } catch (ClassNotFoundException e){
                //Should never happen as  this was found on the classpath
                throw new RuntimeException(e);
            }

            if(Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface()){
                continue;
            }

            if(Layer.class.isAssignableFrom(clazz)){
                if(!clazz.getName().endsWith("CustomLayer") && !clazz.getName().contains("samediff.testlayers"))
                    layerClasses.add(clazz);
            } else if(InputPreProcessor.class.isAssignableFrom(clazz)){
                preprocClasses.add(clazz);
            } else if(GraphVertex.class.isAssignableFrom(clazz)){
                vertexClasses.add(clazz);
            }
        }

        boolean fail = false;
        if(seenLayers.size() < layerClasses.size()){
            for(Class<?> c : layerClasses){
                if(!seenLayers.contains(c)){
                    log.warn("Layer class not tested for global vs. network datatypes: {}", c);
                }
            }
            fail = true;
        }

        if(fail) {
            fail("Tested " + seenLayers.size() + " of " + layerClasses.size() + " layers, " + seenPreprocs + " of " + preprocClasses.size() +
                    " preprocessors, " + seenVertices + " of " + vertexClasses.size() + " vertices");
        }
    }

    public static void logUsedClasses(MultiLayerNetwork net){
        MultiLayerConfiguration conf = net.getLayerWiseConfigurations();
        for(NeuralNetConfiguration nnc : conf.getConfs()){
            Layer l = nnc.getLayer();
            seenLayers.add(l.getClass());
            if(l instanceof BaseWrapperLayer){
                BaseWrapperLayer bwl = (BaseWrapperLayer) l;
                seenLayers.add(bwl.getUnderlying().getClass());
            } else if(l instanceof Bidirectional){
                seenLayers.add(((Bidirectional) l).getFwd().getClass());
            }
        }

        Map<Integer,InputPreProcessor> preprocs = conf.getInputPreProcessors();
        if(preprocs != null){
            for(InputPreProcessor ipp : preprocs.values()){
                seenPreprocs.add(ipp.getClass());
            }
        }
    }

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
    public void testDtypesModelVsGlobalDtypeCnn(){
        for(DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}){
            for(DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}){
                for( int outputLayer=0; outputLayer<5; outputLayer++ ) {

                    String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", outputLayer=" + outputLayer;

                    Nd4j.setDefaultDataTypes(networkDtype, networkDtype);

                    Layer ol;
                    Layer secondLast;
                    switch (outputLayer){
                        case 0:
                            ol = new OutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new GlobalPoolingLayer(PoolingType.MAX);
                            break;
                        case 1:
                            ol = new LossLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new DenseLayer.Builder().nOut(10).activation(Activation.SIGMOID).build();
                            break;
                        case 2:
                            ol = new CenterLossOutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new VariationalAutoencoder.Builder().encoderLayerSizes(10).decoderLayerSizes(10).nOut(10).activation(Activation.SIGMOID).build();
                            break;
                        case 3:
                            ol = new CnnLossLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(3).activation(Activation.TANH).build();
                            break;
                        case 4:
                            ol = new Yolo2OutputLayer.Builder().boundingBoxPriors(Nd4j.create(new double[][]{{1.0, 1.0}, {2.0, 2.0}}).castTo(networkDtype)).build();
                            secondLast = new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(14).activation(Activation.TANH).build();
                            break;
                        default:
                            throw new RuntimeException();
                    }




                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .convolutionMode(ConvolutionMode.Same)
                            .updater(new Adam(1e-2))
                            .list()
                            .layer(new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(3).activation(Activation.TANH).build())
                            .layer(new LocalResponseNormalization())
                            .layer(new DropoutLayer(0.5))
                            .layer(new DropoutLayer(new AlphaDropout(0.5)))
                            .layer(new DropoutLayer(new GaussianDropout(0.5)))
                            .layer(new DropoutLayer(new GaussianNoise(0.1)))
                            .layer(new DropoutLayer(new SpatialDropout(0.5)))
                            .layer(new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.AVG).kernelSize(3, 3).stride(2, 2).build())
                            .layer(new Pooling2D.Builder().poolingType(SubsamplingLayer.PoolingType.AVG).kernelSize(2, 2).stride(1, 1).build())
                            .layer(new Deconvolution2D.Builder().kernelSize(2, 2).stride(2, 2).nOut(3).activation(Activation.TANH).build())
//                            .layer(new LocallyConnected2D.Builder().nOut(3).kernelSize(2,2).stride(1,1).activation(Activation.SIGMOID).build())   //EXCEPTION
                            .layer(new ZeroPaddingLayer(1,1))
                            .layer(new Cropping2D(1,1))
                            .layer(new IdentityLayer())
                            .layer(new DepthwiseConvolution2D.Builder().nOut(3).activation(Activation.RELU).build())
                            .layer(new SeparableConvolution2D.Builder().nOut(3).activation(Activation.HARDTANH).build())
                            .layer(new MaskLayer())
                            .layer(new BatchNormalization.Builder().build())
                            .layer(new ActivationLayer(Activation.LEAKYRELU))
                            .layer(secondLast)
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
                    INDArray label;
                    if(outputLayer < 3){
                        label = TestUtils.randomOneHot(2, 10).castTo(networkDtype);
                    } else if(outputLayer == 3){
                        //CNN loss
                        label = Nd4j.rand(networkDtype, 2, 3, 28, 28);
                    } else if(outputLayer == 4){
                        //YOLO
                        label = Nd4j.ones(networkDtype, 2, 6, 28, 28);
                    } else {
                        throw new IllegalStateException();
                    }

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

                    logUsedClasses(net);
                }
            }
        }
    }

    @Test
    public void testDtypesModelVsGlobalDtypeRnn(){
        for(DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}){
            for(DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}){
                for( int outputLayer=0; outputLayer<2; outputLayer++ ) {

                    String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", outputLayer=" + outputLayer;

                    Nd4j.setDefaultDataTypes(networkDtype, networkDtype);

                    Layer ol;
                    switch (outputLayer){
                        case 0:
                            ol = new RnnOutputLayer.Builder().nOut(5).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            break;
                        case 1:
                            ol = new RnnLossLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            break;
                        default:
                            throw new RuntimeException();
                    }

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .convolutionMode(ConvolutionMode.Same)
                            .updater(new Adam(1e-2))
                            .list()
                            .layer(new LSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build())
                            .layer(new GravesLSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build())
                            .layer(new GravesBidirectionalLSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build())
                            .layer(new Bidirectional(new LSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build()))
                            .layer(new SimpleRnn.Builder().nIn(10).nOut(5).build())
                            .layer(ol)
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    Nd4j.setDefaultDataTypes(globalDtype, globalDtype);

                    net.initGradientsView();
                    assertEquals(msg, networkDtype, net.params().dataType());
                    assertEquals(msg, networkDtype, net.getFlattenedGradients().dataType());
                    assertEquals(msg, networkDtype, net.getUpdater(true).getStateViewArray().dataType());

                    INDArray in = Nd4j.rand(networkDtype, 2, 5, 4);
                    INDArray label = TestUtils.randomOneHotTimeSeries(2, 5, 4).castTo(networkDtype);

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

                    logUsedClasses(net);
                }
            }
        }
    }
}
