/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.dtypes;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.common.config.DL4JClassLoading;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.dropout.AlphaDropout;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.dropout.GaussianNoise;
import org.deeplearning4j.nn.conf.dropout.SpatialDropout;
import org.deeplearning4j.nn.conf.graph.AttentionVertex;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.FrozenVertex;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.graph.L2Vertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.PoolHelperVertex;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.graph.ShiftVertex;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.graph.UnstackVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.graph.rnn.ReverseTimeSeriesVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.CapsuleLayer;
import org.deeplearning4j.nn.conf.layers.CapsuleStrengthLayer;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.Cnn3DLossLayer;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Convolution1D;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Deconvolution2D;
import org.deeplearning4j.nn.conf.layers.Deconvolution3D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DepthwiseConvolution2D;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.LearnedSelfAttentionLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.LocallyConnected1D;
import org.deeplearning4j.nn.conf.layers.LocallyConnected2D;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PReLULayer;
import org.deeplearning4j.nn.conf.layers.Pooling1D;
import org.deeplearning4j.nn.conf.layers.Pooling2D;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.PrimaryCapsules;
import org.deeplearning4j.nn.conf.layers.RecurrentAttentionLayer;
import org.deeplearning4j.nn.conf.layers.RnnLossLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SelfAttentionLayer;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;
import org.deeplearning4j.nn.conf.layers.SpaceToBatchLayer;
import org.deeplearning4j.nn.conf.layers.SpaceToDepthLayer;
import org.deeplearning4j.nn.conf.layers.Subsampling1DLayer;
import org.deeplearning4j.nn.conf.layers.Subsampling3DLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling1D;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.layers.Upsampling3D;
import org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPadding3DLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping3D;
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop;
import org.deeplearning4j.nn.conf.layers.misc.RepeatVector;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.layers.recurrent.TimeDistributed;
import org.deeplearning4j.nn.conf.layers.util.MaskLayer;
import org.deeplearning4j.nn.conf.layers.util.MaskZeroLayer;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.ComposableInputPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnn3DPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.util.IdentityLayer;
import org.deeplearning4j.nn.modelimport.keras.layers.TFOpLayer;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.KerasFlattenRnnPreprocessor;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.PermutePreprocessor;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.TensorFlowCnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.junit.AfterClass;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.nd4j.linalg.profiler.ProfilerConfig;
import org.nd4j.shade.guava.collect.ImmutableSet;
import org.nd4j.shade.guava.reflect.ClassPath;

import java.io.IOException;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
@Ignore
public class DTypeTests extends BaseDL4JTest {

    protected static Set<Class<?>> seenLayers = new HashSet<>();
    protected static Set<Class<?>> seenPreprocs = new HashSet<>();
    protected static Set<Class<?>> seenVertices = new HashSet<>();

    protected static Set<Class<?>> ignoreClasses = new HashSet<>(Arrays.<Class<?>>asList(
            Pooling2D.class,        //Alias for SubsamplingLayer
            Convolution2D.class,    //Alias for ConvolutionLayer
            Pooling1D.class,        //Alias for Subsampling1D
            Convolution1D.class,    //Alias for  Convolution1DLayer
            TensorFlowCnnToFeedForwardPreProcessor.class    //Deprecated
    ));

    @Override
    public long getTimeoutMilliseconds() {
        return 9999999L;
    }

    @AfterClass
    public static void after() {
        ImmutableSet<ClassPath.ClassInfo> info;
        try {
            //Dependency note: this ClassPath class was added in Guava 14
            info = org.nd4j.shade.guava.reflect.ClassPath.from(DTypeTests.class.getClassLoader())
                    .getTopLevelClassesRecursive("org.deeplearning4j");
        } catch (IOException e) {
            //Should never happen
            throw new RuntimeException(e);
        }

        Set<Class<?>> layerClasses = new HashSet<>();
        Set<Class<?>> preprocClasses = new HashSet<>();
        Set<Class<?>> vertexClasses = new HashSet<>();
        for (ClassPath.ClassInfo ci : info) {
            Class<?> clazz = DL4JClassLoading.loadClassByName(ci.getName());

            if (Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface() || TFOpLayer.class == clazz) {
                // Skip TFOpLayer here - dtype depends on imported model dtype
                continue;
            }

            if (clazz.getName().toLowerCase().contains("custom")
                    || clazz.getName().contains("samediff.testlayers")
                    || clazz.getName().toLowerCase().contains("test")
                    || ignoreClasses.contains(clazz)) {
                continue;
            }

            if (Layer.class.isAssignableFrom(clazz)) {
                layerClasses.add(clazz);
            } else if (InputPreProcessor.class.isAssignableFrom(clazz)) {
                preprocClasses.add(clazz);
            } else if (GraphVertex.class.isAssignableFrom(clazz)) {
                vertexClasses.add(clazz);
            }
        }

        boolean fail = false;
        if (seenLayers.size() < layerClasses.size()) {
            for (Class<?> c : layerClasses) {
                if (!seenLayers.contains(c) && !ignoreClasses.contains(c)) {
                    log.warn("Layer class not tested for global vs. network datatypes: {}", c);
                    fail = true;
                }
            }
        }
        if (seenPreprocs.size() < preprocClasses.size()) {
            for (Class<?> c : preprocClasses) {
                if (!seenPreprocs.contains(c) && !ignoreClasses.contains(c)) {
                    log.warn("Preprocessor class not tested for global vs. network datatypes: {}", c);
                    fail = true;
                }
            }
        }
        if (seenVertices.size() < vertexClasses.size()) {
            for (Class<?> c : vertexClasses) {
                if (!seenVertices.contains(c) && !ignoreClasses.contains(c)) {
                    log.warn("GraphVertex class not tested for global vs. network datatypes: {}", c);
                    fail = true;
                }
            }
        }

       /* if (fail) {
            fail("Tested " + seenLayers.size() + " of " + layerClasses.size() + " layers, " + seenPreprocs.size() + " of " + preprocClasses.size() +
                    " preprocessors, " + seenVertices.size() + " of " + vertexClasses.size() + " vertices");
        }*/
    }

    public static void logUsedClasses(MultiLayerNetwork net) {
        MultiLayerConfiguration conf = net.getLayerWiseConfigurations();
        for (NeuralNetConfiguration nnc : conf.getConfs()) {
            Layer l = nnc.getLayer();
            seenLayers.add(l.getClass());
            if (l instanceof BaseWrapperLayer) {
                BaseWrapperLayer bwl = (BaseWrapperLayer) l;
                seenLayers.add(bwl.getUnderlying().getClass());
            } else if (l instanceof Bidirectional) {
                seenLayers.add(((Bidirectional) l).getFwd().getClass());
            }
        }

        Map<Integer, InputPreProcessor> preprocs = conf.getInputPreProcessors();
        if (preprocs != null) {
            for (InputPreProcessor ipp : preprocs.values()) {
                seenPreprocs.add(ipp.getClass());
            }
        }
    }

    public static void logUsedClasses(ComputationGraph net) {
        ComputationGraphConfiguration conf = net.getConfiguration();
        for (GraphVertex gv : conf.getVertices().values()) {
            seenVertices.add(gv.getClass());
            if (gv instanceof LayerVertex) {
                seenLayers.add(((LayerVertex) gv).getLayerConf().getLayer().getClass());
                InputPreProcessor ipp = ((LayerVertex) gv).getPreProcessor();
                if (ipp != null) {
                    seenPreprocs.add(ipp.getClass());
                }
            } else if (gv instanceof PreprocessorVertex) {
                seenPreprocs.add(((PreprocessorVertex) gv).getPreProcessor().getClass());
            }
        }

    }

    @Test
    public void testMultiLayerNetworkTypeConversion() {

        for (DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(dt, dt);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(0.01))
                    .dataType(DataType.DOUBLE)
                    .list()
                    .layer(new DenseLayer.Builder().activation(Activation.TANH).nIn(10).nOut(10).build())
                    .layer(new DenseLayer.Builder().activation(Activation.TANH).nIn(10).nOut(10).build())
                    .layer(new OutputLayer.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            INDArray inD = Nd4j.rand(DataType.DOUBLE, 1, 10);
            INDArray lD = Nd4j.create(DataType.DOUBLE, 1, 10);
            net.fit(inD, lD);

            INDArray outDouble = net.output(inD);
            net.setInput(inD);
            net.setLabels(lD);
            net.computeGradientAndScore();
            double scoreDouble = net.score();
            INDArray grads = net.getFlattenedGradients();
            INDArray u = net.getUpdater().getStateViewArray();
            assertEquals(DataType.DOUBLE, net.params().dataType());
            assertEquals(DataType.DOUBLE, grads.dataType());
            assertEquals(DataType.DOUBLE, u.dataType());


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
    }

    @Test
    public void testComputationGraphTypeConversion() {

        for (DataType dt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(dt, dt);

            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(0.01))
                    .dataType(DataType.DOUBLE)
                    .graphBuilder()
                    .addInputs("in")
                    .layer("l0", new DenseLayer.Builder().activation(Activation.TANH).nIn(10).nOut(10).build(), "in")
                    .layer("l1", new DenseLayer.Builder().activation(Activation.TANH).nIn(10).nOut(10).build(), "l0")
                    .layer("out", new OutputLayer.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build(), "l1")
                    .setOutputs("out")
                    .build();

            ComputationGraph net = new ComputationGraph(conf);
            net.init();

            INDArray inD = Nd4j.rand(DataType.DOUBLE, 1, 10);
            INDArray lD = Nd4j.create(DataType.DOUBLE, 1, 10);
            net.fit(new DataSet(inD, lD));

            INDArray outDouble = net.outputSingle(inD);
            net.setInput(0, inD);
            net.setLabels(lD);
            net.computeGradientAndScore();
            double scoreDouble = net.score();
            INDArray grads = net.getFlattenedGradients();
            INDArray u = net.getUpdater().getStateViewArray();
            assertEquals(DataType.DOUBLE, net.params().dataType());
            assertEquals(DataType.DOUBLE, grads.dataType());
            assertEquals(DataType.DOUBLE, u.dataType());


            ComputationGraph netFloat = net.convertDataType(DataType.FLOAT);
            netFloat.initGradientsView();
            assertEquals(DataType.FLOAT, netFloat.params().dataType());
            assertEquals(DataType.FLOAT, netFloat.getFlattenedGradients().dataType());
            assertEquals(DataType.FLOAT, netFloat.getUpdater(true).getStateViewArray().dataType());
            INDArray inF = inD.castTo(DataType.FLOAT);
            INDArray lF = lD.castTo(DataType.FLOAT);
            INDArray outFloat = netFloat.outputSingle(inF);
            netFloat.setInput(0, inF);
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

            ComputationGraph netFP16 = net.convertDataType(DataType.HALF);
            netFP16.initGradientsView();
            assertEquals(DataType.HALF, netFP16.params().dataType());
            assertEquals(DataType.HALF, netFP16.getFlattenedGradients().dataType());
            assertEquals(DataType.HALF, netFP16.getUpdater(true).getStateViewArray().dataType());

            INDArray inH = inD.castTo(DataType.HALF);
            INDArray lH = lD.castTo(DataType.HALF);
            INDArray outHalf = netFP16.outputSingle(inH);
            netFP16.setInput(0, inH);
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
    }


    @Test
    public void testDtypesModelVsGlobalDtypeCnn() {
        for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                for (int outputLayer = 0; outputLayer < 5; outputLayer++) {
                    assertEquals(globalDtype, Nd4j.dataType());
                    assertEquals(globalDtype, Nd4j.defaultFloatingPointType());

                    String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", outputLayer=" + outputLayer;

                    Layer ol;
                    Layer secondLast;
                    switch (outputLayer) {
                        case 0:
                            ol = new OutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new GlobalPoolingLayer(PoolingType.MAX);
                            break;
                        case 1:
                            ol = new LossLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new FrozenLayerWithBackprop(new DenseLayer.Builder().nOut(10).activation(Activation.SIGMOID).build());
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
                            .dataType(networkDtype)
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
                            .layer(new ZeroPaddingLayer(1, 1))
                            .layer(new Cropping2D(1, 1))
                            .layer(new IdentityLayer())
                            .layer(new Upsampling2D.Builder().size(2).build())
                            .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).build())
                            .layer(new DepthwiseConvolution2D.Builder().nOut(3).activation(Activation.RELU).build())
                            .layer(new SeparableConvolution2D.Builder().nOut(3).activation(Activation.HARDTANH).build())
                            .layer(new MaskLayer())
                            .layer(new BatchNormalization.Builder().build())
                            .layer(new ActivationLayer(Activation.LEAKYRELU))
                            .layer(secondLast)
                            .layer(ol)
                            .setInputType(InputType.convolutionalFlat(8, 8, 1))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    net.initGradientsView();
                    assertEquals(msg, networkDtype, net.params().dataType());
                    assertEquals(msg, networkDtype, net.getFlattenedGradients().dataType());
                    assertEquals(msg, networkDtype, net.getUpdater(true).getStateViewArray().dataType());

                    INDArray in = Nd4j.rand(networkDtype, 2, 8 * 8);
                    INDArray label;
                    if (outputLayer < 3) {
                        label = TestUtils.randomOneHot(2, 10).castTo(networkDtype);
                    } else if (outputLayer == 3) {
                        //CNN loss
                        label = Nd4j.rand(networkDtype, 2, 3, 8, 8);
                    } else if (outputLayer == 4) {
                        //YOLO
                        label = Nd4j.ones(networkDtype, 2, 6, 8, 8);
                    } else {
                        throw new IllegalStateException();
                    }

                    INDArray out = net.output(in);
                    assertEquals(msg, networkDtype, out.dataType());
                    List<INDArray> ff = net.feedForward(in);
                    for (int i = 0; i < ff.size(); i++) {
                        String s = msg + " - layer " + (i - 1) + " - " + (i == 0 ? "input" : net.getLayer(i - 1).conf().getLayer().getClass().getSimpleName());
                        assertEquals(s, networkDtype, ff.get(i).dataType());
                    }

                    net.setInput(in);
                    net.setLabels(label);
                    net.computeGradientAndScore();

                    net.fit(new DataSet(in, label));

                    logUsedClasses(net);

                    //Now, test mismatched dtypes for input/labels:
                    for (DataType inputLabelDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                        log.info(msg + " - input/label type: " + inputLabelDtype);
                        INDArray in2 = in.castTo(inputLabelDtype);
                        INDArray label2 = label.castTo(inputLabelDtype);
                        net.output(in2);
                        net.setInput(in2);
                        net.setLabels(label2);
                        net.computeGradientAndScore();

                        net.fit(new DataSet(in2, label2));
                    }
                }
            }
        }
    }

    @Test
    public void testDtypesModelVsGlobalDtypeCnn3d() {
        for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                for (int outputLayer = 0; outputLayer < 3; outputLayer++) {
                    assertEquals(globalDtype, Nd4j.dataType());
                    assertEquals(globalDtype, Nd4j.defaultFloatingPointType());

                    String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", outputLayer=" + outputLayer;
                    log.info(msg);

                    Layer ol;
                    Layer secondLast;
                    switch (outputLayer) {
                        case 0:
                            ol = new OutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new GlobalPoolingLayer(PoolingType.AVG);
                            break;
                        case 1:
                            ol = new Cnn3DLossLayer.Builder(Convolution3D.DataFormat.NCDHW).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new Convolution3D.Builder().nOut(3).activation(Activation.ELU).build();
                            break;
                        case 2:
                            ol = new OutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new Convolution3D.Builder().nOut(3).activation(Activation.ELU).build();
                            break;
                        default:
                            throw new RuntimeException();
                    }


                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .dataType(networkDtype)
                            .convolutionMode(ConvolutionMode.Same)
                            .updater(new Nesterovs(1e-2, 0.9))
                            .list()
                            .layer(new Convolution3D.Builder().kernelSize(2, 2, 2).stride(1, 1, 1).nOut(3).activation(Activation.TANH).build())
                            .layer(new Convolution3D.Builder().kernelSize(2, 2, 2).stride(1, 1, 1).nOut(3).activation(Activation.TANH).build())
                            .layer(new Subsampling3DLayer.Builder().poolingType(PoolingType.AVG).kernelSize(2, 2, 2).stride(2, 2, 2).build())
                            .layer(new Deconvolution3D.Builder().kernelSize(2,2,2).stride(1,1,1).nIn(3).nOut(3).activation(Activation.TANH).build())
                            .layer(new Cropping3D.Builder(1, 1, 1, 1, 1, 1).build())
                            .layer(new ZeroPadding3DLayer.Builder(1, 1, 1, 1, 1, 1).build())
                            .layer(new ActivationLayer(Activation.LEAKYRELU))
                            .layer(new Upsampling3D.Builder().size(2).build())
                            .layer(secondLast)
                            .layer(ol)
                            .setInputType(InputType.convolutional3D(Convolution3D.DataFormat.NCDHW, 8, 8, 8, 1))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    net.initGradientsView();
                    assertEquals(msg, networkDtype, net.params().dataType());
                    assertEquals(msg, networkDtype, net.getFlattenedGradients().dataType());
                    assertEquals(msg, networkDtype, net.getUpdater(true).getStateViewArray().dataType());

                    INDArray in = Nd4j.rand(networkDtype, 2, 1, 8, 8, 8);
                    INDArray label;
                    if (outputLayer == 0) {
                        label = TestUtils.randomOneHot(2, 10).castTo(networkDtype);
                    } else if (outputLayer == 1) {
                        //CNN3D loss
                        label = Nd4j.rand(networkDtype, 2, 3, 8, 8, 8);
                    } else if (outputLayer == 2) {
                        label = TestUtils.randomOneHot(2, 10).castTo(networkDtype);
                    } else {
                        throw new RuntimeException();
                    }

                    INDArray out = net.output(in);
                    assertEquals(msg, networkDtype, out.dataType());
                    List<INDArray> ff = net.feedForward(in);
                    for (int i = 0; i < ff.size(); i++) {
                        String s = msg + " - layer " + (i - 1) + " - " + (i == 0 ? "input" : net.getLayer(i - 1).conf().getLayer().getClass().getSimpleName());
                        assertEquals(s, networkDtype, ff.get(i).dataType());
                    }

                    net.setInput(in);
                    net.setLabels(label);
                    net.computeGradientAndScore();

                    net.fit(new DataSet(in, label));

                    logUsedClasses(net);

                    //Now, test mismatched dtypes for input/labels:
                    for (DataType inputLabelDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                        INDArray in2 = in.castTo(inputLabelDtype);
                        INDArray label2 = label.castTo(inputLabelDtype);
                        net.output(in2);
                        net.setInput(in2);
                        net.setLabels(label2);
                        net.computeGradientAndScore();

                        net.fit(new DataSet(in2, label2));
                    }
                }
            }
        }
    }

    @Test
    @Ignore
    public void testDtypesModelVsGlobalDtypeCnn1d() {
        //Nd4jCpu.Environment.getInstance().setUseMKLDNN(false);
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().setProfilingConfig(ProfilerConfig.builder()
                .checkForNAN(true)
                .checkWorkspaces(true)
                .checkForINF(true)
                .build());
        for (DataType globalDtype : new DataType[]{DataType.DOUBLE}) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[]{DataType.DOUBLE}) {
                for (int outputLayer = 0; outputLayer < 3; outputLayer++) {
                    assertEquals(globalDtype, Nd4j.dataType());
                    assertEquals(globalDtype, Nd4j.defaultFloatingPointType());

                    String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", outputLayer=" + outputLayer + " at index " + outputLayer;

                    Layer ol;
                    Layer secondLast;
                    switch (outputLayer) {
                        case 0:
                            ol = new OutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new GlobalPoolingLayer(PoolingType.MAX);
                            break;
                        case 1:
                            ol = new RnnOutputLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).nOut(5).build();
                            secondLast = new Convolution1D.Builder().kernelSize(2).nOut(5).build();
                            break;
                        case 2:
                            ol = new RnnLossLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new Convolution1D.Builder().kernelSize(2).nOut(5).build();
                            break;
                        default:
                            throw new RuntimeException();
                    }


                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .trainingWorkspaceMode(WorkspaceMode.NONE)
                            .inferenceWorkspaceMode(WorkspaceMode.NONE)
                            .dataType(networkDtype)
                            .convolutionMode(ConvolutionMode.Same)
                            .updater(new Adam(1e-2))
                            .list()
                            .layer(new Convolution1D.Builder()
                                    .kernelSize(2)
                                    .stride(1).nOut(3).
                                            activation(Activation.TANH).build())
                            .layer(new Subsampling1DLayer.Builder().poolingType(PoolingType.MAX).kernelSize(5).stride(1).build())
                            .layer(new Cropping1D.Builder(1).build())
                            .layer(new ZeroPadding1DLayer(1))
                            .layer(new Upsampling1D.Builder(2).build())
                            .layer(secondLast)
                            .layer(ol)
                            .setInputType(InputType.recurrent(5, 10,RNNFormat.NCW))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    net.initGradientsView();
                    assertEquals(msg, networkDtype, net.params().dataType());
                    assertEquals(msg, networkDtype, net.getFlattenedGradients().dataType());
                    assertEquals(msg, networkDtype, net.getUpdater(true).getStateViewArray().dataType());

                    INDArray in = Nd4j.rand(networkDtype, 2, 5, 10);
                    INDArray label;
                    if (outputLayer == 0) {
                        //OutputLayer
                        label = TestUtils.randomOneHot(2, 10).castTo(networkDtype);
                    } else {
                        //RnnOutputLayer, RnnLossLayer
                        label = Nd4j.rand(networkDtype, 2, 5, 20);   //Longer sequence due to upsampling
                    }

                    INDArray out = net.output(in);
                    assertEquals(msg, networkDtype, out.dataType());
                    List<INDArray> ff = net.feedForward(in);
                    for (int i = 0; i < ff.size(); i++) {
                        String s = msg + " - layer " + (i - 1) + " - " + (i == 0 ? "input" : net.getLayer(i - 1).conf().getLayer().getClass().getSimpleName());
                        assertEquals(s, networkDtype, ff.get(i).dataType());
                    }

                    net.setInput(in);
                    net.setLabels(label);
                    net.computeGradientAndScore();

                    //net.fit(new DataSet(in, label));

                    logUsedClasses(net);

                    //Now, test mismatched dtypes for input/labels:
                    for (DataType inputLabelDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT}) {
                        System.out.println(msg + " - " + inputLabelDtype);
                        INDArray in2 = in.castTo(inputLabelDtype);
                        INDArray label2 = label.castTo(inputLabelDtype);
                        net.output(in2);
                        net.setInput(in2);
                        net.setLabels(label2);
                        net.computeGradientAndScore();

                        //net.fit(new DataSet(in2, label2));
                    }
                }
            }
        }
    }

    @Test
    public void testDtypesModelVsGlobalDtypeMisc() {
        for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                assertEquals(globalDtype, Nd4j.dataType());
                assertEquals(globalDtype, Nd4j.defaultFloatingPointType());

                String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype;


                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .dataType(networkDtype)
                        .convolutionMode(ConvolutionMode.Same)
                        .updater(new Adam(1e-2))
                        .list()
                        .layer(new SpaceToBatchLayer.Builder().blocks(1, 1).build())
                        .layer(new SpaceToDepthLayer.Builder().blocks(2).build())
                        .layer(new OutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .setInputType(InputType.convolutional(28, 28, 5))
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                net.initGradientsView();
                assertEquals(msg, networkDtype, net.params().dataType());
                assertEquals(msg, networkDtype, net.getFlattenedGradients().dataType());
                assertEquals(msg, networkDtype, net.getUpdater(true).getStateViewArray().dataType());

                INDArray in = Nd4j.rand(networkDtype, 2, 5, 28, 28);
                INDArray label = TestUtils.randomOneHot(2, 10).castTo(networkDtype);

                INDArray out = net.output(in);
                assertEquals(msg, networkDtype, out.dataType());
                List<INDArray> ff = net.feedForward(in);
                for (int i = 0; i < ff.size(); i++) {
                    String s = msg + " - layer " + (i - 1) + " - " + (i == 0 ? "input" : net.getLayer(i - 1).conf().getLayer().getClass().getSimpleName());
                    assertEquals(s, networkDtype, ff.get(i).dataType());
                }

                net.setInput(in);
                net.setLabels(label);
                net.computeGradientAndScore();

                net.fit(new DataSet(in, label));

                logUsedClasses(net);

                //Now, test mismatched dtypes for input/labels:
                for (DataType inputLabelDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                    INDArray in2 = in.castTo(inputLabelDtype);
                    INDArray label2 = label.castTo(inputLabelDtype);
                    net.output(in2);
                    net.setInput(in2);
                    net.setLabels(label2);
                    net.computeGradientAndScore();

                    net.fit(new DataSet(in2, label2));
                }
            }
        }
    }

    @Test
    public void testDtypesModelVsGlobalDtypeRnn() {
        for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                for (int outputLayer = 0; outputLayer < 3; outputLayer++) {
                    assertEquals(globalDtype, Nd4j.dataType());
                    assertEquals(globalDtype, Nd4j.defaultFloatingPointType());

                    String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", outputLayer=" + outputLayer;

                    Layer ol;
                    Layer secondLast;
                    switch (outputLayer) {
                        case 0:
                            ol = new RnnOutputLayer.Builder().nOut(5).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new SimpleRnn.Builder().nOut(5).activation(Activation.TANH).build();
                            break;
                        case 1:
                            ol = new RnnLossLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build();
                            secondLast = new SimpleRnn.Builder().nOut(5).activation(Activation.TANH).build();
                            break;
                        case 2:
                            ol = new OutputLayer.Builder().nOut(5).build();
                            secondLast = new LastTimeStep(new SimpleRnn.Builder().nOut(5).activation(Activation.TANH).build());
                            break;
                        default:
                            throw new RuntimeException();
                    }

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .dataType(networkDtype)
                            .convolutionMode(ConvolutionMode.Same)
                            .updater(new Adam(1e-2))
                            .list()
                            .layer(new LSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build())
                            .layer(new GravesLSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build())
                            .layer(new DenseLayer.Builder().nOut(5).build())
                            .layer(new GravesBidirectionalLSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build())
                            .layer(new Bidirectional(new LSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build()))
                            .layer(new TimeDistributed(new DenseLayer.Builder().nIn(10).nOut(5).activation(Activation.TANH).build()))
                            .layer(new SimpleRnn.Builder().nIn(5).nOut(5).build())
                            .layer(new MaskZeroLayer.Builder().underlying(new SimpleRnn.Builder().nIn(5).nOut(5).build()).maskValue(0.0).build())
                            .layer(secondLast)
                            .layer(ol)
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    net.initGradientsView();
                    assertEquals(msg, networkDtype, net.params().dataType());
                    assertEquals(msg, networkDtype, net.getFlattenedGradients().dataType());
                    assertEquals(msg, networkDtype, net.getUpdater(true).getStateViewArray().dataType());

                    INDArray in = Nd4j.rand(networkDtype, 2, 5, 2);
                    INDArray label;
                    if (outputLayer == 2) {
                        label = TestUtils.randomOneHot(2, 5).castTo(networkDtype);
                    } else {
                        label = TestUtils.randomOneHotTimeSeries(2, 5, 2).castTo(networkDtype);
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

                    net.fit(new DataSet(in, label, Nd4j.ones(networkDtype, 2, 2), outputLayer == 2 ? null : Nd4j.ones(networkDtype, 2, 2)));

                    logUsedClasses(net);

                    //Now, test mismatched dtypes for input/labels:
                    for (DataType inputLabelDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                        INDArray in2 = in.castTo(inputLabelDtype);
                        INDArray label2 = label.castTo(inputLabelDtype);
                        net.output(in2);
                        net.setInput(in2);
                        net.setLabels(label2);
                        net.computeGradientAndScore();

                        net.fit(new DataSet(in2, label2));
                    }
                }
            }
        }
    }

    @Test
    public void testCapsNetDtypes() {
        for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                assertEquals(globalDtype, Nd4j.dataType());
                assertEquals(globalDtype, Nd4j.defaultFloatingPointType());

                String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype;

                int primaryCapsDim = 2;
                int primarpCapsChannel = 8;
                int capsule = 5;
                int minibatchSize = 8;
                int routing = 1;
                int capsuleDim = 4;
                int height = 6;
                int width = 6;
                int inputDepth = 4;

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .dataType(networkDtype)
                        .seed(123)
                        .updater(new NoOp())
                        .weightInit(new WeightInitDistribution(new UniformDistribution(-6, 6)))
                        .list()
                        .layer(new PrimaryCapsules.Builder(primaryCapsDim, primarpCapsChannel)
                                .kernelSize(3, 3)
                                .stride(2, 2)
                                .build())
                        .layer(new CapsuleLayer.Builder(capsule, capsuleDim, routing).build())
                        .layer(new CapsuleStrengthLayer.Builder().build())
                        .layer(new ActivationLayer.Builder(new ActivationSoftmax()).build())
                        .layer(new LossLayer.Builder(new LossNegativeLogLikelihood()).build())
                        .setInputType(InputType.convolutional(height, width, inputDepth))
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                INDArray in = Nd4j.rand(networkDtype, minibatchSize, inputDepth * height * width).mul(10)
                        .reshape(-1, inputDepth, height, width);
                INDArray label = Nd4j.zeros(networkDtype, minibatchSize, capsule);
                for (int i = 0; i < minibatchSize; i++) {
                    label.putScalar(new int[]{i, i % capsule}, 1.0);
                }

                INDArray out = net.output(in);
                assertEquals(msg, networkDtype, out.dataType());
                List<INDArray> ff = net.feedForward(in);
                for (int i = 0; i < ff.size(); i++) {
                    String s = msg + " - layer " + (i - 1) + " - " + (i == 0 ? "input" : net.getLayer(i - 1).conf().getLayer().getClass().getSimpleName());
                    assertEquals(s, networkDtype, ff.get(i).dataType());
                }

                net.setInput(in);
                net.setLabels(label);
                net.computeGradientAndScore();

                net.fit(new DataSet(in, label));

                logUsedClasses(net);

                //Now, test mismatched dtypes for input/labels:
                for (DataType inputLabelDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                    INDArray in2 = in.castTo(inputLabelDtype);
                    INDArray label2 = label.castTo(inputLabelDtype);
                    net.output(in2);
                    net.setInput(in2);
                    net.setLabels(label2);
                    net.computeGradientAndScore();

                    net.fit(new DataSet(in2, label2));
                }
            }
        }
    }

    @Test
    public void testEmbeddingDtypes() {
        for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                for (boolean frozen : new boolean[]{false, true}) {
                    for (int test = 0; test < 3; test++) {
                        assertEquals(globalDtype, Nd4j.dataType());
                        assertEquals(globalDtype, Nd4j.defaultFloatingPointType());

                        String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", test=" + test;

                        ComputationGraphConfiguration.GraphBuilder conf = new NeuralNetConfiguration.Builder()
                                .dataType(networkDtype)
                                .seed(123)
                                .updater(new NoOp())
                                .weightInit(new WeightInitDistribution(new UniformDistribution(-6, 6)))
                                .graphBuilder()
                                .addInputs("in")
                                .setOutputs("out");

                        INDArray input;
                        if (test == 0) {
                            if (frozen) {
                                conf.layer("0", new FrozenLayer(new EmbeddingLayer.Builder().nIn(5).nOut(5).build()), "in");
                            } else {
                                conf.layer("0", new EmbeddingLayer.Builder().nIn(5).nOut(5).build(), "in");
                            }

                            input = Nd4j.zeros(networkDtype, 10, 1).muli(5).castTo(DataType.INT);
                            conf.setInputTypes(InputType.feedForward(1));
                        } else if (test == 1) {
                            if (frozen) {
                                conf.layer("0", new FrozenLayer(new EmbeddingSequenceLayer.Builder().nIn(5).nOut(5).build()), "in");
                            } else {
                                conf.layer("0", new EmbeddingSequenceLayer.Builder().nIn(5).nOut(5).build(), "in");
                            }
                            conf.layer("gp", new GlobalPoolingLayer.Builder(PoolingType.PNORM).pnorm(2).poolingDimensions(2).build(), "0");
                            input = Nd4j.zeros(networkDtype, 10, 1, 5).muli(5).castTo(DataType.INT);
                            conf.setInputTypes(InputType.recurrent(1));
                        } else {
                            conf.layer("0", new RepeatVector.Builder().repetitionFactor(5).nOut(5).build(), "in");
                            conf.layer("gp", new GlobalPoolingLayer.Builder(PoolingType.SUM).build(), "0");
                            input = Nd4j.zeros(networkDtype, 10, 5);
                            conf.setInputTypes(InputType.feedForward(5));
                        }

                        conf.appendLayer("el", new ElementWiseMultiplicationLayer.Builder().nOut(5).build())
                                .appendLayer("ae", new AutoEncoder.Builder().nOut(5).build())
                                .appendLayer("prelu", new PReLULayer.Builder().nOut(5).inputShape(5).build())
                                .appendLayer("out", new OutputLayer.Builder().nOut(10).build());

                        ComputationGraph net = new ComputationGraph(conf.build());
                        net.init();

                        INDArray label = Nd4j.zeros(networkDtype, 10, 10);

                        INDArray out = net.outputSingle(input);
                        assertEquals(msg, networkDtype, out.dataType());
                        Map<String, INDArray> ff = net.feedForward(input, false);
                        for (Map.Entry<String, INDArray> e : ff.entrySet()) {
                            if (e.getKey().equals("in"))
                                continue;
                            String s = msg + " - layer: " + e.getKey();
                            assertEquals(s, networkDtype, e.getValue().dataType());
                        }

                        net.setInput(0, input);
                        net.setLabels(label);
                        net.computeGradientAndScore();

                        net.fit(new DataSet(input, label));

                        logUsedClasses(net);

                        //Now, test mismatched dtypes for input/labels:
                        for (DataType inputLabelDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                            INDArray in2 = input.castTo(inputLabelDtype);
                            INDArray label2 = label.castTo(inputLabelDtype);
                            net.output(in2);
                            net.setInput(0, in2);
                            net.setLabels(label2);
                            net.computeGradientAndScore();

                            net.fit(new DataSet(in2, label2));
                        }
                    }
                }
            }
        }
    }

    @Test
    public void testVertexDtypes() {
        for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                assertEquals(globalDtype, Nd4j.dataType());
                assertEquals(globalDtype, Nd4j.defaultFloatingPointType());

                INDArray[] in = null;
                for (int test = 0; test < 8; test++) {
                    String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", test=" + test;

                    ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
                            .dataType(networkDtype)
                            .seed(123)
                            .updater(new NoOp())
                            .weightInit(WeightInit.XAVIER)
                            .convolutionMode(ConvolutionMode.Same)
                            .graphBuilder();

                    switch (test) {
                        case 0:
                            b.addInputs("in")
                                    .addLayer("l", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(1).build(), "in")
                                    .addVertex("preproc", new PreprocessorVertex(new CnnToRnnPreProcessor(28, 28, 1)), "l")
                                    .addLayer("out", new OutputLayer.Builder().nOut(10).build(), "preproc")
                                    .setInputTypes(InputType.convolutional(28, 28, 1))
                                    .setOutputs("out");
                            in = new INDArray[]{Nd4j.rand(networkDtype, 2, 1, 28, 28)};
                            break;
                        case 1:
                            b.addInputs("in")
                                    .addLayer("l", new DenseLayer.Builder().nOut(16).build(), "in")
                                    .addVertex("preproc", new PreprocessorVertex(new FeedForwardToCnn3DPreProcessor(2, 2, 2, 2, true)), "l")
                                    .addVertex("preproc2", new PreprocessorVertex(new PermutePreprocessor(0, 2, 3, 4, 1)), "preproc")
                                    .addVertex("preproc3", new PreprocessorVertex(new ReshapePreprocessor(new long[]{2, 2, 2, 2}, new long[]{16}, false)), "preproc2")
                                    .addLayer("out", new OutputLayer.Builder().nIn(16).nOut(10).build(), "preproc3")
                                    .setInputTypes(InputType.feedForward(5))
                                    .setOutputs("out");
                            in = new INDArray[]{Nd4j.rand(networkDtype, 2, 5)};
                            break;
                        case 2:
                            b.addInputs("in")
                                    .addLayer("1", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(1).build(), "in")
                                    .addVertex("1a", new PoolHelperVertex(), "1")
                                    .addVertex("2", new ShiftVertex(1), "1a")
                                    .addVertex("3", new ScaleVertex(2), "2")
                                    .addVertex("4", new ReshapeVertex(2, -1), "3")
                                    .addVertex("5", new SubsetVertex(0, 99), "4")
                                    .addVertex("6", new L2NormalizeVertex(), "5")
                                    .addLayer("out", new OCNNOutputLayer.Builder().hiddenLayerSize(10).nIn(100).build(), "6")
                                    .setInputTypes(InputType.convolutional(28, 28, 1))
                                    .setOutputs("out");
                            in = new INDArray[]{Nd4j.rand(networkDtype, 2, 1, 28, 28)};
                            break;
                        case 3:
                            b.addInputs("in1", "in2", "in3")
                                    .addVertex("1", new ElementWiseVertex(ElementWiseVertex.Op.Add), "in1", "in2")
                                    .addVertex("2a", new UnstackVertex(0, 2), "1")
                                    .addVertex("2b", new UnstackVertex(1, 2), "1")
                                    .addVertex("3", new StackVertex(), "2a", "2b")
                                    .addVertex("4", new DuplicateToTimeSeriesVertex("in3"), "3")
                                    .addVertex("5", new ReverseTimeSeriesVertex(), "4")
                                    .addLayer("6", new GlobalPoolingLayer(PoolingType.AVG), "5")
                                    .addVertex("7", new LastTimeStepVertex("in3"), "in3")
                                    .addVertex("8", new MergeVertex(), "6", "7")
                                    .addVertex("9", new PreprocessorVertex(new ComposableInputPreProcessor()), "8")
                                    .addLayer("out", new OutputLayer.Builder().nOut(10).build(), "9")
                                    .setInputTypes(InputType.feedForward(8), InputType.feedForward(8), InputType.recurrent(8))
                                    .setOutputs("out");
                            in = new INDArray[]{Nd4j.rand(networkDtype, 2, 8), Nd4j.rand(networkDtype, 2, 8), Nd4j.rand(networkDtype, 2, 8, 5)};
                            break;
                        case 4:
                            b.addInputs("in1", "in2")
                                    .addLayer("1", new LSTM.Builder().nOut(8).build(), "in1")
                                    .addVertex("preproc1", new PreprocessorVertex(new RnnToCnnPreProcessor(2, 2, 2)), "1")
                                    .addVertex("preproc2", new PreprocessorVertex(new CnnToRnnPreProcessor(2, 2, 2)), "preproc1")
                                    .addLayer("pool", new GlobalPoolingLayer(), "preproc2")
                                    .addLayer("pool2", new GlobalPoolingLayer(), "in2")
                                    .addLayer("out", new OutputLayer.Builder().nOut(10).build(), "pool", "pool2")
                                    .setInputTypes(InputType.recurrent(8), InputType.convolutional(28, 28, 1))
                                    .setOutputs("out");
                            in = new INDArray[]{Nd4j.rand(networkDtype, 2, 8, 5), Nd4j.rand(networkDtype, 2, 1, 28, 28)};
                            break;
                        case 5:
                            b.addInputs("in1", "in2")
                                    .addVertex("fv", new FrozenVertex(new ScaleVertex(2.0)), "in1")
                                    .addLayer("1", new DenseLayer.Builder().nOut(5).build(), "fv")
                                    .addLayer("2", new DenseLayer.Builder().nOut(5).build(), "in2")
                                    .addVertex("v", new L2Vertex(), "1", "2")
                                    .addLayer("out", new OutputLayer.Builder().nOut(10).build(), "v")
                                    .setInputTypes(InputType.feedForward(5), InputType.feedForward(5))
                                    .setOutputs("out");
                            in = new INDArray[]{Nd4j.rand(networkDtype, 2, 5), Nd4j.rand(networkDtype, 2, 5)};
                            break;
                        case 6:
                            b.addInputs("in")
                                    .addLayer("1", new LSTM.Builder().nOut(5).build(), "in")
                                    .addVertex("2", new PreprocessorVertex(new KerasFlattenRnnPreprocessor(5, 4)), "1")
                                    .addLayer("out", new OutputLayer.Builder().nOut(10).build(), "2")
                                    .setOutputs("out")
                                    .setInputTypes(InputType.recurrent(5, 4));
                            in = new INDArray[]{Nd4j.rand(networkDtype, 2, 5, 4)};
                            break;
                        case 7:
                            b.addInputs("in")
                                    .addLayer("1", new ConvolutionLayer.Builder().kernelSize(2, 2).nOut(5).convolutionMode(ConvolutionMode.Same).build(), "in")
                                    .addVertex("2", new PreprocessorVertex(new CnnToFeedForwardPreProcessor(28, 28, 5)), "1")
                                    .addLayer("out", new OutputLayer.Builder().nOut(10).build(), "2")
                                    .setOutputs("out")
                                    .setInputTypes(InputType.convolutional(28, 28, 1));
                            in = new INDArray[]{Nd4j.rand(networkDtype, 2, 1, 28, 28)};
                            break;
                    }

                    ComputationGraph net = new ComputationGraph(b.build());
                    net.init();

                    INDArray label = TestUtils.randomOneHot(2, 10).castTo(networkDtype);

                    INDArray out = net.outputSingle(in);
                    assertEquals(msg, networkDtype, out.dataType());
                    Map<String, INDArray> ff = net.feedForward(in, false);
                    for (Map.Entry<String, INDArray> e : ff.entrySet()) {
                        if (e.getKey().equals("in"))
                            continue;
                        String s = msg + " - layer: " + e.getKey();
                        assertEquals(s, networkDtype, e.getValue().dataType());
                    }

                    net.setInputs(in);
                    net.setLabels(label);
                    net.computeGradientAndScore();

                    net.fit(new MultiDataSet(in, new INDArray[]{label}));

                    logUsedClasses(net);

                    //Now, test mismatched dtypes for input/labels:
                    for (DataType inputLabelDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                        INDArray[] in2 = new INDArray[in.length];
                        for (int i = 0; i < in.length; i++) {
                            in2[i] = in[i].castTo(inputLabelDtype);
                        }
                        INDArray label2 = label.castTo(inputLabelDtype);
                        net.output(in2);
                        net.setInputs(in2);
                        net.setLabels(label2);
                        net.computeGradientAndScore();

                        net.fit(new MultiDataSet(in2, new INDArray[]{label2}));
                    }
                }
            }
        }
    }

    @Test
    public void testLocallyConnected() {
        for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                assertEquals(globalDtype, Nd4j.dataType());
                assertEquals(globalDtype, Nd4j.defaultFloatingPointType());

                INDArray[] in = null;
                for (int test = 0; test < 2; test++) {
                    String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype + ", test=" + test;

                    ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
                            .dataType(networkDtype)
                            .seed(123)
                            .updater(new NoOp())
                            .weightInit(WeightInit.XAVIER)
                            .convolutionMode(ConvolutionMode.Same)
                            .graphBuilder();

                    INDArray label;
                    switch (test) {
                        case 0:
                            b.addInputs("in")
                                    .addLayer("1", new LSTM.Builder().nOut(5).build(), "in")
                                    .addLayer("2", new LocallyConnected1D.Builder().kernelSize(2).nOut(4).build(), "1")
                                    .addLayer("out", new RnnOutputLayer.Builder().nOut(10).build(), "2")
                                    .setOutputs("out")
                                    .setInputTypes(InputType.recurrent(5, 2));
                            in = new INDArray[]{Nd4j.rand(networkDtype, 2, 5, 2)};
                            label = TestUtils.randomOneHotTimeSeries(2, 10, 2);
                            break;
                        case 1:
                            b.addInputs("in")
                                    .addLayer("1", new ConvolutionLayer.Builder().kernelSize(2, 2).nOut(5).convolutionMode(ConvolutionMode.Same).build(), "in")
                                    .addLayer("2", new LocallyConnected2D.Builder().kernelSize(2, 2).nOut(5).build(), "1")
                                    .addLayer("out", new OutputLayer.Builder().nOut(10).build(), "2")
                                    .setOutputs("out")
                                    .setInputTypes(InputType.convolutional(8, 8, 1));
                            in = new INDArray[]{Nd4j.rand(networkDtype, 2, 1, 8, 8)};
                            label = TestUtils.randomOneHot(2, 10).castTo(networkDtype);
                            break;
                        default:
                            throw new RuntimeException();
                    }

                    ComputationGraph net = new ComputationGraph(b.build());
                    net.init();

                    INDArray out = net.outputSingle(in);
                    assertEquals(msg, networkDtype, out.dataType());
                    Map<String, INDArray> ff = net.feedForward(in, false);
                    for (Map.Entry<String, INDArray> e : ff.entrySet()) {
                        if (e.getKey().equals("in"))
                            continue;
                        String s = msg + " - layer: " + e.getKey();
                        assertEquals(s, networkDtype, e.getValue().dataType());
                    }

                    net.setInputs(in);
                    net.setLabels(label);
                    net.computeGradientAndScore();

                    net.fit(new MultiDataSet(in, new INDArray[]{label}));

                    logUsedClasses(net);

                    //Now, test mismatched dtypes for input/labels:
                    for (DataType inputLabelDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                        INDArray[] in2 = new INDArray[in.length];
                        for (int i = 0; i < in.length; i++) {
                            in2[i] = in[i].castTo(inputLabelDtype);
                        }
                        INDArray label2 = label.castTo(inputLabelDtype);
                        net.output(in2);
                        net.setInputs(in2);
                        net.setLabels(label2);
                        net.computeGradientAndScore();

                        net.fit(new MultiDataSet(in2, new INDArray[]{label2}));
                    }
                }
            }
        }
    }

    @Test
    public void testAttentionDTypes() {
        for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                assertEquals(globalDtype, Nd4j.dataType());
                assertEquals(globalDtype, Nd4j.defaultFloatingPointType());

                String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype;

                int mb = 3;
                int nIn = 3;
                int nOut = 5;
                int tsLength = 4;
                int layerSize = 8;
                int numQueries = 6;

                INDArray in = Nd4j.rand(networkDtype, new long[]{mb, nIn, tsLength});
                INDArray labels = TestUtils.randomOneHot(mb, nOut);

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .dataType(networkDtype)
                        .activation(Activation.TANH)
                        .updater(new NoOp())
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .layer(new LSTM.Builder().nOut(layerSize).build())
                        .layer(new SelfAttentionLayer.Builder().nOut(8).nHeads(2).projectInput(true).build())
                        .layer(new LearnedSelfAttentionLayer.Builder().nOut(8).nHeads(2).nQueries(numQueries).projectInput(true).build())
                        .layer(new RecurrentAttentionLayer.Builder().nIn(layerSize).nOut(layerSize).nHeads(1).projectInput(false).hasBias(false).build())
                        .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build())
                        .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .setInputType(InputType.recurrent(nIn))
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                INDArray out = net.output(in);
                assertEquals(msg, networkDtype, out.dataType());
                List<INDArray> ff = net.feedForward(in);
                for (int i = 0; i < ff.size(); i++) {
                    String s = msg + " - layer " + (i - 1) + " - " + (i == 0 ? "input" : net.getLayer(i - 1).conf().getLayer().getClass().getSimpleName());
                    assertEquals(s, networkDtype, ff.get(i).dataType());
                }

                net.setInput(in);
                net.setLabels(labels);
                net.computeGradientAndScore();

                net.fit(new DataSet(in, labels));

                logUsedClasses(net);

                //Now, test mismatched dtypes for input/labels:
                for (DataType inputLabelDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                    INDArray in2 = in.castTo(inputLabelDtype);
                    INDArray label2 = labels.castTo(inputLabelDtype);
                    net.output(in2);
                    net.setInput(in2);
                    net.setLabels(label2);
                    net.computeGradientAndScore();

                    net.fit(new DataSet(in2, label2));
                }
            }
        }
    }


    @Test
    public void testAttentionDTypes2() {
        int nIn = 3;
        int nOut = 5;
        int tsLength = 4;
        int layerSize = 8;
        int mb = 3;

        for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            Nd4j.setDefaultDataTypes(globalDtype, globalDtype);
            for (DataType networkDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {

                assertEquals(globalDtype, Nd4j.dataType());
                assertEquals(globalDtype, Nd4j.defaultFloatingPointType());

                String msg = "Global dtype: " + globalDtype + ", network dtype: " + networkDtype;

                INDArray in = Nd4j.rand(networkDtype, new long[]{mb, nIn, tsLength});
                INDArray labels = TestUtils.randomOneHot(mb, nOut).castTo(networkDtype);
                String maskType = "inputMask";

                INDArray inMask = Nd4j.ones(networkDtype, mb, tsLength);
                for (int i = 0; i < mb; i++) {
                    int firstMaskedStep = tsLength - 1 - i;
                    if (firstMaskedStep == 0) {
                        firstMaskedStep = tsLength;
                    }
                    for (int j = firstMaskedStep; j < tsLength; j++) {
                        inMask.putScalar(i, j, 0.0);
                    }
                }

                String name = "testAttentionVertex() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType;
                System.out.println("Starting test: " + name);


                ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                        .dataType(networkDtype)
                        .activation(Activation.TANH)
                        .updater(new NoOp())
                        .weightInit(WeightInit.XAVIER)
                        .graphBuilder()
                        .addInputs("input")
                        .addLayer("lstmKeys", new LSTM.Builder().nOut(layerSize).build(), "input")
                        .addLayer("lstmQueries", new LSTM.Builder().nOut(layerSize).build(), "input")
                        .addLayer("lstmValues", new LSTM.Builder().nOut(layerSize).build(), "input")
                        .addVertex("attention", new AttentionVertex.Builder().nOut(8).nHeads(2).projectInput(true).nInQueries(layerSize).nInKeys(layerSize).nInValues(layerSize).build(), "lstmQueries", "lstmKeys", "lstmValues")
                        .addLayer("pooling", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build(), "attention")
                        .addLayer("output", new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build(), "pooling")
                        .setOutputs("output")
                        .setInputTypes(InputType.recurrent(nIn))
                        .build();
                ComputationGraph net = new ComputationGraph(conf);
                net.init();

                INDArray out = net.outputSingle(in);
                assertEquals(msg, networkDtype, out.dataType());
                Map<String,INDArray> ff = net.feedForward(in, false);
                for(Map.Entry<String,INDArray> e : ff.entrySet()){
                    String s = msg + " - layer " + e.getKey();
                    assertEquals(s, networkDtype, e.getValue().dataType());
                }

                net.setInput(0, in);
                net.setLabels(labels);
                net.computeGradientAndScore();

                net.fit(new DataSet(in, labels));

                logUsedClasses(net);

                //Now, test mismatched dtypes for input/labels:
                for (DataType inputLabelDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                    INDArray in2 = in.castTo(inputLabelDtype);
                    INDArray label2 = labels.castTo(inputLabelDtype);
                    net.output(in2);
                    net.setInput(0, in2);
                    net.setLabels(label2);
                    net.computeGradientAndScore();

                    net.fit(new DataSet(in2, label2));
                }
            }
        }
    }
}
