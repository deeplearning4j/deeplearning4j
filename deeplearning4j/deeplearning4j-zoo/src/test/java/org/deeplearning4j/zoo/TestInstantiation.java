/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.BenchmarkDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.zoo.model.*;
import org.deeplearning4j.zoo.model.helper.DarknetHelper;
import org.junit.After;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assume.assumeTrue;

/**
 * Tests workflow for zoo model instantiation.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class TestInstantiation extends BaseDL4JTest {

    protected static void ignoreIfCuda(){
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if("CUDA".equalsIgnoreCase(backend)) {
            log.warn("IGNORING TEST ON CUDA DUE TO CI CRASHES - SEE ISSUE #7657");
            assumeTrue(false);
        }
    }

    @After
    public void after() throws Exception {
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();
        Thread.sleep(1000);
        System.gc();
    }

    @Override
    public DataType getDataType(){
        return DataType.FLOAT;
    }

    @Test
    public void testCnnTrainingDarknet() throws Exception {
        runTest(Darknet19.builder().numClasses(10).build(), "Darknet19", 10);
    }

    @Test
    public void testCnnTrainingTinyYOLO() throws Exception {
        runTest(TinyYOLO.builder().numClasses(10).build(), "TinyYOLO", 10);
    }

    @Test @Ignore("AB 2019/05/28 - Crashing on CI linux-x86-64 CPU only - Issue #7657")
    public void testCnnTrainingYOLO2() throws Exception {
        runTest(YOLO2.builder().numClasses(10).build(), "YOLO2", 10);
    }

    public static void runTest(ZooModel model, String modelName, int numClasses) throws Exception {
        ignoreIfCuda();
        int gridWidth = -1;
        int gridHeight = -1;
        if (modelName.equals("TinyYOLO") || modelName.equals("YOLO2")) {
            int[] inputShapes = model.metaData().getInputShape()[0];
            gridWidth = DarknetHelper.getGridWidth(inputShapes);
            gridHeight = DarknetHelper.getGridHeight(inputShapes);
            numClasses += 4;
        }

        // set up data iterator
        int[] inputShape = model.metaData().getInputShape()[0];
        DataSetIterator iter = new BenchmarkDataSetIterator(
                new int[]{8, inputShape[0], inputShape[1], inputShape[2]}, numClasses, 1,
                gridWidth, gridHeight);

        Model initializedModel = model.init();
        AsyncDataSetIterator async = new AsyncDataSetIterator(iter);
        if (initializedModel instanceof MultiLayerNetwork) {
            ((MultiLayerNetwork) initializedModel).fit(async);
        } else {
            ((ComputationGraph) initializedModel).fit(async);
        }
        async.shutdown();

        // clean up for current model
        model = null;
        initializedModel = null;
        async = null;
        iter = null;
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        System.gc();
        Thread.sleep(1000);
        System.gc();
    }


    @Test
    public void testInitPretrained() throws IOException {
        ignoreIfCuda();
        ZooModel model = ResNet50.builder().numClasses(0).build(); //num labels doesn't matter since we're getting pretrained imagenet
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();
        INDArray f = Nd4j.rand(new int[]{1, 3, 224, 224});
        INDArray[] result = initializedModel.output(f);
        assertArrayEquals(result[0].shape(), new long[]{1, 1000});

        //Test fitting. Not ewe need to use transfer learning, as ResNet50 has a dense layer, not an OutputLayer
        initializedModel = new TransferLearning.GraphBuilder(initializedModel)
                .removeVertexAndConnections("fc1000")
                .addLayer("fc1000", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nIn(2048).nOut(1000).activation(Activation.SOFTMAX).build(), "flatten_1")
                .setOutputs("fc1000")
                .build();
        initializedModel.fit(new org.nd4j.linalg.dataset.DataSet(f, TestUtils.randomOneHot(1, 1000, 12345)));

    }

    @Test
    public void testInitPretrainedVGG16() throws Exception {
        testInitPretrained(VGG16.builder().numClasses(0).build(), new long[]{1,3,224,224}, new long[]{1,1000});
    }

    @Test
    public void testInitPretrainedVGG19() throws Exception {
        testInitPretrained(VGG19.builder().numClasses(0).build(), new long[]{1,3,224,224}, new long[]{1,1000});
    }

    @Test @Ignore("AB 2019/05/28 - JVM crash on linux CUDA CI machines - Issue 7657")
    public void testInitPretrainedDarknet19() throws Exception {
        testInitPretrained(Darknet19.builder().numClasses(0).build(), new long[]{1,3,224,224}, new long[]{1,1000});
    }

    @Test @Ignore("AB 2019/05/28 - JVM crash on linux CUDA CI machines - Issue 7657")
    public void testInitPretrainedDarknet19S2() throws Exception {
        testInitPretrained(Darknet19.builder().numClasses(0).inputShape(new int[]{3,448,448}).build(), new long[]{1,3,448,448}, new long[]{1,1000});
    }

    @Test
    public void testInitPretrainedTinyYOLO() throws Exception {
        testInitPretrained(TinyYOLO.builder().numClasses(0).build(), new long[]{1,3,416,416}, new long[]{1,125,13,13});
    }

    @Test
    public void testInitPretrainedYOLO2() throws Exception {
        testInitPretrained(YOLO2.builder().numClasses(0).build(), new long[]{1,3,608,608}, new long[]{1, 425, 19, 19});
    }

    @Test
    public void testInitPretrainedXception() throws Exception {
        testInitPretrained(Xception.builder().numClasses(0).build(), new long[]{1,3,299,299}, new long[]{1, 1000});
    }

    @Test
    public void testInitPretrainedSqueezenet() throws Exception {
        testInitPretrained(SqueezeNet.builder().numClasses(0).build(), new long[]{1,3,227,227}, new long[]{1, 1000});
    }

    public void testInitPretrained(ZooModel model, long[] inShape, long[] outShape) throws Exception {
        ignoreIfCuda();
        assertTrue(model.pretrainedAvailable(PretrainedType.IMAGENET));

        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();
        INDArray[] result = initializedModel.output(Nd4j.rand(inShape));
        assertArrayEquals(result[0].shape(),outShape);

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        initializedModel.params().close();
        for(INDArray arr : result){
            arr.close();
        }
        System.gc();
    }


    @Test
    public void testInitRandomModelResNet50() throws IOException {
        testInitRandomModel(ResNet50.builder().numClasses(1000).build(), new long[]{1,3,224,224}, new long[]{1,1000});
    }

    @Test
    public void testInitRandomModelVGG16() throws IOException {
        testInitRandomModel(VGG16.builder().numClasses(1000).build(), new long[]{1,3,224,224}, new long[]{1,1000});
    }

    @Test
    public void testInitRandomModelVGG19() throws IOException {
        testInitRandomModel(VGG19.builder().numClasses(1000).build(), new long[]{1,3,224,224}, new long[]{1,1000});
    }

    @Test
    public void testInitRandomModelDarknet19() throws IOException {
        testInitRandomModel(Darknet19.builder().numClasses(1000).build(), new long[]{1,3,224,224}, new long[]{1,1000});
    }

    @Test
    public void testInitRandomModelDarknet19_2() throws IOException {
        testInitRandomModel(Darknet19.builder().inputShape(new int[]{3,448,448}).numClasses(1000).build(), new long[]{1,3,448,448}, new long[]{1,1000});
    }

    @Test
    public void testInitRandomModelXception() throws IOException {
        testInitRandomModel(Xception.builder().numClasses(1000).build(), new long[]{1,3,299,299}, new long[]{1,1000});
    }

    @Test @Ignore("AB - 2019/05/28 - JVM crash on CI - intermittent? Issue 7657")
    public void testInitRandomModelSqueezenet() throws IOException {
        testInitRandomModel(SqueezeNet.builder().numClasses(1000).build(), new long[]{1,3,227,227}, new long[]{1,1000});
    }

    @Test
    public void testInitRandomModelFaceNetNN4Small2() throws IOException {
        testInitRandomModel(FaceNetNN4Small2.builder().embeddingSize(100).numClasses(10).build(), new long[]{1,3,64,64}, new long[]{1,10});
    }

    @Test @Ignore("AB 2019/05/29 - Crashing on CI linux-x86-64 CPU only - Issue #7657")
    public void testInitRandomModelUNet() throws IOException {
        testInitRandomModel(UNet.builder().build(), new long[]{1,3,512,512}, new long[]{1,1,512,512});
    }


    public void testInitRandomModel(ZooModel model, long[] inShape, long[] outShape){
        ignoreIfCuda();
        //Test initialization of NON-PRETRAINED models

        log.info("Testing {}", model.getClass().getSimpleName());
        ComputationGraph initializedModel = model.init();
        INDArray f = Nd4j.rand(DataType.FLOAT, inShape);
        INDArray[] result = initializedModel.output(f);
        assertArrayEquals(result[0].shape(), outShape);
        INDArray l = outShape.length == 2 ? TestUtils.randomOneHot(1, (int)outShape[1], 12345) : Nd4j.rand(DataType.FLOAT, outShape);
        initializedModel.fit(new org.nd4j.linalg.dataset.DataSet(f, l));

        // clean up for current model
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        f.close();
        l.close();
        initializedModel.params().close();
        initializedModel.getFlattenedGradients().close();
        System.gc();
    }


    @Test
    public void testYolo4635() throws Exception {
        ignoreIfCuda();
        //https://github.com/deeplearning4j/deeplearning4j/issues/4635

        int nClasses = 10;
        TinyYOLO model = TinyYOLO.builder().numClasses(nClasses).build();
        ComputationGraph computationGraph = (ComputationGraph) model.initPretrained();
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(computationGraph, "conv2d_9");
    }

    @Test
    public void testTransferLearning() throws Exception {
        ignoreIfCuda();
        //https://github.com/deeplearning4j/deeplearning4j/issues/7193

        ComputationGraph cg = (ComputationGraph) ResNet50.builder().build().initPretrained();

        cg = new TransferLearning.GraphBuilder(cg)
                .addLayer("out", new LossLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).activation(Activation.IDENTITY).build(), "fc1000")
                .setInputTypes(InputType.convolutional(224, 224, 3))
                .setOutputs("out")
                .build();

    }
}
