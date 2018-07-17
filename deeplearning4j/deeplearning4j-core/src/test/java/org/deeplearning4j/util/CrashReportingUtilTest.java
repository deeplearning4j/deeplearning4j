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

package org.deeplearning4j.util;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.*;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class CrashReportingUtilTest extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @After
    public void after(){
        //Reset dir
        CrashReportingUtil.crashDumpOutputDirectory(null);
    }

    @Test
    public void test() throws Exception {
        File dir = testDir.newFolder();
        CrashReportingUtil.crashDumpOutputDirectory(dir);

        int kernel = 2;
        int stride = 1;
        int padding = 0;
        PoolingType poolingType = PoolingType.MAX;
        int inputDepth = 1;
        int height = 28;
        int width = 28;


        MultiLayerConfiguration conf =
                new NeuralNetConfiguration.Builder().updater(new NoOp())
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 1))
                        .list().layer(0,
                        new ConvolutionLayer.Builder()
                                .kernelSize(kernel, kernel)
                                .stride(stride, stride)
                                .padding(padding, padding)
                                .nIn(inputDepth)
                                .nOut(3).build())
                        .layer(1, new SubsamplingLayer.Builder(poolingType)
                                .kernelSize(kernel, kernel)
                                .stride(stride, stride)
                                .padding(padding, padding)
                                .build())
                        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .nOut(10).build())
                        .setInputType(InputType.convolutionalFlat(height, width,
                                inputDepth))
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.addListeners(new ScoreIterationListener(1));

        //Test net that hasn't been trained yet
        Exception e = new Exception();
        CrashReportingUtil.writeMemoryCrashDump(net, e);

        File[] list = dir.listFiles();
        assertNotNull(list);
        assertEquals(1, list.length);
        String str = FileUtils.readFileToString(list[0]);
//        System.out.println(str);
        assertTrue(str.contains("Network Information"));
        assertTrue(str.contains("Layer Helpers"));
        assertTrue(str.contains("JavaCPP"));
        assertTrue(str.contains("ScoreIterationListener"));


        //Train:
        DataSetIterator iter = new EarlyTerminationDataSetIterator(new MnistDataSetIterator(32, true, 12345), 5);
        net.fit(iter);
        dir = testDir.newFolder();
        CrashReportingUtil.crashDumpOutputDirectory(dir);
        CrashReportingUtil.writeMemoryCrashDump(net, e);

        list = dir.listFiles();
        assertNotNull(list);
        assertEquals(1, list.length);
        str = FileUtils.readFileToString(list[0]);
        assertTrue(str.contains("Network Information"));
        assertTrue(str.contains("Layer Helpers"));
        assertTrue(str.contains("JavaCPP"));
        assertTrue(str.contains("ScoreIterationListener(1)"));

//        System.out.println("///////////////////////////////////////////////////////////");
//        System.out.println(str);
//        System.out.println("///////////////////////////////////////////////////////////");


        //Also test manual memory info
        String mlnMemoryInfo = net.memoryInfo(32, InputType.convolutionalFlat(28, 28, 1));
//        System.out.println("///////////////////////////////////////////////////////////");
//        System.out.println(mlnMemoryInfo);
//        System.out.println("///////////////////////////////////////////////////////////");

        assertTrue(mlnMemoryInfo.contains("Network Information"));
        assertTrue(mlnMemoryInfo.contains("Layer Helpers"));
        assertTrue(mlnMemoryInfo.contains("JavaCPP"));
        assertTrue(mlnMemoryInfo.contains("ScoreIterationListener(1)"));



        ////////////////////////////////////////
        //Same thing on ComputationGraph:
        dir = testDir.newFolder();
        CrashReportingUtil.crashDumpOutputDirectory(dir);

        ComputationGraph cg = net.toComputationGraph();
        cg.setListeners(new ScoreIterationListener(1));

        //Test net that hasn't been trained yet
        CrashReportingUtil.writeMemoryCrashDump(cg, e);

        list = dir.listFiles();
        assertNotNull(list);
        assertEquals(1, list.length);
        str = FileUtils.readFileToString(list[0]);
        assertTrue(str.contains("Network Information"));
        assertTrue(str.contains("Layer Helpers"));
        assertTrue(str.contains("JavaCPP"));
        assertTrue(str.contains("ScoreIterationListener(1)"));

        //Train:
        cg.fit(iter);
        dir = testDir.newFolder();
        CrashReportingUtil.crashDumpOutputDirectory(dir);
        CrashReportingUtil.writeMemoryCrashDump(cg, e);

        list = dir.listFiles();
        assertNotNull(list);
        assertEquals(1, list.length);
        str = FileUtils.readFileToString(list[0]);
        assertTrue(str.contains("Network Information"));
        assertTrue(str.contains("Layer Helpers"));
        assertTrue(str.contains("JavaCPP"));
        assertTrue(str.contains("ScoreIterationListener(1)"));

//        System.out.println("///////////////////////////////////////////////////////////");
//        System.out.println(str);
//        System.out.println("///////////////////////////////////////////////////////////");


        //Also test manual memory info
        String cgMemoryInfo = cg.memoryInfo(32, InputType.convolutionalFlat(28, 28, 1));
//        System.out.println("///////////////////////////////////////////////////////////");
//        System.out.println(cgMemoryInfo);
//        System.out.println("///////////////////////////////////////////////////////////");

        assertTrue(cgMemoryInfo.contains("Network Information"));
        assertTrue(cgMemoryInfo.contains("Layer Helpers"));
        assertTrue(cgMemoryInfo.contains("JavaCPP"));
        assertTrue(cgMemoryInfo.contains("ScoreIterationListener(1)"));

    }


}
