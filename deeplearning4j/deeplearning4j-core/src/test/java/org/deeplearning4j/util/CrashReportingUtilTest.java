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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
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
    public void testMLN() throws Exception {
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

        System.out.println("///////////////////////////////////////////////////////////");
        System.out.println(str);
        System.out.println("///////////////////////////////////////////////////////////");

    }


}
