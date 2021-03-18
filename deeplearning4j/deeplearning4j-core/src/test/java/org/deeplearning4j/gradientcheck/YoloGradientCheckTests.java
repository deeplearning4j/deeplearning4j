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

package org.deeplearning4j.gradientcheck;

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.NoOp;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class YoloGradientCheckTests extends BaseDL4JTest {

    static {
        Nd4j.setDataType(DataType.DOUBLE);
    }


    @TempDir Path testDir;

    public static Stream<Arguments> params() {
        List<Arguments> args = new ArrayList<>();
        for(Nd4jBackend nd4jBackend : BaseNd4jTestWithBackends.BACKENDS) {
            for(CNN2DFormat format : CNN2DFormat.values()) {
                args.add(Arguments.of(format,nd4jBackend));
            }
        }
        return args.stream();
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    @ParameterizedTest
    @MethodSource("org.deeplearning4j.gradientcheck.YoloGradientCheckTests#params")
    public void testYoloOutputLayer(CNN2DFormat format,Nd4jBackend backend) {
        int depthIn = 2;
        int c = 3;
        int b = 3;

        int yoloDepth = b * (5 + c);
        Activation a = Activation.TANH;

        Nd4j.getRandom().setSeed(1234567);

        int[] minibatchSizes = {1, 3};
        int[] widths = new int[]{4, 7};
        int[] heights = new int[]{4, 5};
        double[] l1 = new double[]{0.0, 0.3};
        double[] l2 = new double[]{0.0, 0.4};

        for( int i = 0; i<widths.length; i++ ) {

            int w = widths[i];
            int h = heights[i];
            int mb = minibatchSizes[i];

            Nd4j.getRandom().setSeed(12345);
            INDArray bbPrior = Nd4j.rand(b, 2).muliRowVector(Nd4j.create(new double[]{w, h})).addi(0.1);

            Nd4j.getRandom().setSeed(12345);

            INDArray input, labels;
            if(format == CNN2DFormat.NCHW){
                input = Nd4j.rand(DataType.DOUBLE, mb, depthIn, h, w);
                labels = yoloLabels(mb, c, h, w);
            } else {
                input = Nd4j.rand(DataType.DOUBLE, mb, h, w, depthIn);
                labels = yoloLabels(mb, c, h, w).permute(0,2,3,1);
            }

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                    .dataType(DataType.DOUBLE)
                    .updater(new NoOp())
                    .activation(a)
                    .l1(l1[i]).l2(l2[i])
                    .convolutionMode(ConvolutionMode.Same)
                    .list()
                    .layer(new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1)
                            .dataFormat(format)
                            .nIn(depthIn).nOut(yoloDepth).build())//output: (5-2+0)/1+1 = 4
                    .layer(new Yolo2OutputLayer.Builder()
                            .boundingBoxPriors(bbPrior)
                            .build())
                    .setInputType(InputType.convolutional(h, w, depthIn, format))
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            String msg = "testYoloOutputLayer() - minibatch = " + mb + ", w=" + w + ", h=" + h + ", l1=" + l1[i] + ", l2=" + l2[i];
            System.out.println(msg);

            INDArray out = net.output(input);
            if(format == CNN2DFormat.NCHW){
                assertArrayEquals(new long[]{mb, yoloDepth, h, w}, out.shape());
            } else {
                assertArrayEquals(new long[]{mb, h, w, yoloDepth}, out.shape());
            }

            net.fit(input, labels);


            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.MLNConfig().net(net).input(input)
                    .minAbsoluteError(1e-6)
                    .labels(labels).subset(true).maxPerParam(100));

            assertTrue(gradOK,msg);
            TestUtils.testModelSerialization(net);
        }
    }

    private static INDArray yoloLabels(int mb, int c, int h, int w) {
        int labelDepth = 4 + c;
        INDArray labels = Nd4j.zeros(mb, labelDepth, h, w);
        //put 1 object per minibatch, at positions (0,0), (1,1) etc.
        //Positions for label boxes: (1,1) to (2,2), (2,2) to (4,4) etc

        for( int i = 0; i < mb; i++) {
            //Class labels
            labels.putScalar(i, 4 + i%c, i%h, i%w, 1);

            //BB coordinates (top left, bottom right)
            labels.putScalar(i, 0, 0, 0, i%w);
            labels.putScalar(i, 1, 0, 0, i%h);
            labels.putScalar(i, 2, 0, 0, (i%w)+1);
            labels.putScalar(i, 3, 0, 0, (i%h)+1);
        }

        return labels;
    }


    @ParameterizedTest
    @MethodSource("org.deeplearning4j.gradientcheck.YoloGradientCheckTests#params")
    public void yoloGradientCheckRealData(CNN2DFormat format,Nd4jBackend backend) throws Exception {
        Nd4j.getRandom().setSeed(12345);
        InputStream is1 = new ClassPathResource("yolo/VOC_TwoImage/JPEGImages/2007_009346.jpg").getInputStream();
        InputStream is2 = new ClassPathResource("yolo/VOC_TwoImage/Annotations/2007_009346.xml").getInputStream();
        InputStream is3 = new ClassPathResource("yolo/VOC_TwoImage/JPEGImages/2008_003344.jpg").getInputStream();
        InputStream is4 = new ClassPathResource("yolo/VOC_TwoImage/Annotations/2008_003344.xml").getInputStream();

        File dir = new File(testDir.toFile(),"testYoloOverfitting");
        dir.mkdirs();
        File jpg = new File(dir, "JPEGImages");
        File annot = new File(dir, "Annotations");
        jpg.mkdirs();
        annot.mkdirs();

        File imgOut = new File(jpg, "2007_009346.jpg");
        File annotationOut = new File(annot, "2007_009346.xml");

        try(FileOutputStream fos = new FileOutputStream(imgOut)){
            IOUtils.copy(is1, fos);
        } finally { is1.close(); }

        try(FileOutputStream fos = new FileOutputStream(annotationOut)){
            IOUtils.copy(is2, fos);
        } finally { is2.close(); }

        imgOut = new File(jpg, "2008_003344.jpg");
        annotationOut = new File(annot, "2008_003344.xml");
        try(FileOutputStream fos = new FileOutputStream(imgOut)){
            IOUtils.copy(is3, fos);
        } finally { is3.close(); }

        try(FileOutputStream fos = new FileOutputStream(annotationOut)){
            IOUtils.copy(is4, fos);
        } finally { is4.close(); }

        INDArray bbPriors = Nd4j.create(new double[][]{
                {3,3},
                {3,5},
                {5,7}});

        //2x downsampling to 13x13 = 26x26 input images
        //Required depth at output layer: 5B+C, with B=3, C=20 object classes, for VOC
        VocLabelProvider lp = new VocLabelProvider(dir.getPath());
        int h = 26;
        int w = 26;
        int c = 3;
        RecordReader rr = new ObjectDetectionRecordReader(h, w, c, 13, 13, lp);
        rr.initialize(new FileSplit(jpg));

        int nClasses = rr.getLabels().size();
        long depthOut = bbPriors.size(0) * (5 + nClasses);


        DataSetIterator iter = new RecordReaderDataSetIterator(rr,2,1,1,true);
        iter.setPreProcessor(new ImagePreProcessingScaler());

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
                .convolutionMode(ConvolutionMode.Same)
                .updater(new NoOp())
                .dist(new GaussianDistribution(0,0.1))
                .seed(12345)
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).nOut(4).build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).build())
                .layer(new ConvolutionLayer.Builder().activation(Activation.IDENTITY).kernelSize(3,3).stride(1,1).nOut(depthOut).build())
                .layer(new Yolo2OutputLayer.Builder()
                        .boundingBoxPriors(bbPriors)
                        .build())
                .setInputType(InputType.convolutional(h,w,c))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DataSet ds = iter.next();
        INDArray f = ds.getFeatures();
        INDArray l = ds.getLabels();

        boolean ok = GradientCheckUtil.checkGradients(new GradientCheckUtil.MLNConfig().net(net).input(f)
                .labels(l).inputMask(null).subset(true).maxPerParam(64));

        assertTrue(ok);
        TestUtils.testModelSerialization(net);
    }
}
