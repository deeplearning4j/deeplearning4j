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

package org.deeplearning4j.parallelism;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.deeplearning4j.parallelism.inference.InferenceObservable;
import org.deeplearning4j.parallelism.inference.observers.BasicInferenceObserver;
import org.deeplearning4j.parallelism.inference.observers.BatchedInferenceObservable;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.lang.reflect.Field;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class ParallelInferenceTest {
    private static MultiLayerNetwork model;
    private static DataSetIterator iterator;

    @Before
    public void setUp() throws Exception {
        if (model == null) {
            File file = new ClassPathResource("models/LenetMnistMLN.zip").getFile();
            model = ModelSerializer.restoreMultiLayerNetwork(file, true);

            iterator = new MnistDataSetIterator(1, false, 12345);
        }
    }

    @After
    public void tearDown() throws Exception {
        iterator.reset();
    }

    @Test(timeout = 30000L)
    public void testInferenceSequential1() throws Exception {

        long count0 = 0;
        long count1 = 0;

        //We can't guarantee that on any particular run each thread will get data - it might randomly be assigned to
        // only one. Consequently: we'll run the test multiple times and ensure that in at least *some* of the test
        // runs both workers get some data.
        for (int i = 0; i < 20 && (count0 == 0 || count1 == 0); i++) {
            iterator = new MnistDataSetIterator(1, false, 12345);

            ParallelInference inf =
                    new ParallelInference.Builder(model).inferenceMode(InferenceMode.SEQUENTIAL).workers(2).build();


            log.info("Features shape: {}",
                    Arrays.toString(iterator.next().getFeatures().shapeInfoDataBuffer().asInt()));

            INDArray array1 = inf.output(iterator.next().getFeatures());
            INDArray array2 = inf.output(iterator.next().getFeatures());

            assertFalse(array1.isAttached());
            assertFalse(array2.isAttached());

            INDArray array3 = inf.output(iterator.next().getFeatures());
            assertFalse(array3.isAttached());

            iterator.reset();

            evalClassifcationSingleThread(inf, iterator);

            count0 = inf.getWorkerCounter(0);
            count1 = inf.getWorkerCounter(1);
//            System.out.println("Counts: " + count0 + ", " + count1);
        }
        // both workers threads should have non-zero
        assertTrue(count0 > 0L);
        assertTrue(count1 > 0L);
    }

    @Test(timeout = 30000L)
    public void testInferenceSequential2() throws Exception {

        long count0 = 0;
        long count1 = 0;

        //We can't guarantee that on any particular run each thread will get data - it might randomly be assigned to
        // only one. Consequently: we'll run the test multiple times and ensure that in at least *some* of the test
        // runs both workers get some data.
        for (int i = 0; i < 20 && (count0 == 0 || count1 == 0); i++) {
            iterator = new MnistDataSetIterator(1, false, 12345);
            ParallelInference inf =
                    new ParallelInference.Builder(model).inferenceMode(InferenceMode.SEQUENTIAL).workers(2).build();


            log.info("Features shape: {}",
                    Arrays.toString(iterator.next().getFeatures().shapeInfoDataBuffer().asInt()));

            INDArray array1 = inf.output(iterator.next().getFeatures());
            INDArray array2 = inf.output(iterator.next().getFeatures());

            assertFalse(array1.isAttached());
            assertFalse(array2.isAttached());

            INDArray array3 = inf.output(iterator.next().getFeatures());
            assertFalse(array3.isAttached());

            iterator.reset();

            evalClassifcationMultipleThreads(inf, iterator, 10);

            // both workers threads should have non-zero
            count0 = inf.getWorkerCounter(0);
            count1 = inf.getWorkerCounter(1);
//            System.out.println("Counts: " + count0 + ", " + count1);
        }
        assertTrue(count0 > 0L);
        assertTrue(count1 > 0L);
    }


    @Test(timeout = 30000L)
    public void testInferenceBatched1() throws Exception {
        long count0 = 0;
        long count1 = 0;

        //We can't guarantee that on any particular run each thread will get data - it might randomly be assigned to
        // only one. Consequently: we'll run the test multiple times and ensure that in at least *some* of the test
        // runs both workers get some data.
        for( int i=0; i<20 && (count0 == 0 || count1 == 0); i++ ) {
            ParallelInference inf = new ParallelInference.Builder(model).inferenceMode(InferenceMode.BATCHED).batchLimit(8)
                    .workers(2).build();

            iterator = new MnistDataSetIterator(1, false, 12345);


            log.info("Features shape: {}",
                    Arrays.toString(iterator.next().getFeatures().shapeInfoDataBuffer().asInt()));

            INDArray array1 = inf.output(iterator.next().getFeatures());
            INDArray array2 = inf.output(iterator.next().getFeatures());

            assertFalse(array1.isAttached());
            assertFalse(array2.isAttached());

            INDArray array3 = inf.output(iterator.next().getFeatures());
            assertFalse(array3.isAttached());

            iterator.reset();

            evalClassifcationMultipleThreads(inf, iterator, 20);

            // both workers threads should have non-zero
            count0 = inf.getWorkerCounter(0);
            count1 = inf.getWorkerCounter(1);
//            System.out.println("Counts: " + count0 + ", " + count1);
        }
        assertTrue(count0 > 0L);
        assertTrue(count1 > 0L);
    }


    @Test
    public void testProvider1() throws Exception {
        LinkedBlockingQueue queue = new LinkedBlockingQueue();
        BasicInferenceObserver observer = new BasicInferenceObserver();

        ParallelInference.ObservablesProvider provider =
                        new ParallelInference.ObservablesProvider(10000000L, 100, queue);

        InferenceObservable observable1 = provider.setInput(observer, Nd4j.create(100));
        InferenceObservable observable2 = provider.setInput(observer, Nd4j.create(100));

        assertNotEquals(null, observable1);

        assertTrue(observable1 == observable2);
    }

    @Test
    public void testProvider2() throws Exception {
        LinkedBlockingQueue queue = new LinkedBlockingQueue();
        BasicInferenceObserver observer = new BasicInferenceObserver();
        ParallelInference.ObservablesProvider provider =
                        new ParallelInference.ObservablesProvider(10000000L, 100, queue);

        InferenceObservable observable1 = provider.setInput(observer, Nd4j.create(100).assign(1.0));
        InferenceObservable observable2 = provider.setInput(observer, Nd4j.create(100).assign(2.0));

        assertNotEquals(null, observable1);

        assertTrue(observable1 == observable2);

        List<Pair<INDArray[],INDArray[]>> l = observable1.getInputBatches();
        assertEquals(1, l.size());
        INDArray[] input = l.get(0).getFirst();
        assertNull(l.get(0).getSecond());

        assertEquals(1, input.length);
        assertArrayEquals(new long[] {2, 100}, input[0].shape());
        assertEquals(1.0f, input[0].tensorAlongDimension(0, 1).meanNumber().floatValue(), 0.001);
        assertEquals(2.0f, input[0].tensorAlongDimension(1, 1).meanNumber().floatValue(), 0.001);
    }

    @Test
    public void testProvider3() throws Exception {
        LinkedBlockingQueue queue = new LinkedBlockingQueue();
        BasicInferenceObserver observer = new BasicInferenceObserver();
        ParallelInference.ObservablesProvider provider = new ParallelInference.ObservablesProvider(10000000L, 2, queue);

        InferenceObservable observable1 = provider.setInput(observer, Nd4j.create(100).assign(1.0));
        InferenceObservable observable2 = provider.setInput(observer, Nd4j.create(100).assign(2.0));

        InferenceObservable observable3 = provider.setInput(observer, Nd4j.create(100).assign(3.0));


        assertNotEquals(null, observable1);
        assertNotEquals(null, observable3);

        assertTrue(observable1 == observable2);
        assertTrue(observable1 != observable3);

        List<Pair<INDArray[],INDArray[]>> l = observable1.getInputBatches();
        assertEquals(1, l.size());
        INDArray[] input = l.get(0).getFirst();
        assertNull(l.get(0).getSecond());

        assertEquals(1.0f, input[0].tensorAlongDimension(0, 1).meanNumber().floatValue(), 0.001);
        assertEquals(2.0f, input[0].tensorAlongDimension(1, 1).meanNumber().floatValue(), 0.001);


        l = observable3.getInputBatches();
        assertEquals(1, l.size());
        input = l.get(0).getFirst();
        assertNull(l.get(0).getSecond());
        assertEquals(3.0f, input[0].tensorAlongDimension(0, 1).meanNumber().floatValue(), 0.001);
    }

    @Test
    public void testProvider4() throws Exception {
        LinkedBlockingQueue queue = new LinkedBlockingQueue();
        BasicInferenceObserver observer = new BasicInferenceObserver();
        ParallelInference.ObservablesProvider provider = new ParallelInference.ObservablesProvider(10000000L, 4, queue);

        BatchedInferenceObservable observable1 =
                        (BatchedInferenceObservable) provider.setInput(observer, Nd4j.create(100).assign(1.0));
        BatchedInferenceObservable observable2 =
                        (BatchedInferenceObservable) provider.setInput(observer, Nd4j.create(100).assign(2.0));
        BatchedInferenceObservable observable3 =
                        (BatchedInferenceObservable) provider.setInput(observer, Nd4j.create(100).assign(3.0));

        INDArray bigOutput = Nd4j.create(3, 10);
        for (int i = 0; i < bigOutput.rows(); i++)
            bigOutput.getRow(i).assign((float) i);


        Field f = BatchedInferenceObservable.class.getDeclaredField("outputBatchInputArrays");
        f.setAccessible(true);
        List<int[]> l = new ArrayList<>();
        l.add(new int[]{0,2});
        f.set(observable3, l);

        f = BatchedInferenceObservable.class.getDeclaredField("inputs");
        f.setAccessible(true);
        f.set(observable3, Arrays.asList(new INDArray[]{bigOutput.getRow(0)},
                new INDArray[]{bigOutput.getRow(1)}, new INDArray[]{bigOutput.getRow(2)}));


        observable3.setOutputBatches(Collections.singletonList(new INDArray[]{bigOutput}));
        INDArray out = null;

        observable3.setPosition(0);
        out = observable3.getOutput()[0];
        assertArrayEquals(new long[] {1, 10}, out.shape());
        assertEquals(0.0f, out.meanNumber().floatValue(), 0.01f);

        observable3.setPosition(1);
        out = observable3.getOutput()[0];
        assertArrayEquals(new long[] {1, 10}, out.shape());
        assertEquals(1.0f, out.meanNumber().floatValue(), 0.01f);

        observable3.setPosition(2);
        out = observable3.getOutput()[0];
        assertArrayEquals(new long[] {1, 10}, out.shape());
        assertEquals(2.0f, out.meanNumber().floatValue(), 0.01f);
    }


    protected void evalClassifcationSingleThread(@NonNull ParallelInference inf, @NonNull DataSetIterator iterator) {
        DataSet ds = iterator.next();
        log.info("NumColumns: {}", ds.getLabels().columns());
        iterator.reset();
        Evaluation eval = new Evaluation(ds.getLabels().columns());
        int count = 0;
        while (iterator.hasNext() && (count++ < 100)) {
            ds = iterator.next();
            INDArray output = inf.output(ds.getFeatures());
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
    }

    protected void evalClassifcationMultipleThreads(@NonNull ParallelInference inf, @NonNull DataSetIterator iterator,
                    int numThreads) throws Exception {
        DataSet ds = iterator.next();
        log.info("NumColumns: {}", ds.getLabels().columns());
        iterator.reset();
        Evaluation eval = new Evaluation(ds.getLabels().columns());
        final Queue<DataSet> dataSets = new LinkedBlockingQueue<>();
        final Queue<Pair<INDArray, INDArray>> outputs = new LinkedBlockingQueue<>();
        int cnt = 0;
        // first of all we'll build datasets
        while (iterator.hasNext() && cnt < 256) {
            ds = iterator.next();
            dataSets.add(ds);
            cnt++;
        }

        // now we'll build outputs in parallel
        Thread[] threads = new Thread[numThreads];
        for (int i = 0; i < numThreads; i++) {
            threads[i] = new Thread(new Runnable() {
                @Override
                public void run() {
                    DataSet ds;
                    while ((ds = dataSets.poll()) != null) {
                        INDArray output = inf.output(ds);
                        outputs.add(Pair.makePair(ds.getLabels(), output));
                    }
                }
            });
        }

        for (int i = 0; i < numThreads; i++) {
            threads[i].start();
        }

        for (int i = 0; i < numThreads; i++) {
            threads[i].join();
        }

        // and now we'll evaluate in single thread once again
        Pair<INDArray, INDArray> output;
        while ((output = outputs.poll()) != null) {
            eval.eval(output.getFirst(), output.getSecond());
        }
        log.info(eval.stats());
    }


    @Test(timeout = 30000L)
    public void testParallelInferenceVariableLengthTS() throws Exception {
        Nd4j.getRandom().setSeed(12345);

        int nIn = 10;
        int[] tsLengths = {3,5,7,10,50,100};

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .list()
                .layer(new LSTM.Builder().nIn(nIn).nOut(5).build())
                .layer(new RnnOutputLayer.Builder().nIn(5).nOut(5).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        for( InferenceMode m : InferenceMode.values()) {
            for( int w : new int[]{1,2}) {

                final ParallelInference inf =
                        new ParallelInference.Builder(net)
                                .inferenceMode(m)
                                .batchLimit(20)
                                .queueLimit(64)
                                .workers(w).build();

                List<INDArray> arrs = new ArrayList<>();
                List<INDArray> exp = new ArrayList<>();
                for (int l : tsLengths) {
                    INDArray in = Nd4j.rand(new int[]{1, nIn, l});
                    arrs.add(in);
                    INDArray out = net.output(in);
                    exp.add(out);
                }

                testParallelInference(inf, arrs, exp);

                inf.shutdown();
            }
        }
    }

    @Test(timeout = 60000L)
    public void testParallelInferenceVariableLengthTS2() throws Exception {
        Nd4j.getRandom().setSeed(12345);

        int nIn = 10;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .list()
                .layer(new LSTM.Builder().nIn(nIn).nOut(5).build())
                .layer(new RnnOutputLayer.Builder().nIn(5).nOut(5).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        int[] defaultSize = new int[]{1, 10, 5};

        for( InferenceMode m : InferenceMode.values()) {
            for( int w : new int[]{2,3}) {

                final ParallelInference inf =
                        new ParallelInference.Builder(net)
                                .inferenceMode(m)
                                .batchLimit(20)
                                .queueLimit(64)
                                .workers(w).build();

                List<INDArray> arrs = new ArrayList<>();
                List<INDArray> exp = new ArrayList<>();

                Random r = new Random();
                for( int i=0; i<500; i++ ){
                    int[] shape = defaultSize;
                    if(r.nextDouble() < 0.4){
                        shape = new int[]{r.nextInt(5)+1, 10, r.nextInt(10)+1};
                    }

                    INDArray in = Nd4j.rand(shape);
                    arrs.add(in);
                    INDArray out = net.output(in);
                    exp.add(out);
                }
                testParallelInference(inf, arrs, exp);

                inf.shutdown();
            }
        }
    }



    @Test(timeout = 30000L)
    public void testParallelInferenceVariableSizeCNN() throws Exception {
        //Variable size input for CNN model - for example, YOLO models
        //In these cases, we can't batch and have to execute the different size inputs separately

        Nd4j.getRandom().setSeed(12345);

        int nIn = 3;
        int[][] shapes = new int[][]{
                {1,nIn,10,10},
                {1,nIn,10,15},
                {1,nIn,20,15},
                {1,nIn,20,20},
                {1,nIn,30,30},
                {1,nIn,40,40},
                {1,nIn,40,45},
        };

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .list()
                .layer(new ConvolutionLayer.Builder().nIn(nIn).nOut(5).build())
                .layer(new CnnLossLayer())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        for( InferenceMode m : InferenceMode.values()) {
            for( int w : new int[]{1,2}) {

                final ParallelInference inf =
                        new ParallelInference.Builder(net)
                                .inferenceMode(m)
                                .batchLimit(20)
                                .queueLimit(64)
                                .workers(w).build();

                List<INDArray> arrs = new ArrayList<>();
                List<INDArray> exp = new ArrayList<>();
                for (int[] shape : shapes) {
                    INDArray in = Nd4j.rand(shape);
                    arrs.add(in);
                    INDArray out = net.output(in);
                    exp.add(out);
                }

                testParallelInference(inf, arrs, exp);

                inf.shutdown();
            }
        }
    }


    @Test(timeout = 30000L)
    public void testParallelInferenceVariableSizeCNN2() throws Exception {
        //Variable size input for CNN model - for example, YOLO models
        //In these cases, we can't batch and have to execute the different size inputs separately

        Nd4j.getRandom().setSeed(12345);

        int nIn = 3;
        int[] defaultShape = new int[]{1, nIn, 16, 16};

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .convolutionMode(ConvolutionMode.Same)
                .list()
                .layer(new ConvolutionLayer.Builder().nIn(nIn).nOut(5).build())
                .layer(new CnnLossLayer())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        for( InferenceMode m : InferenceMode.values()) {
            for( int w : new int[]{1,2}) {

                final ParallelInference inf =
                        new ParallelInference.Builder(net)
                                .inferenceMode(m)
                                .batchLimit(20)
                                .queueLimit(64)
                                .workers(w).build();

                List<INDArray> arrs = new ArrayList<>();
                List<INDArray> exp = new ArrayList<>();
                Random r = new Random();
                for( int i=0; i<500; i++ ){
                    int[] shape = defaultShape;
                    if(r.nextDouble() < 0.4){
                        shape = new int[]{r.nextInt(5)+1, nIn, 10, r.nextInt(10)+1};
                    }

                    INDArray in = Nd4j.rand(shape);
                    arrs.add(in);
                    INDArray out = net.output(in);
                    exp.add(out);
                }
                testParallelInference(inf, arrs, exp);

                inf.shutdown();
            }
        }
    }

    @Test(timeout = 20000L)
    public void testParallelInferenceErrorPropagation(){

        int nIn = 10;
        int wrongNIn = 5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .list()
                .layer(new DenseLayer.Builder().nIn(nIn).nOut(5).build())
                .layer(new OutputLayer.Builder().nIn(5).nOut(5).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray inOk = Nd4j.ones(1, nIn);
        INDArray inWrong = Nd4j.ones(1, wrongNIn);

        INDArray expOk = net.output(inOk);

        for( InferenceMode m : InferenceMode.values()) {
            for (int w : new int[]{1, 2}) {

                final ParallelInference inf =
                        new ParallelInference.Builder(net)
                                .inferenceMode(m)
                                .batchLimit(20)
                                .queueLimit(64)
                                .workers(w).build();

                INDArray actOk = inf.output(inOk);
                assertEquals(expOk, actOk);

                try {
                    inf.output(inWrong);
                    fail("Expected exception");
                } catch (DL4JInvalidInputException e){
                    //OK
                    System.out.println("Expected exception: " + e.getMessage());
                } catch (Exception e){
                    e.printStackTrace();
                    fail("Expected other exception type");
                }

                actOk = inf.output(inOk);
                assertEquals(expOk, actOk);

                inf.shutdown();
            }
        }
    }

    @Test
    public void testInputMaskingCyclic() throws Exception {
        for (int e = 0; e < 3; e++) {
            testInputMasking();
            log.info("Iteration: {} finished", e);
            System.gc();
        }
    }

    @Test(timeout = 60000)
    public void testInputMasking() throws Exception {
        Nd4j.getRandom().setSeed(12345);

        int nIn = 10;
        int tsLength = 16;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .list()
                .layer(new LSTM.Builder().nIn(nIn).nOut(5).build())
                .layer(new GlobalPoolingLayer(PoolingType.AVG))
                .layer(new OutputLayer.Builder().nIn(5).nOut(5).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Random r = new Random();
        for( InferenceMode m : InferenceMode.values()) {
            log.info("Testing inference mode: [{}]", m);
            for( int w : new int[]{1,2}) {
                for (boolean randomTSLength : new boolean[]{false, true}) {

                    final ParallelInference inf =
                            new ParallelInference.Builder(net)
                                    .inferenceMode(m)
                                    .batchLimit(5)
                                    .queueLimit(64)
                                    .workers(w).build();

                    List<INDArray> in = new ArrayList<>();
                    List<INDArray> inMasks = new ArrayList<>();
                    List<INDArray> exp = new ArrayList<>();
                    for (int i = 0; i < 100; i++) {
                        int currTSLength = (randomTSLength ? 1 + r.nextInt(tsLength) : tsLength);
                        int currNumEx = 1 + r.nextInt(3);
                        INDArray inArr = Nd4j.rand(new int[]{currNumEx, nIn, currTSLength});
                        in.add(inArr);

                        INDArray inMask = null;
                        if(r.nextDouble() < 0.5){
                            inMask = Nd4j.ones(currNumEx, currTSLength);
                            for( int mb = 0; mb < currNumEx; mb++) {
                                if (currTSLength > 1) {
                                    int firstMaskedStep = 1 + r.nextInt(currTSLength);
                                    for (int j = firstMaskedStep; j < currTSLength; j++) {
                                        inMask.putScalar(mb, j, 0.0);
                                    }
                                }
                            }
                        }
                        inMasks.add(inMask);

                        INDArray out = net.output(inArr, false, inMask, null);
                        exp.add(out);
                    }

                    testParallelInference(inf, in, inMasks, exp);

                    inf.shutdown();
                }
            }
        }
    }

    @Test(timeout = 20000L)
    public void testModelUpdate_1() throws Exception {
        int nIn = 5;

        val conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .layer("out0", new OutputLayer.Builder().nIn(nIn).nOut(4).build(), "in")
                .layer("out1", new OutputLayer.Builder().nIn(nIn).nOut(6).build(), "in")
                .setOutputs("out0", "out1")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        val inf = new ParallelInference.Builder(net)
                        .inferenceMode(InferenceMode.SEQUENTIAL)
                        .batchLimit(5)
                        .queueLimit(64)
                        .workers(4)
                        .build();

        // imitating use of the original model
        for (int e = 0; e < 10; e++) {
            val output = inf.output(new INDArray[]{Nd4j.createUninitialized(1, 5)});
            assertNotNull(output);
            assertNotEquals(0, output.length);
        }

        val modelsBefore = inf.getCurrentModelsFromWorkers();
        assertEquals(4, modelsBefore.length);

        boolean passed = false;
        int cnt0 = 0;
        for (val m:modelsBefore) {
            // model can be null for some of the workers yet, due to race condition
            if (m != null) {
                assertEquals("Failed at model [" + cnt0 + "]", net.params(), m.params());
                passed = true;
            }
            cnt0++;
        }
        assertTrue(passed);


        val conf2 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .layer("out0", new OutputLayer.Builder().nIn(nIn).nOut(4).build(), "in")
                .layer("out1", new OutputLayer.Builder().nIn(nIn).nOut(6).build(), "in")
                .layer("out2", new OutputLayer.Builder().nIn(nIn).nOut(8).build(), "in")
                .setOutputs("out0", "out1", "out2")
                .build();

        val net2 = new ComputationGraph(conf2);
        net2.init();

        inf.updateModel(net2);

        val modelsAfter = inf.getCurrentModelsFromWorkers();
        assertEquals(4, modelsAfter.length);

        cnt0 = 0;
        for (val m:modelsAfter) {
            assertNotNull("Failed at model [" + cnt0 + "]", m);
            assertEquals("Failed at model [" + cnt0++ + "]", net2.params(), m.params());
        }

        inf.shutdown();
    }

    @Test(timeout = 60000L)
    public void testMultiOutputNet() throws Exception {

        int nIn = 5;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .layer("out0", new OutputLayer.Builder().nIn(nIn).nOut(4).build(), "in")
                .layer("out1", new OutputLayer.Builder().nIn(nIn).nOut(6).build(), "in")
                .setOutputs("out0", "out1")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        Random r = new Random();
        for( InferenceMode m : InferenceMode.values()) {
            for( int w : new int[]{1,2}) {

                final ParallelInference inf =
                        new ParallelInference.Builder(net)
                                .inferenceMode(m)
                                .batchLimit(5)
                                .queueLimit(64)
                                .workers(w).build();

                List<INDArray[]> in = new ArrayList<>();
                List<INDArray[]> exp = new ArrayList<>();
                for (int i = 0; i < 100; i++) {
                    int currNumEx = 1 + r.nextInt(3);
                    INDArray inArr = Nd4j.rand(new int[]{currNumEx, nIn});
                    in.add(new INDArray[]{inArr});

                    INDArray[] out = net.output(inArr);
                    exp.add(out);
                }

                testParallelInferenceMulti(inf, in, null, exp);
                inf.shutdown();
            }
        }

    }


    private static void testParallelInference(ParallelInference inf, List<INDArray> in, List<INDArray> exp) throws Exception {
        testParallelInference(inf, in, null, exp);
    }

    private static void testParallelInference(ParallelInference inf, List<INDArray> in, List<INDArray> inMasks, List<INDArray> exp) throws Exception {
        final INDArray[] act = new INDArray[in.size()];
        final AtomicInteger counter = new AtomicInteger(0);
        final AtomicInteger failedCount = new AtomicInteger(0);

        val threads = new ArrayList<Thread>();

        for( int i=0; i<in.size(); i++ ){
            final int j=i;
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    try{
                        INDArray inMask = (inMasks == null ? null : inMasks.get(j));
                        act[j] = inf.output(in.get(j), inMask);
                        counter.incrementAndGet();
                    } catch (Exception e){
                        e.printStackTrace();
                        failedCount.incrementAndGet();
                    }
                }
            });

            t.start();

            threads.add(t);
        }

        // wait for ALL started threads
        for (val t: threads) {
            if (failedCount.get() > 0)
                throw new RuntimeException("One of threads failed!");
            t.join();
        }

        assertEquals(0, failedCount.get());
        assertEquals(in.size(), counter.get());
        for( int i=0; i<in.size(); i++ ){
            INDArray e = exp.get(i);
            INDArray a = act[i];

//            float[] fe = e.dup().data().asFloat();
//            float[] fa = a.dup().data().asFloat();
//            System.out.println(Arrays.toString(fe));
//            System.out.println(Arrays.toString(fa));
//            assertArrayEquals(fe, fa, 1e-8f);
//            System.out.println(Arrays.toString(e.shape()) + " vs " + Arrays.toString(a.shape()));
//            assertArrayEquals(e.shape(), a.shape());

            assertEquals("Failed at iteration [" + i + "]", e, a);
        }
    }

    private static void testParallelInferenceMulti(ParallelInference inf, List<INDArray[]> in, List<INDArray[]> inMasks, List<INDArray[]> exp) throws Exception {
        final INDArray[][] act = new INDArray[in.size()][0];
        final AtomicInteger counter = new AtomicInteger(0);
        final AtomicInteger failedCount = new AtomicInteger(0);

        for( int i=0; i<in.size(); i++ ){
            final int j=i;
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try{
                        INDArray[] inMask = (inMasks == null ? null : inMasks.get(j));
                        act[j] = inf.output(in.get(j), inMask);
                        counter.incrementAndGet();
                    } catch (Exception e){
                        e.printStackTrace();
                        failedCount.incrementAndGet();
                    }
                }
            }).start();
        }

        long start = System.currentTimeMillis();
        long current = System.currentTimeMillis();
        while(current < start + 20000 && failedCount.get() == 0 && counter.get() < in.size()){
            Thread.sleep(1000L);
        }

        assertEquals(0, failedCount.get());
        assertEquals(in.size(), counter.get());
        for( int i=0; i<in.size(); i++ ){
            INDArray[] e = exp.get(i);
            INDArray[] a = act[i];

            assertArrayEquals(e, a);
        }
    }
}
