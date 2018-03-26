package org.deeplearning4j.parallelism;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
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

    @Test
    public void testInferenceSequential1() throws Exception {
        ParallelInference inf =
                        new ParallelInference.Builder(model).inferenceMode(InferenceMode.SEQUENTIAL).workers(2).build();



        log.info("Features shape: {}",
                        Arrays.toString(iterator.next().getFeatureMatrix().shapeInfoDataBuffer().asInt()));

        INDArray array1 = inf.output(iterator.next().getFeatureMatrix());
        INDArray array2 = inf.output(iterator.next().getFeatureMatrix());

        assertFalse(array1.isAttached());
        assertFalse(array2.isAttached());

        INDArray array3 = inf.output(iterator.next().getFeatureMatrix());
        assertFalse(array3.isAttached());

        iterator.reset();

        evalClassifcationSingleThread(inf, iterator);

        // both workers threads should have non-zero
        assertTrue(inf.getWorkerCounter(0) > 100L);
        assertTrue(inf.getWorkerCounter(1) > 100L);
    }

    @Test
    public void testInferenceSequential2() throws Exception {
        ParallelInference inf =
                        new ParallelInference.Builder(model).inferenceMode(InferenceMode.SEQUENTIAL).workers(2).build();



        log.info("Features shape: {}",
                        Arrays.toString(iterator.next().getFeatureMatrix().shapeInfoDataBuffer().asInt()));

        INDArray array1 = inf.output(iterator.next().getFeatureMatrix());
        INDArray array2 = inf.output(iterator.next().getFeatureMatrix());

        assertFalse(array1.isAttached());
        assertFalse(array2.isAttached());

        INDArray array3 = inf.output(iterator.next().getFeatureMatrix());
        assertFalse(array3.isAttached());

        iterator.reset();

        evalClassifcationMultipleThreads(inf, iterator, 10);

        // both workers threads should have non-zero
        assertTrue(inf.getWorkerCounter(0) > 100L);
        assertTrue(inf.getWorkerCounter(1) > 100L);
    }


    @Test
    public void testInferenceBatched1() throws Exception {
        ParallelInference inf = new ParallelInference.Builder(model).inferenceMode(InferenceMode.BATCHED).batchLimit(8)
                        .workers(2).build();



        log.info("Features shape: {}",
                        Arrays.toString(iterator.next().getFeatureMatrix().shapeInfoDataBuffer().asInt()));

        INDArray array1 = inf.output(iterator.next().getFeatureMatrix());
        INDArray array2 = inf.output(iterator.next().getFeatureMatrix());

        assertFalse(array1.isAttached());
        assertFalse(array2.isAttached());

        INDArray array3 = inf.output(iterator.next().getFeatureMatrix());
        assertFalse(array3.isAttached());

        iterator.reset();

        evalClassifcationMultipleThreads(inf, iterator, 20);

        // both workers threads should have non-zero
        assertTrue(inf.getWorkerCounter(0) > 10L);
        assertTrue(inf.getWorkerCounter(1) > 10L);
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

        List<INDArray[]> l = observable1.getInputBatches();
        assertEquals(1, l.size());
        INDArray[] input = l.get(0);

        assertEquals(1, input.length);
        assertArrayEquals(new int[] {2, 100}, input[0].shape());
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

        List<INDArray[]> l = observable1.getInputBatches();
        assertEquals(1, l.size());
        INDArray[] input = l.get(0);

        assertEquals(1.0f, input[0].tensorAlongDimension(0, 1).meanNumber().floatValue(), 0.001);
        assertEquals(2.0f, input[0].tensorAlongDimension(1, 1).meanNumber().floatValue(), 0.001);


        l = observable3.getInputBatches();
        assertEquals(1, l.size());
        input = l.get(0);
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
        assertArrayEquals(new int[] {1, 10}, out.shape());
        assertEquals(0.0f, out.meanNumber().floatValue(), 0.01f);

        observable3.setPosition(1);
        out = observable3.getOutput()[0];
        assertArrayEquals(new int[] {1, 10}, out.shape());
        assertEquals(1.0f, out.meanNumber().floatValue(), 0.01f);

        observable3.setPosition(2);
        out = observable3.getOutput()[0];
        assertArrayEquals(new int[] {1, 10}, out.shape());
        assertEquals(2.0f, out.meanNumber().floatValue(), 0.01f);
    }


    protected void evalClassifcationSingleThread(@NonNull ParallelInference inf, @NonNull DataSetIterator iterator) {
        DataSet ds = iterator.next();
        log.info("NumColumns: {}", ds.getLabels().columns());
        iterator.reset();
        Evaluation eval = new Evaluation(ds.getLabels().columns());
        while (iterator.hasNext()) {
            ds = iterator.next();
            INDArray output = inf.output(ds.getFeatureMatrix());
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
            }
        }
    }

    private static void testParallelInference(ParallelInference inf, List<INDArray> in, List<INDArray> exp) throws Exception {
        final INDArray[] act = new INDArray[in.size()];
        final AtomicInteger counter = new AtomicInteger(0);
        final AtomicInteger failedCount = new AtomicInteger(0);

        for( int i=0; i<in.size(); i++ ){
            final int j=i;
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try{
                        act[j] = inf.output(in.get(j));
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
            INDArray e = exp.get(i);
            INDArray a = act[i];

//            float[] fe = e.dup().data().asFloat();
//            float[] fa = a.dup().data().asFloat();
//            System.out.println(Arrays.toString(fe));
//            System.out.println(Arrays.toString(fa));
//            assertArrayEquals(fe, fa, 1e-8f);
//            System.out.println(Arrays.toString(e.shape()) + " vs " + Arrays.toString(a.shape()));
//            assertArrayEquals(e.shape(), a.shape());

            assertEquals(e, a);
        }
    }
}
