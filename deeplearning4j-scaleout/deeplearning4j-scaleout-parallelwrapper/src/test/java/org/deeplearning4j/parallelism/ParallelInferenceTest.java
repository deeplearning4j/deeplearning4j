package org.deeplearning4j.parallelism;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
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

        INDArray[] input = observable1.getInput();

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

        INDArray[] input = observable1.getInput();

        assertEquals(1.0f, input[0].tensorAlongDimension(0, 1).meanNumber().floatValue(), 0.001);
        assertEquals(2.0f, input[0].tensorAlongDimension(1, 1).meanNumber().floatValue(), 0.001);

        input = observable3.getInput();
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

        observable3.setOutput(bigOutput);
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


    @Test
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

                final INDArray[] act = new INDArray[arrs.size()];
                final AtomicInteger counter = new AtomicInteger(0);
                final AtomicInteger failedCount = new AtomicInteger(0);

                for( int i=0; i<arrs.size(); i++ ){
                    final int j=i;
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            try{
                                act[j] = inf.output(arrs.get(j));
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
                while(current < start + 20000 && failedCount.get() == 0 && counter.get() < arrs.size()){
                    Thread.sleep(1000L);
                }

                assertEquals(0, failedCount.get());
                assertEquals(arrs.size(), counter.get());
                for( int i=0; i<arrs.size(); i++ ){
                    INDArray e = exp.get(i);
                    INDArray a = act[i];

                    assertEquals(e, a);
                }
            }
        }
    }
}
