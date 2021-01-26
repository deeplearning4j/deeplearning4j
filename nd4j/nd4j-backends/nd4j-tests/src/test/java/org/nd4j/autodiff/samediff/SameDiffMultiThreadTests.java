package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.imports.tfgraphs.TFGraphTestZooModels;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.resources.Resources;

import java.io.File;
import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

@Slf4j
public class SameDiffMultiThreadTests extends BaseND4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Override
    public long getTimeoutMilliseconds() {
        return 60000L;
    }

    @Test
    public void testSimple() throws Exception {

        int nThreads = 4;
        int nRuns = 1000;

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 10);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 10);

        SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, 10, 10));
        SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, 10));
        SDVariable w2 = sd.var("w2", Nd4j.rand(DataType.FLOAT, 10, 10));
        SDVariable b2 = sd.var("b2", Nd4j.rand(DataType.FLOAT, 10));
        SDVariable w3 = sd.var("w3", Nd4j.rand(DataType.FLOAT, 10, 10));
        SDVariable b3 = sd.var("b3", Nd4j.rand(DataType.FLOAT, 10));

        SDVariable l1 = sd.nn.tanh(in.mmul(w1).add(b1));
        SDVariable l2 = sd.nn.sigmoid(l1.mmul(w2).add(b2));
        SDVariable l3 = sd.nn.softmax("out", l2.mmul(w3).add(b3));

        SDVariable loss = sd.loss.logLoss("loss", label, l3);

        INDArray[] inputArrs = new INDArray[nThreads];
        INDArray[] expOut = new INDArray[nThreads];
        for( int i=0; i<nThreads; i++ ){
            inputArrs[i] = Nd4j.rand(DataType.FLOAT, i+1, 10);
            expOut[i] = sd.outputSingle(Collections.singletonMap("in", inputArrs[i]), "out");
        }

        Semaphore s = new Semaphore(nThreads);
        CountDownLatch latch = new CountDownLatch(nThreads);

        AtomicBoolean[] failuresByThread = new AtomicBoolean[nThreads];
        AtomicInteger[] counters = new AtomicInteger[nThreads];
        doTest(sd, nThreads, nRuns, inputArrs, expOut, "in", "out", failuresByThread, counters, s, latch);

        s.release(nThreads);
        latch.await();

        for(int i=0; i<nThreads; i++ ){
            assertFalse("Thread " + i + " failed", failuresByThread[i].get());
        }

        for(int i=0; i<nThreads; i++ ){
            assertEquals("Thread " + i + " number of runs", nRuns, counters[i].get());
        }
    }

    @Test
    @Ignore //2020/03/24 AB - https://github.com/eclipse/deeplearning4j/issues/8802
    public void testMobilenet() throws Exception {
        TFGraphTestZooModels.currentTestDir = testDir.newFolder();
        File f = Resources.asFile("tf_graphs/zoo_models/mobilenet_v2_1.0_224/tf_model.txt");
        SameDiff sd = TFGraphTestZooModels.LOADER.apply(f, "mobilenet_v2_1.0_224");
//        System.out.println(sd.summary());

        int nThreads = 4;
        int nRuns = 30;
        INDArray[] inputArrs = new INDArray[nThreads];
        INDArray[] expOut = new INDArray[nThreads];
        for( int i=0; i<nThreads; i++ ){
            if(i == 0 || i > 2)
                inputArrs[i] = Nd4j.rand(DataType.FLOAT, 1, 224, 224, 3);
            else if(i == 1)
                inputArrs[i] = Nd4j.zeros(DataType.FLOAT, 1, 224, 224, 3);
            else if(i == 2)
                inputArrs[i] = Nd4j.ones(DataType.FLOAT, 1, 224, 224, 3);

            expOut[i] = sd.outputSingle(Collections.singletonMap("input", inputArrs[i]), "MobilenetV2/Predictions/Reshape_1");
            Nd4j.getExecutioner().commit();
        }

        AtomicBoolean[] failuresByThread = new AtomicBoolean[nThreads];
        AtomicInteger[] counters = new AtomicInteger[nThreads];
        Semaphore s = new Semaphore(nThreads);
        CountDownLatch latch = new CountDownLatch(nThreads);

        doTest(sd, nThreads, nRuns, inputArrs, expOut, "input", "MobilenetV2/Predictions/Reshape_1", failuresByThread, counters, s, latch);

        s.release(nThreads);
        latch.await();

        for(int i=0; i<nThreads; i++ ){
            assertFalse("Thread " + i + " failed", failuresByThread[i].get());
        }

        for(int i=0; i<nThreads; i++ ){
            assertEquals("Thread " + i + " number of runs", nRuns, counters[i].get());
        }
    }


    public static void doTest(SameDiff sd, int nThreads, int nRuns, INDArray[] inputArrs, INDArray[] expOut,
                              String inName, String outName,
                              AtomicBoolean[] failuresByThread, AtomicInteger[] counters, Semaphore s, CountDownLatch latch){

        for( int i=0; i<nThreads; i++ ){
            failuresByThread[i] = new AtomicBoolean(false);
            counters[i] = new AtomicInteger(0);
            final int j=i;
            Thread t = new Thread(new Runnable() {
                @Override
                public void run() {
                    try{
                        s.acquire(1);
                        for( int i=0; i<nRuns; i++ ){
                            INDArray out = sd.outputSingle(Collections.singletonMap(inName, inputArrs[j]), outName);
                            Nd4j.getExecutioner().commit();
                            INDArray exp = expOut[j];

                            if(!exp.equals(out)){
                                failuresByThread[j].set(true);
                                log.error("Failure in thread: {}/{} - iteration {}\nExpected ={}\nActual={}", Thread.currentThread().getId(), j, i, exp, out);
                                break;
                            }

                            if(out.closeable())
                                out.close();

//                            if(i % 100 == 0){
//                                log.info("Thread {} at {}", Thread.currentThread().getId(), i);
//                            }
                            counters[j].addAndGet(1);
                        }
                    } catch (Throwable t){
                        log.error("Error in thread: {}", Thread.currentThread().getId(), t);
                    } finally {
                        latch.countDown();
                    }
                }
            });
            t.start();
        }
    }
}
