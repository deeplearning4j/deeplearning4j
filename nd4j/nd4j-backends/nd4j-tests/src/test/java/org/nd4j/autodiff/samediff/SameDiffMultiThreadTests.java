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

package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

@Slf4j
public class SameDiffMultiThreadTests extends BaseND4JTest {



    @Override
    public long getTimeoutMilliseconds() {
        return Long.MAX_VALUE;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSimple(Nd4jBackend backend) throws Exception {

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

        for(int i = 0; i < nThreads; i++) {
            assertFalse(failuresByThread[i].get(),"Thread " + i + " failed");
        }

        for(int i = 0; i < nThreads; i++) {
            assertEquals( nRuns, counters[i].get(),"Thread " + i + " number of runs");
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
