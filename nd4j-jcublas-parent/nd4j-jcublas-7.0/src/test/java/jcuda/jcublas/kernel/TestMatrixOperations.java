/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package jcuda.jcublas.kernel;

import static org.junit.Assert.assertEquals;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.time.StopWatch;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.executors.ExecutorServiceProvider;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.ops.transforms.Transforms;

public class TestMatrixOperations {


    @Test
    public void testDot() {
        INDArray four = Nd4j.linspace(1,4,4);
        double dot = Nd4j.getBlasWrapper().dot(four,four);
        assertEquals(30,dot,1e-1);
    }

    @Test
    public void testMean() {
        INDArray mean2 = Nd4j.linspace(1, 5, 5);
        assertEquals(3,mean2.meanNumber().doubleValue(),1e-1);
    }


    @Test
    public void testElementWiseOp() {
        Transforms.sigmoid(Nd4j.ones(5,5));
    }

    @Test
    public void testMultipleThreads() throws InterruptedException {
        int numThreads = 10;
        final INDArray array = Nd4j.rand(300, 300);
        final INDArray expected = array.dup().mmul(array).mmul(array).div(array).div(array);
        final AtomicInteger correct = new AtomicInteger();
        final CountDownLatch latch = new CountDownLatch(numThreads);
        System.out.println("Running on " + ContextHolder.getInstance().deviceNum());
        ExecutorService executors = ExecutorServiceProvider.getExecutorService();

        for(int x = 0; x< numThreads; x++) {
            executors.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        int total = 10;
                        int right = 0;
                        for(int x = 0; x< total; x++) {
                            StopWatch watch = new StopWatch();
                            watch.start();
                            INDArray actual = array.dup().mmul(array).mmul(array).div(array).div(array);
                            watch.stop();
                            System.out.println("MMUL took " + watch.getTime());
                            if(expected.equals(actual)) right++;
                        }

                        if(total == right)
                            correct.incrementAndGet();
                    } finally {
                        latch.countDown();
                    }

                }
            });
        }

        latch.await();

        assertEquals(numThreads, correct.get());

    }

}
