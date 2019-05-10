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

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.optimize.solvers.accumulation.FancyBlockingQueue;
import org.junit.Test;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class FancyBlockingQueueTests extends BaseDL4JTest {

    @Test
    public void testFancyQueue1() throws Exception {
        final FancyBlockingQueue<Integer> queue = new FancyBlockingQueue<>(new LinkedBlockingQueue<Integer>(512), 4);
        long f = 0;
        for (int x = 0; x < 512; x++) {
            queue.add(x);
            f += x;
        }

        assertEquals(512, queue.size());


        final AtomicLong e = new AtomicLong(0);

        queue.registerConsumers(4);
        Thread[] threads = new Thread[4];
        for (int x = 0; x < 4; x++) {
            final int t = x;
            threads[x] = new Thread(new Runnable() {
                @Override
                public void run() {
                    while (!queue.isEmpty()) {
                        Integer i = queue.poll();
                        //log.info("i: {}", i);
                        e.addAndGet(i);
                    }
                }
            });

            threads[x].start();
        }


        for (int x = 0; x < 4; x++) {
            threads[x].join();
        }

        assertEquals(f * 4, e.get());
    }

    /**
     * This test is +- the same as the first one, just adds variable consumption time
     *
     * @throws Exception
     */
    @Test
    public void testFancyQueue2() throws Exception {
        final FancyBlockingQueue<Integer> queue = new FancyBlockingQueue<>(new LinkedBlockingQueue<Integer>(512), 4);
        long f = 0;
        for (int x = 0; x < 512; x++) {
            queue.add(x);
            f += x;
        }

        assertEquals(512, queue.size());


        final AtomicLong e = new AtomicLong(0);
        queue.registerConsumers(4);

        Thread[] threads = new Thread[4];
        for (int x = 0; x < 4; x++) {
            final int t = x;
            threads[x] = new Thread(new Runnable() {
                @Override
                public void run() {
                    while (!queue.isEmpty()) {
                        Integer i = queue.poll();
                        e.addAndGet(i);

                        try {
                            Thread.sleep(RandomUtils.nextInt(1, 5));
                        } catch (Exception e) {
                            //
                        }
                    }
                }
            });

            threads[x].start();
        }


        for (int x = 0; x < 4; x++) {
            threads[x].join();
        }

        assertEquals(f * 4, e.get());
    }


    /**
     * This test checks for compatibility with single producer - single consumer model
     * @throws Exception
     */
    @Test
    public void testFancyQueue3() throws Exception {
        final FancyBlockingQueue<Integer> queue = new FancyBlockingQueue<>(new LinkedBlockingQueue<Integer>(512), 4);
        long f = 0;
        for (int x = 0; x < 512; x++) {
            queue.add(x);
            f += x;
        }

        assertEquals(512, queue.size());


        final AtomicLong e = new AtomicLong(0);
        queue.registerConsumers(1);
        Thread[] threads = new Thread[1];
        for (int x = 0; x < 1; x++) {
            final int t = x;
            threads[x] = new Thread(new Runnable() {
                @Override
                public void run() {
                    while (!queue.isEmpty()) {
                        Integer i = queue.poll();
                        //log.info("i: {}", i);
                        e.addAndGet(i);
                    }
                }
            });

            threads[x].start();
        }


        for (int x = 0; x < 1; x++) {
            threads[x].join();
        }

        assertEquals(f, e.get());
    }

    /**
     * This test checks for compatibility with single producer - single consumer model
     * @throws Exception
     */
    @Test
    public void testFancyQueue4() throws Exception {
        final FancyBlockingQueue<Integer> queue = new FancyBlockingQueue<>(new LinkedBlockingQueue<Integer>(512), 4);
        long f = 0;
        for (int x = 0; x < 512; x++) {
            queue.add(x);
            f += x;
        }

        assertEquals(512, queue.size());


        final AtomicLong e = new AtomicLong(0);
        queue.fallbackToSingleConsumerMode(true);
        Thread[] threads = new Thread[1];
        for (int x = 0; x < 1; x++) {
            final int t = x;
            threads[x] = new Thread(new Runnable() {
                @Override
                public void run() {
                    while (!queue.isEmpty()) {
                        Integer i = queue.poll();
                        //log.info("i: {}", i);
                        e.addAndGet(i);
                    }
                }
            });

            threads[x].start();
        }


        for (int x = 0; x < 1; x++) {
            threads[x].join();
        }

        assertEquals(f, e.get());
    }
}
