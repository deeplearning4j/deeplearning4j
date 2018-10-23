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

package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.concurrent.LinkedBlockingQueue;

import static org.junit.Assert.*;

@Slf4j
public class SmartFancyBlockingQueueTest {
    @Test
    public void testSFBQ_1() throws Exception {
        val queue = new SmartFancyBlockingQueue(8, Nd4j.create(5, 5));

        val array = Nd4j.create(5, 5);

        for (int e = 0; e < 6; e++) {
            queue.put(Nd4j.create(5, 5).assign(e));
        };

        assertEquals(6, queue.size());

        for (int e = 6; e < 10; e++) {
            queue.put(Nd4j.create(5, 5).assign(e));
        }

        assertEquals(1, queue.size());
    }

    @Test
    public void testSFBQ_2() throws Exception {
        final val queue = new SmartFancyBlockingQueue(8, Nd4j.create(5, 5));

        val threads = new ArrayList<Thread>();
        for (int e = 0; e< 4; e++) {
            val f = e;
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    int cnt = 0;
                    while (true) {
                        while (cnt < 1000) {
                            log.info("Thread {}: [{}]", f, cnt);
                            if (!queue.isEmpty()) {
                                val arr = queue.poll();

                                assertNotNull(arr);
                                val local = arr.unsafeDuplication(true);

                                assertEquals(cnt, local.meanNumber().intValue());
                                cnt++;
                            }
                        }
                        break;
                    }


                }
            });
            t.start();
            threads.add(t);
        }

        for (int e = 0; e < 1000; e++) {
            queue.put(Nd4j.create(5, 5).assign(e));
            queue.registerConsumers(4);
        }


        for (val t: threads)
            t.join();
    }
}