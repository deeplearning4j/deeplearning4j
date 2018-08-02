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

package org.nd4j.jita.concurrency;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaAffinityManagerTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void getDeviceForCurrentThread() throws Exception {
        CudaAffinityManager manager = new CudaAffinityManager();

        Integer deviceId = manager.getDeviceForCurrentThread();

        assertEquals(0, deviceId.intValue());

        manager.attachThreadToDevice(Thread.currentThread().getId(), 1);

        assertEquals(1, manager.getDeviceForCurrentThread().intValue());

        manager.attachThreadToDevice(Thread.currentThread().getId(), 0);

        assertEquals(0, manager.getDeviceForCurrentThread().intValue());
    }

    @Test
    public void getDeviceForAnotherThread() throws Exception {
        CudaAffinityManager manager = new CudaAffinityManager();

        Integer deviceId = manager.getDeviceForCurrentThread();

        assertEquals(0, deviceId.intValue());

        manager.attachThreadToDevice(1731L, 0);

        assertEquals(0, manager.getDeviceForThread(1731L).intValue());
    }

    @Test
    public void getDeviceForAnotherThread2() throws Exception {
        CudaAffinityManager manager = new CudaAffinityManager();

        Integer deviceId = manager.getDeviceForCurrentThread();

        assertEquals(0, deviceId.intValue());

        System.out.println("Current threadId: " + Thread.currentThread().getId());

        Thread thread = new Thread();

        long threadIdPrior = thread.getId();

        System.out.println("Next threadId: " + thread.getId());
        assertNotEquals(Thread.currentThread().getId(), thread.getId());

        thread.start();
        System.out.println("Current threadId: " + thread.getId());

        assertEquals(threadIdPrior, thread.getId());
    }

    /**
     * This is special test for multi-threaded environment
     * @throws Exception
     */
    @Test
    public void getDeviceForAnotherThread3() throws Exception {
        final int limit = 10;
        final CudaAffinityManager manager = new CudaAffinityManager();

        final Thread threads[] = new Thread[limit];
        final AtomicBoolean[] results = new AtomicBoolean[limit];

        for (int cnt = 0; cnt < limit; cnt++) {
            final int c = cnt;
            results[cnt] = new AtomicBoolean(false);
            threads[cnt] = new Thread(new Runnable() {
                @Override
                public void run() {
                    assertEquals(0, manager.getDeviceForCurrentThread().intValue());
                    results[c].set(true);
                }
            });

            manager.attachThreadToDevice(threads[cnt], 0);

            threads[cnt].start();
        }

        for (int cnt = 0; cnt < limit; cnt++) {
            threads[cnt].join();
            assertTrue("Failed for thread ["+ cnt+"]", results[cnt].get());
        }
    }

    /**
     * This is special test for multi-threaded multi-gpu environment
     * @throws Exception
     */
    @Test
    public void getDeviceForAnotherThread4() throws Exception {
        final int limit = 10;
        final CudaAffinityManager manager = new CudaAffinityManager();

        final Thread threads[] = new Thread[limit];
        final AtomicBoolean[] results = new AtomicBoolean[limit];
        final int cards[] = new int[limit];

        for (int cnt = 0; cnt < limit; cnt++) {
            final int c = cnt;
            results[cnt] = new AtomicBoolean(false);

            threads[cnt] = new Thread(new Runnable() {
                @Override
                public void run() {
                    // this is pseudo-master thread

                    final int deviceId = manager.getDeviceForCurrentThread();

                    Thread thread = new Thread(new Runnable() {
                        @Override
                        public void run() {
                            int cdev = manager.getDeviceForCurrentThread();
                            assertEquals(deviceId, cdev);
                            results[c].set(true);
                            cards[c] = cdev;
                        }
                    });

                    manager.attachThreadToDevice(thread, deviceId);

                    thread.start();
                    try {
                        thread.join();
                    } catch (Exception e) {
                        ;
                    }
                }
            });

            threads[cnt].start();
        }

        for (int cnt = 0; cnt < limit; cnt++) {
            threads[cnt].join();
            assertTrue("Failed for thread ["+ cnt+"]", results[cnt].get());
        }


        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int c = 0; c < numDevices; c++) {
            assertTrue("Failed to find device ["+ c +"] in used devices", ArrayUtils.contains(cards, c));
        }
    }

}