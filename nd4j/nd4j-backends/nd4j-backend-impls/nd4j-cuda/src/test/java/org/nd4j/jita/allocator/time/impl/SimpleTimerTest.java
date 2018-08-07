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

package org.nd4j.jita.allocator.time.impl;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class SimpleTimerTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testIsAlive1() throws Exception {
        SimpleTimer timer = new SimpleTimer(2, TimeUnit.SECONDS);
        timer.triggerEvent();

        assertTrue(timer.getNumberOfEvents() == 1);
    }

    @Test
    public void testIsAlive2() throws Exception {
        SimpleTimer timer = new SimpleTimer(2, TimeUnit.SECONDS);
        timer.triggerEvent();

        Thread.sleep(3000);

        assertEquals(0, timer.getNumberOfEvents());
    }

    @Test
    public void testIsAlive3() throws Exception {
        SimpleTimer timer = new SimpleTimer(2, TimeUnit.SECONDS);
        timer.triggerEvent();
        timer.triggerEvent();

        assertEquals(2, timer.getNumberOfEvents());
    }

    @Test
    public void testIsAlive4() throws Exception {
        SimpleTimer timer = new SimpleTimer(10, TimeUnit.SECONDS);
        timer.triggerEvent();
        timer.triggerEvent();

        Thread.sleep(1000);

        assertEquals(2, timer.getNumberOfEvents());
    }

    @Test
    public void testIsAlive5() throws Exception {
        SimpleTimer timer = new SimpleTimer(10, TimeUnit.SECONDS);
        timer.triggerEvent();
        timer.triggerEvent();

        Thread.sleep(1100);

        timer.triggerEvent();
        timer.triggerEvent();


        Thread.sleep(1100);

        timer.triggerEvent();
        timer.triggerEvent();

        assertEquals(6, timer.getNumberOfEvents());

        Thread.sleep(9000);

        assertEquals(2, timer.getNumberOfEvents());
    }

    @Test
    public void testIsAlive6() throws Exception {
        SimpleTimer timer = new SimpleTimer(20, TimeUnit.SECONDS);
        timer.triggerEvent();
        timer.triggerEvent();

        Thread.sleep(1000);

        timer.triggerEvent();
        timer.triggerEvent();


        Thread.sleep(1000);

        timer.triggerEvent();
        timer.triggerEvent();

        assertEquals(6, timer.getNumberOfEvents());

        timer.triggerEvent();
        timer.triggerEvent();

        Thread.sleep(8000);

        assertEquals(8, timer.getNumberOfEvents());
    }

    @Test
    public void testIsAlive7() throws Exception {
        SimpleTimer timer = new SimpleTimer(5, TimeUnit.SECONDS);
        timer.triggerEvent();
        timer.triggerEvent();

        Thread.sleep(1000);

        timer.triggerEvent();
        timer.triggerEvent();


        Thread.sleep(1000);

        timer.triggerEvent();
        timer.triggerEvent();

        assertEquals(6, timer.getNumberOfEvents());

        timer.triggerEvent();
        timer.triggerEvent();

        Thread.sleep(6000);

        assertEquals(0, timer.getNumberOfEvents());
    }
}