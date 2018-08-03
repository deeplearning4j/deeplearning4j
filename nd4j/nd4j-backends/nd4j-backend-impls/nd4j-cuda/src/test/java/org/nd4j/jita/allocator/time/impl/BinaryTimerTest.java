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
public class BinaryTimerTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testIsAlive1() throws Exception {
        BinaryTimer timer = new BinaryTimer(2, TimeUnit.SECONDS);
        timer.triggerEvent();

        assertTrue(timer.isAlive());
    }

    @Test
    public void testIsAlive2() throws Exception {
        BinaryTimer timer = new BinaryTimer(2, TimeUnit.SECONDS);
        timer.triggerEvent();

        Thread.sleep(3000);

        assertFalse(timer.isAlive());
    }
}