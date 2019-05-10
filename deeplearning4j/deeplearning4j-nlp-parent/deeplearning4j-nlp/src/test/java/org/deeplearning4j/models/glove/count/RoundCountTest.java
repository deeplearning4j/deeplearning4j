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

package org.deeplearning4j.models.glove.count;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by fartovii on 23.12.15.
 */
public class RoundCountTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testGet1() throws Exception {
        RoundCount count = new RoundCount(1);

        assertEquals(0, count.get());

        count.tick();
        assertEquals(1, count.get());

        count.tick();
        assertEquals(0, count.get());
    }

    @Test
    public void testGet2() throws Exception {
        RoundCount count = new RoundCount(3);

        assertEquals(0, count.get());

        count.tick();
        assertEquals(1, count.get());

        count.tick();
        assertEquals(2, count.get());

        count.tick();
        assertEquals(3, count.get());

        count.tick();
        assertEquals(0, count.get());
    }

    @Test
    public void testPrevious1() throws Exception {
        RoundCount count = new RoundCount(3);

        assertEquals(0, count.get());
        assertEquals(3, count.previous());

        count.tick();
        assertEquals(1, count.get());
        assertEquals(0, count.previous());

        count.tick();
        assertEquals(2, count.get());
        assertEquals(1, count.previous());

        count.tick();
        assertEquals(3, count.get());
        assertEquals(2, count.previous());

        count.tick();
        assertEquals(0, count.get());
        assertEquals(3, count.previous());
    }
}
