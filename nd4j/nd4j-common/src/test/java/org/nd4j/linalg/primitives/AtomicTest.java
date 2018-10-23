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

package org.nd4j.linalg.primitives;

import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.util.SerializationUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

import static org.junit.Assert.*;

public class AtomicTest {

    @Test
    public void testEquality_1() {
        val v0 = new Atomic<Integer>(1327541);
        val v1 = new Atomic<Integer>(1327541);
        val v3 = new Atomic<Integer>(1327542);

        assertEquals(v0, v1);
        assertNotEquals(v0, v3);
    }

    @Test
    public void testSerialization_1() throws Exception {
        val v0 = new Atomic<Integer>(1327541);

        try (val baos = new ByteArrayOutputStream()) {
            SerializationUtils.serialize(v0, baos);

            try (val bais = new ByteArrayInputStream(baos.toByteArray())) {
                Atomic<Integer> v1 = SerializationUtils.deserialize(bais);

                assertEquals(v1, v0);
            }
        }
    }

    @Test
    public void testCas_1() throws Exception {
        val v0 = new Atomic<String>();

        v0.cas(null, "alpha");
        assertEquals("alpha", v0.get());
    }

    @Test
    public void testCas_2() throws Exception {
        val v0 = new Atomic<String>("beta");

        v0.cas(null, "alpha");
        assertEquals("beta", v0.get());
    }
}