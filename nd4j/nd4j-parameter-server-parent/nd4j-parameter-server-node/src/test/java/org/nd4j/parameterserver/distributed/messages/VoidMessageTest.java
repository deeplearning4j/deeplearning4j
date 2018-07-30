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

package org.nd4j.parameterserver.distributed.messages;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class VoidMessageTest {
    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Test(timeout = 30000L)
    public void testSerDe1() throws Exception {
        SkipGramRequestMessage message = new SkipGramRequestMessage(10, 12, new int[] {10, 20, 30, 40},
                        new byte[] {(byte) 0, (byte) 0, (byte) 1, (byte) 0}, (short) 0, 0.0, 117L);

        byte[] bytes = message.asBytes();

        SkipGramRequestMessage restored = (SkipGramRequestMessage) VoidMessage.fromBytes(bytes);

        assertNotEquals(null, restored);

        assertEquals(message, restored);
        assertArrayEquals(message.getPoints(), restored.getPoints());
        assertArrayEquals(message.getCodes(), restored.getCodes());
    }

}
