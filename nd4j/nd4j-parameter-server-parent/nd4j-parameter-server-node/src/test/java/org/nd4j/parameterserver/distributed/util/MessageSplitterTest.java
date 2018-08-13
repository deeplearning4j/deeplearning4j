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

package org.nd4j.parameterserver.distributed.util;

import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Optional;
import org.nd4j.parameterserver.distributed.messages.v2.GradientsUpdateMessage;

import static org.junit.Assert.*;

public class MessageSplitterTest {

    @Test
    public void testMessageSplit_1() throws Exception {
        val array = Nd4j.linspace(1, 100000, 100000).reshape(-1, 1000);
        val splitter = new MessageSplitter();

        val message = new GradientsUpdateMessage("123", array);

        val messages = splitter.split(message, 1024);

        assertNotNull(messages);
        assertFalse(messages.isEmpty());

        for (val m:messages)
            assertEquals("123", m.getOriginalId());

        Optional<GradientsUpdateMessage> dec = null;
        for (val m:messages)
            dec = splitter.merge(m);

        assertTrue(dec.isPresent());
    }
}