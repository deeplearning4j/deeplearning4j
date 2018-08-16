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

package org.nd4j.parameterserver.distributed.v2.chunks.impl;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.v2.chunks.VoidChunk;
import org.nd4j.parameterserver.distributed.v2.chunks.impl.FileChunksTracker;
import org.nd4j.parameterserver.distributed.v2.messages.impl.GradientsUpdateMessage;
import org.nd4j.parameterserver.distributed.v2.util.MessageSplitter;

import java.util.ArrayList;

import static org.junit.Assert.*;

@Slf4j
public class FileChunksTrackerTest {
    @Test
    public void testTracker_1() throws Exception {
        val array = Nd4j.linspace(1, 100000, 100000).reshape(-1, 1000);
        val splitter = MessageSplitter.getInstance();

        val message = new GradientsUpdateMessage("123", array);
        val messages = new ArrayList<VoidChunk>(splitter.split(message, 16384));

        val tracker = new FileChunksTracker<GradientsUpdateMessage>(messages.get(0));

        assertFalse(tracker.isComplete());

        for (val m:messages)
            tracker.append(m);

        assertTrue(tracker.isComplete());

        val des = tracker.getMessage();
        assertNotNull(des);

        val restored = des.getPayload();
        assertNotNull(restored);

        assertEquals(array, restored);
    }
}