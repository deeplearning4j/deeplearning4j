/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.parameterserver.distributed.v2.chunks.impl;

import lombok.extern.slf4j.Slf4j;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.v2.chunks.VoidChunk;
import org.nd4j.parameterserver.distributed.v2.messages.impl.GradientsUpdateMessage;
import org.nd4j.parameterserver.distributed.v2.util.MessageSplitter;

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@Disabled
@Tag(TagNames.FILE_IO)
@Tag(TagNames.DIST_SYSTEMS)
@NativeTag
public class FileChunksTrackerTest extends BaseND4JTest {
    @Test
    public void testTracker_1() throws Exception {
        var array = Nd4j.linspace(1, 100000, 100000).reshape(-1, 1000);
        var splitter = MessageSplitter.getInstance();

        var message = new GradientsUpdateMessage("123", array);
        var messages = new ArrayList<>(splitter.split(message, 16384));

        var tracker = new FileChunksTracker<GradientsUpdateMessage>(messages.get(0));

        assertFalse(tracker.isComplete());

        for (var m:messages)
            tracker.append(m);

        assertTrue(tracker.isComplete());

        var des = tracker.getMessage();
        assertNotNull(des);

        var restored = des.getPayload();
        assertNotNull(restored);

        assertEquals(array, restored);
    }

    @Test
    public void testDoubleSpending_1() throws Exception {
        var array = Nd4j.linspace(1, 100000, 100000).reshape(-1, 1000);
        var splitter = MessageSplitter.getInstance();

        var message = new GradientsUpdateMessage("123", array);
        var messages = new ArrayList<VoidChunk>(splitter.split(message, 16384));

        var tracker = new FileChunksTracker<GradientsUpdateMessage>(messages.get(0));

        assertFalse(tracker.isComplete());

        for (var m:messages)
            tracker.append(m);

        assertTrue(tracker.isComplete());

        var des = tracker.getMessage();
        assertNotNull(des);

        for (var m:messages)
            tracker.append(m);
    }
}