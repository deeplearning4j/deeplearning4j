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

package org.nd4j.aeron.ipc.chunk;

import org.junit.Test;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 11/20/16.
 */
public class ChunkAccumulatorTests {

    @Test
    public void testAccumulator() {
        ChunkAccumulator chunkAccumulator = new InMemoryChunkAccumulator();
        NDArrayMessage message = NDArrayMessage.wholeArrayUpdate(Nd4j.ones(1000));
        int chunkSize = 128;
        NDArrayMessageChunk[] chunks = NDArrayMessage.chunks(message, chunkSize);
        for (int i = 0; i < chunks.length; i++) {
            chunkAccumulator.accumulateChunk(chunks[i]);
        }

        NDArrayMessage message1 = chunkAccumulator.reassemble(chunks[0].getId());
        assertEquals(message, message1);
    }

}
