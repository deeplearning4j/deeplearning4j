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

package org.nd4j.aeron.util;


import java.nio.ByteBuffer;

/**
 * Minor {@link ByteBuffer} utils
 *
 * @author Adam Gibson
 */
public class BufferUtil {
    /**
     * Merge all byte buffers together
     * @param buffers the bytebuffers to merge
     * @param overAllCapacity the capacity of the
     *                        merged bytebuffer
     * @return the merged byte buffer
     *
     */
    public static ByteBuffer concat(ByteBuffer[] buffers, int overAllCapacity) {
        ByteBuffer all = ByteBuffer.allocateDirect(overAllCapacity);
        for (int i = 0; i < buffers.length; i++) {
            ByteBuffer curr = buffers[i].slice();
            all.put(curr);
        }

        all.rewind();
        return all;
    }

    /**
     * Merge all bytebuffers together
     * @param buffers the bytebuffers to merge
     * @return the merged bytebuffer
     */
    public static ByteBuffer concat(ByteBuffer[] buffers) {
        int overAllCapacity = 0;
        for (int i = 0; i < buffers.length; i++)
            overAllCapacity += buffers[i].limit() - buffers[i].position();
        //padding
        overAllCapacity += buffers[0].limit() - buffers[0].position();
        ByteBuffer all = ByteBuffer.allocateDirect(overAllCapacity);
        for (int i = 0; i < buffers.length; i++) {
            ByteBuffer curr = buffers[i];
            all.put(curr);
        }

        all.flip();
        return all;
    }

}
