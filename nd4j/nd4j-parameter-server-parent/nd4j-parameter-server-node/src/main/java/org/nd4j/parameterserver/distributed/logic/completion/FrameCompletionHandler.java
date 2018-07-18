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

package org.nd4j.parameterserver.distributed.logic.completion;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class FrameCompletionHandler {

    private Map<RequestDescriptor, FrameDescriptor> frames = new ConcurrentHashMap<>();

    public boolean isTrackingFrame(RequestDescriptor descriptor) {
        return frames.containsKey(descriptor);
    }

    public boolean isTrackingFrame(long originatorId, long frameId) {
        return frames.containsKey(RequestDescriptor.createDescriptor(originatorId, frameId));
    }

    /**
     *
     *
     * @param originatorId
     * @param frameId
     * @param messageId
     */
    public void addHook(Long originatorId, Long frameId, Long messageId) {
        RequestDescriptor descriptor = RequestDescriptor.createDescriptor(originatorId, frameId);
        if (!frames.containsKey(descriptor))
            frames.put(descriptor, new FrameDescriptor(originatorId));

        frames.get(descriptor).addMessage(messageId);
    }

    public void notifyFrame(RequestDescriptor descriptor, Long messageId) {
        FrameDescriptor frameDescriptor = frames.get(descriptor);

        if (frameDescriptor != null)
            frameDescriptor.finishedMessage(messageId);
    }

    public void notifyFrame(Long originatorId, Long frameId, Long messageId) {
        notifyFrame(RequestDescriptor.createDescriptor(originatorId, frameId), messageId);
    }

    public boolean isCompleted(RequestDescriptor descriptor) {
        if (isTrackingFrame(descriptor)) {
            // FIXME: double spending possible here
            FrameDescriptor frameDescriptor = frames.get(descriptor);
            if (frameDescriptor == null)
                return false;

            return frameDescriptor.isFinished();
        } else {
            log.warn("DOUBLE SPENDING!!!");
            return false;
        }
    }

    public boolean isCompleted(long originatorId, long frameId) {
        RequestDescriptor descriptor = RequestDescriptor.createDescriptor(originatorId, frameId);
        return isCompleted(descriptor);
    }

    public FrameDescriptor getCompletedFrameInfo(RequestDescriptor descriptor) {
        try {
            return frames.get(descriptor);
        } finally {
            frames.remove(descriptor);
        }
    }

    public FrameDescriptor getCompletedFrameInfo(long originatorId, long frameId) {
        RequestDescriptor descriptor = RequestDescriptor.createDescriptor(originatorId, frameId);
        return getCompletedFrameInfo(descriptor);
    }


    public static class FrameDescriptor {

        @Getter
        private long frameOriginatorId;

        // messageId within frame, and it's state
        private Map<Long, AtomicBoolean> states = new ConcurrentHashMap<>();
        private AtomicInteger messages = new AtomicInteger(0);
        private AtomicInteger finished = new AtomicInteger(0);


        public FrameDescriptor(long frameOriginatorId) {
            this.frameOriginatorId = frameOriginatorId;
        }

        public boolean isFinished() {
            return messages.get() == finished.get();
        }

        public void addMessage(Long messageId) {
            states.put(messageId, new AtomicBoolean(false));
            messages.incrementAndGet();
        }

        public void finishedMessage(Long messageId) {
            AtomicBoolean boo = states.get(messageId);
            if (boo != null)
                boo.set(true);

            finished.incrementAndGet();
        }

        public int getIncompleteNumber() {
            return messages.get() - finished.get();
        }
    }
}
