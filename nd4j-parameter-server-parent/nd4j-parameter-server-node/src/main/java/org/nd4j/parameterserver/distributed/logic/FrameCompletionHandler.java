package org.nd4j.parameterserver.distributed.logic;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.util.Pair;

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

    private Map<Long, FrameDescriptor> frames = new ConcurrentHashMap<>();

    public boolean isTrackingFrame(long frameId) {
        return frames.containsKey(frameId);
    }

    public void addHook(Long originatorId, Long frameId, Long messageId) {
        if (!frames.containsKey(frameId))
            frames.put(frameId, new FrameDescriptor(originatorId));

        frames.get(frameId).addMessage(messageId);
    }

    public void notifyFrame(Long originatorId, Long frameId, Long messageId) {

        frames.get(frameId).finishedMessage(messageId);
    }

    public boolean isCompleted(Long frameId) {
        if (isTrackingFrame(frameId))
            return frames.get(frameId).isFinished();
        else return false;
    }

    public int getIncompleteTasksNumber(Long frameId) {
        return frames.get(frameId).getIncompleteNumber();
    }

    public FrameDescriptor getCompletedFrameInfo(Long frameId) {
        try {
            return frames.get(frameId);
        } finally {
            frames.remove(frameId);
        }
    }


    public static class FrameDescriptor {

        @Getter private long frameOriginatorId;

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
            states.get(messageId).set(true);
            finished.incrementAndGet();
        }

        public int getIncompleteNumber() {
            return messages.get() - finished.get();
        }
    }
}
