package org.nd4j.parameterserver.distributed.messages;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.Transport;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Simple wrapper for multiple request messages OF THE SAME TYPE being stacked into single message
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class Frame<T extends TrainingMessage> implements Serializable, Iterable<T>, VoidMessage {

    @Getter(AccessLevel.PROTECTED)
    @Setter(AccessLevel.PROTECTED)
    protected List<T> list = new ArrayList<T>();

    @Getter
    protected long originatorId;
    @Getter
    @Setter
    protected short targetId;
    @Getter
    @Setter
    protected long taskId;


    protected transient VoidConfiguration voidConfiguration;
    protected transient Clipboard clipboard;
    protected transient Transport transport;
    protected transient Storage storage;
    protected transient NodeRole role;
    protected transient short shardIndex;
    protected transient TrainingDriver<? extends TrainingMessage> trainer;

    @Getter
    @Setter(AccessLevel.PRIVATE)
    protected transient int retransmitCount = 0;

    protected Frame() {

    }

    public Frame(long taskId) {
        this.taskId = taskId;
    }

    public Frame(@NonNull T message) {
        this();
        list.add(message);
    }

    @Override
    public void setOriginatorId(long id) {
        this.originatorId = id;
        if (list != null)
            list.forEach((msg) -> {
                msg.setOriginatorId(this.getOriginatorId());
            });
    }

    /**
     * This method adds single TrainingMessage to this Frame
     *
     * PLEASE NOTE: This method is synchronized
     * @param message
     */
    public synchronized void stackMessage(@NonNull T message) {
        stackMessageUnlocked(message);
    }

    private void stackMessageUnlocked(@NonNull T message) {
        if (message.isJoinSupported()) {
            int index = list.indexOf(message);
            if (index >= 0)
                list.get(index).joinMessage(message);
            else {
                message.setFrameId(this.getTaskId());
                list.add(message);
            }
        } else {
            message.setFrameId(this.getTaskId());
            list.add(message);
        }
    }

    /**
     * This method adds multiple messages to this frame
     *
     * PLEASE NOTE: This method is synchronized
     * @param messages
     */
    public synchronized void stackMessages(@NonNull Collection<T> messages) {
        for (T message : messages) {
            stackMessageUnlocked(message);
        }
    }

    /**
     * This method adds multiple messages to this frame
     *
     * PLEASE NOTE: This method is synchronized
     * @param messages
     */
    public synchronized void stackMessages(T... messages) {
        for (T message : messages) {
            if (message != null)
                stackMessageUnlocked(message);
        }
    }

    public Collection<T> getMessages() {
        return list;
    }

    public int size() {
        return list.size();
    }

    @Override
    public Iterator<T> iterator() {
        return list.iterator();
    }

    @Override
    public int getMessageType() {
        return 3;
    }

    @Override
    public byte[] asBytes() {
        return SerializationUtils.serialize(this);
    }

    @Override
    public UnsafeBuffer asUnsafeBuffer() {
        return new UnsafeBuffer(asBytes());
    }

    @Override
    public void attachContext(@NonNull VoidConfiguration voidConfiguration,
                    @NonNull TrainingDriver<? extends TrainingMessage> trainer, @NonNull Clipboard clipboard,
                    @NonNull Transport transport, @NonNull Storage storage, @NonNull NodeRole role, short shardIndex) {
        this.voidConfiguration = voidConfiguration;
        this.clipboard = clipboard;
        this.transport = transport;
        this.storage = storage;
        this.role = role;
        this.shardIndex = shardIndex;
        this.trainer = trainer;
    }

    @Override
    public void extractContext(@NonNull BaseVoidMessage message) {
        this.voidConfiguration = message.voidConfiguration;
        this.clipboard = message.clipboard;
        this.transport = message.transport;
        this.storage = message.storage;
        this.role = message.role;
        this.shardIndex = message.shardIndex;
        this.trainer = message.trainer;
        this.originatorId = message.originatorId;
    }

    @Override
    public void processMessage() {
        //        log.info("Processing frame {} of {} messages... Originator: {}", this.getTaskId(), list.size(), originatorId);

        // we register all messages first
        //      if(list == null || trainer == null)
        //          return;
        if (trainer != null && transport != null)
            list.forEach((message) -> {
                trainer.addCompletionHook(getOriginatorId(), getTaskId(), message.getTaskId());
            });

        //list.parallelStream().forEach((message) -> {
        for (TrainingMessage message : list) {
            if (trainer != null && transport != null)
                message.attachContext(voidConfiguration, trainer, clipboard, transport, storage, role, shardIndex);

            // if there's more then 1 round should be applied
            for (int i = 0; i < message.getCounter(); i++) {
                //log.info("Firing message {}; originator: {}; frameId: {}; taskId: {}", message.getClass().getSimpleName(), message.getOriginatorId(), message.getFrameId(), message.getTaskId());
                message.processMessage();
            }
        } ;
    }

    @Override
    public boolean isJoinSupported() {
        return false;
    }

    @Override
    public void joinMessage(VoidMessage message) {
        // no-op
    }

    @Override
    public boolean isBlockingMessage() {
        return true;
    }

    @Override
    public void incrementRetransmitCount() {
        retransmitCount++;
    }
}
