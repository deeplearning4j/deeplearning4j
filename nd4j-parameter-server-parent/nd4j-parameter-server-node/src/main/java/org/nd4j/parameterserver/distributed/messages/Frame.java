package org.nd4j.parameterserver.distributed.messages;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.Clipboard;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
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
public class Frame<T extends TrainingMessage> implements Serializable, Iterable<T> , VoidMessage {

    @Getter(AccessLevel.PROTECTED) @Setter(AccessLevel.PROTECTED)
    protected List<T> list = new ArrayList<T>();

    @Getter @Setter protected short targetId;


    protected transient Configuration configuration;
    protected transient Clipboard clipboard;
    protected transient Transport transport;
    protected transient Storage storage;
    protected transient NodeRole role;
    protected transient short shardIndex;
    protected transient TrainingDriver<? extends TrainingMessage> trainer;

    public Frame() {
        // nothing to do here
    }

    public Frame(@NonNull T message) {
        list.add(message);
    }

    public void stackMessage(@NonNull T message) {
        if (message.isJoinSupported()) {
            int index = list.indexOf(message);
            if (index >= 0)
                list.get(index).joinMessage(message);
            else
                list.add(message);
        } else list.add(message);
    }

    public void stackMessages(@NonNull Collection<T> messages) {
        for (T message: messages) {
            stackMessage(message);
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
    public long getTaskId() {
        return 0;
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
    public void attachContext(@NonNull Configuration configuration, @NonNull TrainingDriver<? extends TrainingMessage> trainer, @NonNull Clipboard clipboard, @NonNull Transport transport, @NonNull Storage storage, @NonNull NodeRole role, short shardIndex) {
        this.configuration = configuration;
        this.clipboard = clipboard;
        this.transport = transport;
        this.storage = storage;
        this.role = role;
        this.shardIndex = shardIndex;
        this.trainer = trainer;
    }

    @Override
    public void extractContext(@NonNull BaseVoidMessage message) {
        this.configuration = message.configuration;
        this.clipboard = message.clipboard;
        this.transport = message.transport;
        this.storage = message.storage;
        this.role = message.role;
        this.shardIndex = message.shardIndex;
        this.trainer = message.trainer;
    }

    @Override
    public void processMessage() {
        for(T message: list) {
            message.attachContext(configuration, trainer, clipboard, transport, storage, role, shardIndex);

            // if there's more then 1 round should be applied
            for (int i = 0; i < message.getCounter(); i++)
                message.processMessage();
        }
    }

    @Override
    public boolean isJoinSupported() {
        return false;
    }

    @Override
    public void joinMessage(VoidMessage message) {
        // no-op
    }
}
