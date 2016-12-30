package org.nd4j.parameterserver.distributed.messages;

import lombok.*;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.Clipboard;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.Transport;

import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
@NoArgsConstructor
@Data
public abstract class BaseVoidMessage implements VoidMessage {
    protected int messageType = -1;
    protected long nodeId;
    protected long taskId;
    protected short targetId;

    // these fields are used only for op invocation
    protected transient Configuration configuration;
    protected transient Clipboard clipboard;
    protected transient Transport transport;
    protected transient Storage storage;
    protected transient NodeRole role;
    protected transient short shardIndex;
    protected transient TrainingDriver<? extends TrainingMessage> trainer;

    protected BaseVoidMessage(int messageType) {
        this.messageType = messageType;
    }

    @Override
    public byte[] asBytes() {
        return SerializationUtils.serialize(this);
    }

    @Override
    public int getMessageType() {
        return messageType;
    }


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
        this.configuration =  message.configuration;
        this.clipboard = message.clipboard;
        this.transport = message.transport;
        this.storage = message.storage;
        this.role = message.role;
        this.shardIndex = message.shardIndex;
        this.trainer = message.trainer;
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
