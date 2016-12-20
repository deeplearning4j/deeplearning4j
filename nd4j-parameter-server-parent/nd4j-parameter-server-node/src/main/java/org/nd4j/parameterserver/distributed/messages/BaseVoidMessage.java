package org.nd4j.parameterserver.distributed.messages;

import lombok.*;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
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
    protected long batchId;

    protected transient Configuration configuration;
    protected transient Transport transport;
    protected transient NodeRole role;

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
    public void attachContext(@NonNull Configuration configuration, @NonNull Transport transport, @NonNull NodeRole role) {
        this.configuration = configuration;
        this.transport = transport;
        this.role = role;
    }
}
