package org.nd4j.parameterserver.distributed.messages;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.lang3.SerializationUtils;

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
}
