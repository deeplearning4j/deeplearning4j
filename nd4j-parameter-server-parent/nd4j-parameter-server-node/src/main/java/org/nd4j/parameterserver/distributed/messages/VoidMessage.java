package org.nd4j.parameterserver.distributed.messages;

import org.agrona.DirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.lang3.SerializationUtils;

import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
public interface VoidMessage extends Serializable {

    int getMessageType();

    byte[] asBytes();

    UnsafeBuffer asUnsafeBuffer();

    public static VoidMessage fromBytes(byte[] array) {
        return (VoidMessage) SerializationUtils.deserialize(array);
    }
}
