package org.nd4j.parameterserver.distributed.messages;

import lombok.extern.slf4j.Slf4j;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.io.input.ClassLoaderObjectInputStream;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.Transport;

import java.io.ByteArrayInputStream;
import java.io.ObjectInputStream;
import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
public interface VoidMessage extends Serializable {

    void setTargetId(short id);

    short getTargetId();

    long getTaskId();

    int getMessageType();

    long getOriginatorId();

    void setOriginatorId(long id);

    byte[] asBytes();

    UnsafeBuffer asUnsafeBuffer();

    static <T extends VoidMessage> T fromBytes(byte[] array) {
        try {
            ObjectInputStream in = new ClassLoaderObjectInputStream(Thread.currentThread().getContextClassLoader(),
                            new ByteArrayInputStream(array));

            T result = (T) in.readObject();
            return result;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        //return SerializationUtils.deserialize(array);
    }

    /**
     * This method initializes message for further processing
     */
    void attachContext(VoidConfiguration voidConfiguration, TrainingDriver<? extends TrainingMessage> trainer,
                    Clipboard clipboard, Transport transport, Storage storage, NodeRole role, short shardIndex);

    void extractContext(BaseVoidMessage message);

    /**
     * This method will be started in context of executor, either Shard, Client or Backup node
     */
    void processMessage();

    boolean isJoinSupported();

    boolean isBlockingMessage();

    void joinMessage(VoidMessage message);

    int getRetransmitCount();

    void incrementRetransmitCount();
}
