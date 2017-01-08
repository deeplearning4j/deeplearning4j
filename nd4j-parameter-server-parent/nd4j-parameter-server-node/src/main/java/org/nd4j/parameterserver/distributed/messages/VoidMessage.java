package org.nd4j.parameterserver.distributed.messages;

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
public interface VoidMessage extends Serializable {

    void setTargetId(short id);

    short getTargetId();

    long getTaskId();

    int getMessageType();

    byte[] asBytes();

    UnsafeBuffer asUnsafeBuffer();

    static <T extends VoidMessage> T fromBytes(byte[] array) {
        return SerializationUtils.deserialize(array);
    }

    /**
     * This method initializes message for further processing
     */
    void attachContext(Configuration configuration, TrainingDriver<? extends TrainingMessage> trainer, Clipboard clipboard, Transport transport, Storage storage, NodeRole role, short shardIndex);

    void extractContext(BaseVoidMessage message);

    /**
     * This method will be started in context of executor, either Shard, Client or Backup node
     */
    void processMessage();

    boolean isJoinSupported();

    boolean isBlockingMessage();

    void joinMessage(VoidMessage message);

}
