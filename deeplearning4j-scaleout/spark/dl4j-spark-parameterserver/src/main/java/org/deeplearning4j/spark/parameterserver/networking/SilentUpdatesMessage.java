package org.deeplearning4j.spark.parameterserver.networking;

import lombok.Getter;
import org.agrona.concurrent.UnsafeBuffer;
import org.apache.commons.lang3.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
public class SilentUpdatesMessage implements TrainingMessage {

    @Getter protected INDArray updates;

    protected SilentUpdatesMessage() {
        // just for ser/de
    }

    public SilentUpdatesMessage(INDArray encodedUpdates) {
        this.updates = encodedUpdates;
    }


    @Override
    public byte getCounter() {
        return 0;
    }

    @Override
    public long getFrameId() {
        return 0;
    }

    @Override
    public void setFrameId(long frameId) {

    }

    @Override
    public void setTargetId(short id) {

    }

    @Override
    public short getTargetId() {
        return 0;
    }

    @Override
    public long getTaskId() {
        return 0;
    }

    @Override
    public int getMessageType() {
        return 0;
    }

    @Override
    public long getOriginatorId() {
        return 0;
    }

    @Override
    public void setOriginatorId(long id) {

    }

    @Override
    public byte[] asBytes() {
        return SerializationUtils.serialize(this);
    }

    public UnsafeBuffer asUnsafeBuffer() {
        return new UnsafeBuffer(asBytes());
    }

    @Override
    public void attachContext(VoidConfiguration voidConfiguration, TrainingDriver<? extends TrainingMessage> trainer, Clipboard clipboard, Transport transport, Storage storage, NodeRole role, short shardIndex) {

    }

    @Override
    public void extractContext(BaseVoidMessage message) {

    }

    @Override
    public void processMessage() {

    }

    @Override
    public boolean isJoinSupported() {
        return false;
    }

    @Override
    public boolean isBlockingMessage() {
        return false;
    }

    @Override
    public void joinMessage(VoidMessage message) {
        throw new UnsupportedOperationException("Join isn't supported for updaes");
    }

    @Override
    public int getRetransmitCount() {
        return 0;
    }

    @Override
    public void incrementRetransmitCount() {

    }
}
