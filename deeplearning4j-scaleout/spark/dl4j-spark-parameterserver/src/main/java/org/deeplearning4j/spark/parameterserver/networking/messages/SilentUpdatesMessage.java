package org.deeplearning4j.spark.parameterserver.networking.messages;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SilentUpdatesMessage extends BaseVoidMessage implements TrainingMessage, RequestMessage {

    @Getter
    protected long updateId;
    @Getter
    protected INDArray updates;
    protected long frameId;

    protected SilentUpdatesMessage() {
        // just for ser/de
    }

    public SilentUpdatesMessage(INDArray encodedUpdates, long updateId) {
        this.updates = encodedUpdates;
        this.updateId = updateId;
    }


    @Override
    public void attachContext(VoidConfiguration voidConfiguration, TrainingDriver<? extends TrainingMessage> trainer,
                    Clipboard clipboard, Transport transport, Storage storage, NodeRole role, short shardIndex) {
        this.voidConfiguration = voidConfiguration;
        this.trainer = trainer;
        this.transport = transport;
    }

    @Override
    public void processMessage() {
        // basically no-op?
        TrainingDriver<SilentUpdatesMessage> tr = (TrainingDriver<SilentUpdatesMessage>) trainer;
        tr.startTraining(this);
    }

    @Override
    public byte getCounter() {
        return 0;
    }

    @Override
    public long getFrameId() {
        return frameId;
    }

    @Override
    public void setFrameId(long frameId) {
        this.frameId = frameId;
    }

    @Override
    public boolean isJoinSupported() {
        return false;
    }
}
