package org.nd4j.parameterserver.distributed.training;

import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
public interface TrainingDriver<T extends TrainingMessage> {

    void init(VoidConfiguration voidConfiguration, Transport transport, Storage storage, Clipboard clipboard);

    void startTraining(T message);

    void pickTraining(T message);

    void aggregationFinished(VoidAggregation aggregation);

    void finishTraining(long originatorId, long taskId);

    void addCompletionHook(long originatorId, long frameId, long messageId);

    String targetMessageClass();
}
