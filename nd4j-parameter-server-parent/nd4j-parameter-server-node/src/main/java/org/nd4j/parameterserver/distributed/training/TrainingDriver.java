package org.nd4j.parameterserver.distributed.training;

import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.logic.Clipboard;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
public interface TrainingDriver<T extends TrainingMessage> {

    void init(Configuration configuration, Transport transport, Clipboard clipboard);

    void doTraining(T message);

    String targetMessageClass();
}
