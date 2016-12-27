package org.nd4j.parameterserver.distributed.training;

import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.logic.Clipboard;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.aggregations.VoidAggregation;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
public interface TrainingDriver<T extends TrainingMessage> {

    void init(Configuration configuration, Transport transport, Storage storage, Clipboard clipboard);

    void doTraining(T message);

    void aggregationFinished(VoidAggregation aggregation);

    String targetMessageClass();
}
