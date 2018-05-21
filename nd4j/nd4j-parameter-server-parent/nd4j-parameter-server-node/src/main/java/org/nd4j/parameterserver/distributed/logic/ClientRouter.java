package org.nd4j.parameterserver.distributed.logic;

import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * This interface describes routing for messaging
 * flowing in Client->Shard direction
 *
 * @author raver119@gmail.com
 */
public interface ClientRouter {

    void init(VoidConfiguration voidConfiguration, Transport transport);

    int assignTarget(TrainingMessage message);

    int assignTarget(VoidMessage message);

    void setOriginator(VoidMessage message);
}
