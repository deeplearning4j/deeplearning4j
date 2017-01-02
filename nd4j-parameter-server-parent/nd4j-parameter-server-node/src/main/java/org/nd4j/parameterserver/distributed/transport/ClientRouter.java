package org.nd4j.parameterserver.distributed.transport;

import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;

/**
 * This interface describes routing for messagins flowing in Client->Shard direction
 *
 * @author raver119@gmail.com
 */
public interface ClientRouter {

    void init(Configuration configuration, Transport transport);

    void assignTarget(TrainingMessage message);

    void assignTarget(VoidMessage message);
}
