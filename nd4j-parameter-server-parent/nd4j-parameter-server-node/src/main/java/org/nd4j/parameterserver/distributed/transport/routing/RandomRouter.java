package org.nd4j.parameterserver.distributed.transport.routing;

import lombok.NonNull;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.transport.ClientRouter;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * Basic implementation for ClientRouter: we route each message to random Shard
 *
 *
 *
 * @author raver119@gmail.com
 */
public class RandomRouter extends BaseRouter {
    protected int numShards;

    @Override
    public void init(@NonNull Configuration configuration, @NonNull Transport transport) {
        super.init(configuration, transport);

        int numShards = configuration.getNumberOfShards();
    }

    @Override
    public void assignTarget(TrainingMessage message) {
        message.setTargetId(getNextShard());
    }

    @Override
    public void assignTarget(VoidMessage message) {
        message.setTargetId(getNextShard());
    }

    protected short getNextShard() {
        return (short) RandomUtils.nextInt(0, numShards);
    }
}
