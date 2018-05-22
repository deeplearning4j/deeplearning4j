package org.nd4j.parameterserver.distributed.logic.routing;

import lombok.NonNull;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
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
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Transport transport) {
        super.init(voidConfiguration, transport);

        int numShards = voidConfiguration.getNumberOfShards();
    }

    @Override
    public int assignTarget(TrainingMessage message) {
        setOriginator(message);
        message.setTargetId(getNextShard());
        return message.getTargetId();
    }

    @Override
    public int assignTarget(VoidMessage message) {
        setOriginator(message);
        message.setTargetId(getNextShard());
        return message.getTargetId();
    }

    protected short getNextShard() {
        return (short) RandomUtils.nextInt(0, numShards);
    }
}
