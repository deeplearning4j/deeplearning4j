package org.nd4j.parameterserver.distributed.logic.routing;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * Static router implementation, the same Shard will be used for all messages
 *
 * PLEASE NOTE: Never use this router in real world! It's suitable for debugging only.
 *
 * @author raver119@gmail.com
 */
public class StaticRouter extends BaseRouter {
    protected short targetIndex;

    public StaticRouter(int targetIndex) {
        this.targetIndex = (short) targetIndex;
    }

    public StaticRouter(short targetIndex) {
        this.targetIndex = targetIndex;
    }

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Transport transport) {
        super.init(voidConfiguration, transport);
    }

    @Override
    public int assignTarget(TrainingMessage message) {
        setOriginator(message);
        message.setTargetId(targetIndex);
        return targetIndex;
    }

    @Override
    public int assignTarget(VoidMessage message) {
        setOriginator(message);
        message.setTargetId(targetIndex);
        return targetIndex;
    }
}
