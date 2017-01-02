package org.nd4j.parameterserver.distributed.transport.routing;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.Configuration;
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
    public void init(@NonNull Configuration configuration, @NonNull Transport transport) {
        super.init(configuration, transport);
    }

    @Override
    public void assignTarget(TrainingMessage message) {
        message.setTargetId(targetIndex);
    }

    @Override
    public void assignTarget(VoidMessage message) {
        message.setTargetId(targetIndex);
    }
}
