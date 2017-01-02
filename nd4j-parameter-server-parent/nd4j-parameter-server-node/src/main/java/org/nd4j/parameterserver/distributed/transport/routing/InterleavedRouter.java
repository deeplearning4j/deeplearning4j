package org.nd4j.parameterserver.distributed.transport.routing;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * This is main router implementation for VoidParameterServer
 * Basic idea: We route TrainingMessages conditionally, based on Huffman tree index (aka frequency-ordered position)
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class InterleavedRouter extends BaseRouter {
    protected short targetIndex;

    @Override
    public void init(@NonNull Configuration configuration, @NonNull Transport transport) {
        super.init(configuration, transport);

        // by default messages are being routed to any random shard
        targetIndex = (short) RandomUtils.nextInt(0, configuration.getNumberOfShards());
    }

    @Override
    public void assignTarget(TrainingMessage message) {
        if (message instanceof SkipGramRequestMessage) {
            SkipGramRequestMessage sgrm = (SkipGramRequestMessage) message;

            int w1 = sgrm.getW1();
            if (w1 >= configuration.getNumberOfShards())
                message.setTargetId((short) (w1 % configuration.getNumberOfShards()));
            else
                message.setTargetId((short) configuration.getNumberOfShards());
        } else {
            message.setTargetId(targetIndex);
        }
    }

    @Override
    public void assignTarget(VoidMessage message) {
        message.setTargetId(targetIndex);
    }
}
