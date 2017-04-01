package org.nd4j.parameterserver.distributed.logic.routing;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.io.StringUtils;
import org.nd4j.linalg.util.HashUtil;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.logic.ClientRouter;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class BaseRouter implements ClientRouter {
    protected VoidConfiguration voidConfiguration;
    protected Transport transport;

    protected transient long originatorIdx = 0;

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Transport transport) {
        this.voidConfiguration = voidConfiguration;
        this.transport = transport;
    }

    @Override
    public void setOriginator(VoidMessage message) {
        if (originatorIdx == 0)
            originatorIdx = HashUtil.getLongHash(transport.getIp() + ":" + transport.getPort());

        message.setOriginatorId(originatorIdx);
    }


}
