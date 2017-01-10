package org.nd4j.parameterserver.distributed.transport.routing;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.transport.ClientRouter;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseRouter implements ClientRouter {
    protected VoidConfiguration voidConfiguration;
    protected Transport transport;

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Transport transport) {
        this.voidConfiguration = voidConfiguration;
        this.transport = transport;
    }
}
