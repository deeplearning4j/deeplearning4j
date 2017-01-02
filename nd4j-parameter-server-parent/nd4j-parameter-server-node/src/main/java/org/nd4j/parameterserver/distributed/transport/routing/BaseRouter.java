package org.nd4j.parameterserver.distributed.transport.routing;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.transport.ClientRouter;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseRouter implements ClientRouter {
    protected Configuration configuration;
    protected Transport transport;

    @Override
    public void init(@NonNull Configuration configuration, @NonNull Transport transport) {
        this.configuration = configuration;
        this.transport = transport;
    }
}
