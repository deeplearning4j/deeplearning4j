package org.nd4j.parameterserver.distributed.logic;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;

/**
 * @author raver119@gmail.com
 */
public class Client extends BaseConnector {

    public Client(@NonNull Configuration configuration) {
        super(configuration, NodeRole.CLIENT);
    }

    public void init(){

    }
}
