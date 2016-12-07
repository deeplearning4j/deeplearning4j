package org.nd4j.parameterserver.distributed.conf;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.enums.FaultToleranceStrategy;

import java.io.Serializable;

/**
 * Basic configuration pojo for VoidParameterServer
 * @author raver119@gmail.com
 */
@NoArgsConstructor
@Slf4j
public class Configuration implements Serializable {
    private int port = 40123;
    private int numberOfShards;
    private FaultToleranceStrategy faultToleranceStrategy;
}
