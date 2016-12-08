package org.nd4j.parameterserver.distributed.conf;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.enums.FaultToleranceStrategy;

import java.io.Serializable;
import java.util.List;

/**
 * Basic configuration pojo for VoidParameterServer
 * @author raver119@gmail.com
 */
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Slf4j
@Data
public class Configuration implements Serializable {
    private int port;
    private int numberOfShards;
    private FaultToleranceStrategy faultToleranceStrategy;
    private List<String> shardAddresses;
    private List<String> backupAddresses;
}
