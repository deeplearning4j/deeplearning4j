package org.nd4j.parameterserver.distributed.conf;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.enums.ExecutionMode;
import org.nd4j.parameterserver.distributed.enums.FaultToleranceStrategy;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.enums.TransportType;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
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
public class VoidConfiguration implements Serializable {

    @Builder.Default private int streamId = 119;
    @Builder.Default private int unicastPort = 49876;
    @Builder.Default private int multicastPort = 59876;
    @Builder.Default private int numberOfShards = 1;;

    @Builder.Default private FaultToleranceStrategy faultToleranceStrategy = FaultToleranceStrategy.NONE;
    @Builder.Default private ExecutionMode executionMode = ExecutionMode.SHARDED;

    @Builder.Default private List<String> shardAddresses = new ArrayList<>();
    @Builder.Default private List<String> backupAddresses = new ArrayList<>();
    @Builder.Default private TransportType transportType = TransportType.ROUTED;

    // this is very important parameter
    private String networkMask;

    // This two values are optional, and have effect only for MulticastTransport
    @Builder.Default private String multicastNetwork = "224.0.1.1";
    private String multicastInterface;
    @Builder.Default private int ttl = 4;
    protected NodeRole forcedRole;

    // FIXME: probably worth moving somewhere else
    // this part is specific to w2v
    private boolean useHS = true;
    private boolean useNS = false;

    @Builder.Default private long retransmitTimeout = 1000;
    @Builder.Default private long responseTimeframe = 500;
    @Builder.Default private long responseTimeout = 30000;

    private String controllerAddress;

    public void setStreamId(int streamId) {
        if (streamId < 1)
            throw new ND4JIllegalStateException("You can't use streamId 0, please specify other one");

        this.streamId = streamId;
    }

    protected void validateNetmask() {
        // micro-validaiton here
        String[] chunks = networkMask.split("\\.");
        if (chunks.length == 1 || networkMask.isEmpty())
            throw new ND4JIllegalStateException("Provided netmask doesn't look like a legit one. Proper format is: 192.168.1.0/24 or 10.0.0.0/8");


        // TODO: add support for IPv6 eventually here
        if (chunks.length != 4) {
            throw new ND4JIllegalStateException("4 octets expected here for network mask");
        }

        for (int i = 0; i < 3; i++) {
            String curr = chunks[i];
            try {
                int conv = Integer.valueOf(curr);
                if (conv < 0 || conv > 255)
                    throw new ND4JIllegalStateException();
            } catch (Exception e) {
                throw new ND4JIllegalStateException("All IP address octets should be in range of 0...255");
            }
        }

        if (Integer.valueOf(chunks[0]) == 0)
            throw new ND4JIllegalStateException("First network mask octet should be non-zero. I.e. 10.0.0.0/8");

        // we enforce last octet to be 0/24 always
        if (!networkMask.contains("/") || !chunks[3].startsWith("\\0\\/")) {
            chunks[3] = "0/24";
        }

        this.networkMask = chunks[0] + "." + chunks[1] + "." + chunks[2] + "." + chunks[3];
    }

    /**
     * This option is very important: in shared network environment and yarn (like on EC2 etc),
     * please set this to the network, which will be available on all boxes. I.e. 10.1.1.0/24 or 192.168.0.0/16
     *
     * @param netmask netmask to be used for IP address selection
     */
    public void setNetworkMask(@NonNull String netmask) {
        this.networkMask = netmask;
        validateNetmask();
    }

    /**
     * This method returns network mask
     *
     * @return
     */
    public String getNetworkMask() {
        validateNetmask();
        return this.networkMask;
    }

    public void setShardAddresses(List<String> addresses) {
        this.shardAddresses = addresses;
    }

    public void setShardAddresses(String... ips) {
        if (shardAddresses == null)
            shardAddresses = new ArrayList<>();

        for (String ip : ips) {
            if (ip != null)
                shardAddresses.add(ip);
        }
    }

    public void setBackupAddresses(List<String> addresses) {
        this.backupAddresses = addresses;
    }

    public void setBackupAddresses(String... ips) {
        if (backupAddresses == null)
            backupAddresses = new ArrayList<>();

        for (String ip : ips) {
            if (ip != null)
                backupAddresses.add(ip);
        }
    }

    public void setExecutionMode(@NonNull ExecutionMode executionMode) {
        this.executionMode = executionMode;
    }
}
