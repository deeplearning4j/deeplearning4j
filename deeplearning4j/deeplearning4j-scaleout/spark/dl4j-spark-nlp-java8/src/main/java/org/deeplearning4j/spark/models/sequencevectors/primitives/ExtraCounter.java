package org.deeplearning4j.spark.models.sequencevectors.primitives;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.primitives.Counter;
import org.nd4j.parameterserver.distributed.util.NetworkInformation;

import java.net.InterfaceAddress;
import java.net.NetworkInterface;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * This class serves as Counter for SparkSequenceVectors vocab creation + for distributed parameters server organization
 * Ip addresses extracted here will be used for ParamServer shards selection, and won't be used for anything else
 *
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class ExtraCounter<E> extends Counter<E> {
    private Set<NetworkInformation> networkInformation;

    public ExtraCounter() {
        super();
        networkInformation = new HashSet<>();
    }

    public void buildNetworkSnapshot() {
        try {
            NetworkInformation netInfo = new NetworkInformation();
            netInfo.setTotalMemory(Runtime.getRuntime().maxMemory());
            netInfo.setAvailableMemory(Runtime.getRuntime().freeMemory());

            String sparkIp = System.getenv("SPARK_PUBLIC_DNS");
            if (sparkIp != null) {
                // if spark ip is defined, we just use it, and don't bother with other interfaces

                netInfo.addIpAddress(sparkIp);
            } else {
                // sparkIp wasn't defined, so we'll go for heuristics here
                List<NetworkInterface> interfaces = Collections.list(NetworkInterface.getNetworkInterfaces());

                for (NetworkInterface networkInterface : interfaces) {
                    if (networkInterface.isLoopback() || !networkInterface.isUp())
                        continue;

                    for (InterfaceAddress address : networkInterface.getInterfaceAddresses()) {
                        String addr = address.getAddress().getHostAddress();

                        if (addr == null || addr.isEmpty() || addr.contains(":"))
                            continue;

                        netInfo.getIpAddresses().add(addr);
                    }
                }
            }
            networkInformation.add(netInfo);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public <T extends E> void incrementAll(Counter<T> counter) {
        if (counter instanceof ExtraCounter) {
            networkInformation.addAll(((ExtraCounter) counter).networkInformation);
        }

        super.incrementAll(counter);
    }
}
