package org.nd4j.parameterserver.distributed;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.Connector;
import org.nd4j.parameterserver.distributed.logic.Shard;

import java.net.InterfaceAddress;
import java.net.NetworkInterface;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * This is "special case" distributed P2P parameter server implementation, suitable for Spark Word2Vec/ParagraphVectors/DeepWalk implementations.
 * Aeron is used as backbone for messaging system here.
 *
 * Highlights:
 * a) It does ONLY one-way messaging. Clients are NOT getting anything back from ParamServer.
 * b) It works sharded. Amount of shards is defined in runtime.
 * c) Data replication strategy is defined in runtime.
 * d) It's supposed to be used as singleton in Spark environment ONLY.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class VoidParameterServer {
    private static final VoidParameterServer INSTANCE = new VoidParameterServer();

    @Getter protected volatile NodeRole nodeRole;

    protected volatile Configuration configuration;


    protected AtomicBoolean initLocker = new AtomicBoolean(false);
    protected AtomicBoolean initFinished = new AtomicBoolean(false);
    protected AtomicBoolean shutdownLocker = new AtomicBoolean(false);
    protected AtomicBoolean shutdownFinished = new AtomicBoolean(false);



    protected Connector connector;

    protected VoidParameterServer() {
        nodeRole = NodeRole.NONE;
    }

    public static VoidParameterServer getInstance() {
        return INSTANCE;
    }


    /**
     * This method starts ParameterServer instance
     *
     * PLEASE NOTE: This method is blocking for first caller only
     */
    public void init(@NonNull Configuration configuration){
        /**
         * Basic plan here:
         *      start publishers/listeners/subscribers
         *      determine role of the current instance:
         *          Shard
         *          Backup
         *          Client
         *      shutdown unwanted aeron helpers (according to role)
         *      wait for incoming task queries (according to role
         *
         */
        if (initFinished.get())
            return;

        synchronized (this) {
            if (initLocker.compareAndSet(false, true)) {
                this.configuration = configuration;

            // first we need to check, if our current IP matches designated shards or backup
            nodeRole = getRole(configuration, getLocalAddresses());

            // role-dependent additional initialization
            switch (nodeRole) {
                case SHARD: {
                    log.info("Initializing as Shard...");

                    connector = new Shard(configuration);
                    break;
                }
                case BACKUP: {
                    log.info("Initializing as Backup...");

                    break;
                }
                case MASTER: {
                    log.info("Initializing as Master...");

                    break;
                }
                case CLIENT:
                default: {
                    log.info("Initializing as Client...");
                    break;
                }
            }

                initFinished.set(true);
            }
        }
    }

    /**
     * This method checks for designated role, according to local IP addresses and configuration passed into method
     *
     * @param configuration
     * @param localIPs
     * @return
     */
    protected NodeRole getRole(@NonNull Configuration configuration, @NonNull Collection<String> localIPs) {
        NodeRole result = NodeRole.CLIENT;

        for (String ip: localIPs) {
            if (configuration.getShardAddresses().contains(ip))
                return NodeRole.SHARD;
        }

        for (String ip: localIPs) {
            if (configuration.getBackupAddresses().contains(ip))
                return NodeRole.BACKUP;
        }

        return result;
    }

    /**
     * This method initiates shutdown sequence for this instance.
     *
     * PLEASE NOTE: This method is blocking for first caller only
     */
    public void shutdown() {
        /**
         * Probably we don't need this method in practice
         */
        if (initLocker.get() && shutdownLocker.compareAndSet(false, true)) {
            // do shutdown
        }
    }

    /**
     * This method returns set of local IP addresses available in system.
     *
     * PLEASE NOTE: loopback, disabled interfaces, IPv6 addresses are ignored here.
     *
     * @return
     */
    public static Set<String> getLocalAddresses() {
        try {
            List<NetworkInterface> interfaces = Collections.list(NetworkInterface.getNetworkInterfaces());

            Set<String> result = new HashSet<>();

            for (NetworkInterface networkInterface : interfaces) {
                if (networkInterface.isLoopback() || !networkInterface.isUp())
                    continue;

                for(InterfaceAddress address: networkInterface.getInterfaceAddresses()) {
                    String addr = address.getAddress().getHostAddress();

                    if (addr == null || addr.isEmpty() || addr.contains(":"))
                        continue;

                    result.add(addr);
                }
            }

            return result;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
