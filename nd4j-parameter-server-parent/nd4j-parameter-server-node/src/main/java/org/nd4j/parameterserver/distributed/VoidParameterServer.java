package org.nd4j.parameterserver.distributed;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;

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


    protected VoidParameterServer() {
        nodeRole = NodeRole.NONE;
    }

    public static VoidParameterServer getInstance() {
        return INSTANCE;
    }


    /**
     * This method starts ParameterServer instance
     *
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

                // we need to start coordination channel, to start election process


                // after coordination channel is set, we start decision process, which role this node going to serve

                initFinished.set(true);
            }
        }
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
}
