/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.parameterserver.distributed;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.config.ND4JEnvironmentVars;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.ExecutionMode;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.*;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.logic.sequence.BasicSequenceProvider;
import org.nd4j.parameterserver.distributed.logic.storage.WordVectorStorage;
import org.nd4j.parameterserver.distributed.messages.*;
import org.nd4j.parameterserver.distributed.messages.requests.*;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.training.impl.SkipGramTrainer;
import org.nd4j.parameterserver.distributed.transport.MulticastTransport;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;
import org.nd4j.parameterserver.distributed.transport.Transport;
import org.nd4j.parameterserver.distributed.util.NetworkOrganizer;

import java.net.InterfaceAddress;
import java.net.NetworkInterface;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
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

    @Getter
    protected volatile NodeRole nodeRole;

    protected volatile VoidConfiguration voidConfiguration;


    protected AtomicBoolean initLocker = new AtomicBoolean(false);
    protected AtomicBoolean initFinished = new AtomicBoolean(false);
    protected AtomicBoolean shutdownLocker = new AtomicBoolean(false);
    protected AtomicBoolean shutdownFinished = new AtomicBoolean(false);

    protected transient Transport transport;

    protected transient AtomicBoolean manualMode = new AtomicBoolean(false);
    protected transient AtomicBoolean runner = new AtomicBoolean(false);

    protected transient Thread[] processingThreads;
    protected transient Runnable[] processingRunnables;

    // FIXME: we want trainer to be configurable here
    protected transient TrainingDriver<? extends TrainingMessage> trainer;

    protected short shardIndex;

    protected Clipboard clipboard = new Clipboard();

    protected Storage storage = new WordVectorStorage();

    protected Map<String, Frame<TrainingMessage>> frames = new ConcurrentHashMap<>();

    protected static final int numThreads = Runtime.getRuntime().availableProcessors() * 2;
    protected ThreadPoolExecutor executor =
                    (ThreadPoolExecutor) Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() * 2);


    ////////////////////// SeqVec part

    protected static double MAX_EXP = 6;

    ////////////////////// end of SeqVec part


    protected VoidParameterServer() {
        nodeRole = NodeRole.NONE;
    }

    protected VoidParameterServer(@NonNull NodeRole nodeRole) {
        this.nodeRole = nodeRole;
    }

    protected VoidParameterServer(boolean manualMode) {
        this();
        this.manualMode.set(manualMode);
    }

    public static VoidParameterServer getInstance() {
        return INSTANCE;
    }

    public void setTrainingDriver(@NonNull TrainingDriver<? extends TrainingMessage> trainer) {
        this.trainer = trainer;
    }

    /**
     * This method returns shardIndex value.
     * If current node is Shard - it reutrns it's value
     * If current node is client - it returns Shard index of paired Shard
     * @return
     */
    public short getShardIndex() {
        return shardIndex;
    }

    protected void setIpPortForShard(String ip, int port) {
        transport.setIpAndPort(ip, port);
    }

    protected void setShardIndex(short idx) {
        shardIndex = idx;
    }

    protected Transport getTransport() {
        return transport;
    }

    protected INDArray getSyn0() {
        return storage.getArray(WordVectorStorage.SYN_0);
    }

    protected INDArray getSyn1() {
        return storage.getArray(WordVectorStorage.SYN_1);
    }

    protected INDArray getSyn1Neg() {
        return storage.getArray(WordVectorStorage.SYN_1_NEGATIVE);
    }

    protected INDArray getExpTable() {
        return storage.getArray(WordVectorStorage.EXP_TABLE);
    }

    protected INDArray getNegTable() {
        return storage.getArray(WordVectorStorage.NEGATIVE_TABLE);
    }

    protected void init(@NonNull VoidConfiguration voidConfiguration) {
        init(voidConfiguration, new RoutedTransport(), new SkipGramTrainer());
    }

    /**
     * This method returns True if initialization was started AND was finished, false otherwise
     * @return
     */
    public boolean isInit() {
        return initFinished.get();
    }

    /**
     * This method starts ParameterServer instance
     *
     * PLEASE NOTE: This method is blocking for first caller only
     */
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Transport transport,
                    TrainingDriver<? extends TrainingMessage> trainer) {
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
                this.trainer = trainer;
                this.voidConfiguration = voidConfiguration;

                this.transport = transport;

                // first we need to check, if our current IP matches designated shards or backup
                if (nodeRole == NodeRole.NONE && (voidConfiguration.getForcedRole() == null
                                || voidConfiguration.getForcedRole() == NodeRole.NONE)) {
                    Pair<NodeRole, String> pair = null;
                    if (voidConfiguration.getShardAddresses().size() == 1
                                    && voidConfiguration.getShardAddresses().get(0).contains("127.0.0.1")) {
                        pair = Pair.create(NodeRole.SHARD, voidConfiguration.getShardAddresses().get(0));
                    } else {
                        pair = getRole(voidConfiguration, getLocalAddresses());
                    }
                    nodeRole = pair.getFirst();

                    String ipAndPort = pair.getSecond();
                    String ip = "127.0.0.1";
                    int port = 0;
                    // if we're Shard and have port enforced
                    if (ipAndPort.contains(":")) {
                        String[] split = ipAndPort.split(":");
                        ip = split[0];
                        port = Integer.valueOf(split[1]);
                    } else {
                        ip = ipAndPort;
                        port = voidConfiguration.getUnicastPort();
                    }

                    // if we're Shard here, we should define shardIndex
                    if (nodeRole == NodeRole.SHARD && voidConfiguration.getShardAddresses().size() > 1) {
                        short cnt = 0;
                        for (String shard : voidConfiguration.getShardAddresses()) {
                            String lIp = null;
                            if (shard.contains(":")) {
                                String[] split = ipAndPort.split(":");
                                lIp = split[0];
                            } else
                                lIp = shard;

                            if (lIp.equals(ip)) {
                                shardIndex = cnt;
                            }
                            cnt++;
                        }
                    }

                    this.transport.init(voidConfiguration, clipboard, nodeRole, ip, port, shardIndex);

                } else {
                    if (nodeRole == NodeRole.NONE)
                        nodeRole = voidConfiguration.getForcedRole();

                    // if we're using forced roles here, we'll assume that controllerAddress belongs to this box
                    String localIp = voidConfiguration.getExecutionMode() == ExecutionMode.MANAGED
                                    ? voidConfiguration.getControllerAddress() : "127.0.0.1";

                    this.transport.init(voidConfiguration, clipboard, nodeRole, localIp,
                                    voidConfiguration.getUnicastPort(), shardIndex);
                }


                // TODO: we need real ip only if this is a shard *FOR NOW*, but later we'll need it for client as well

                // we launch message processing if we're not in debug mode
                if (!manualMode.get()) {
                    processingThreads = new Thread[numThreads];
                    processingRunnables = new Runnable[numThreads];

                    for (int x = 0; x < numThreads; x++) {
                        processingThreads[x] = new Thread(() -> {
                            runner.set(true);
                            while (runner.get()) {
                                try {
                                    //VoidMessage message = transport.takeMessage();

                                    //                                    if (nodeRole == NodeRole.SHARD)
                                    //                                        log.info("Processing message: {}", message.getClass().getSimpleName());

                                    handleMessage(transport.takeMessage());

                                } catch (ND4JIllegalStateException e) {
                                    throw new RuntimeException(e);
                                } catch (Exception e) {
                                    throw new RuntimeException(e);
                                }
                            }
                        });

                        //executor.submit(processingRunnables[x);

                        // TODO: maybe find the way to guarantee affinity in some other way, to make different devices usable as well?
                        Nd4j.getAffinityManager().attachThreadToDevice(processingThreads[x],
                                        Nd4j.getAffinityManager().getDeviceForCurrentThread());
                        processingThreads[x].setDaemon(true);
                        processingThreads[x].setName("VoidParameterServer messages handling thread");
                        processingThreads[x].start();
                    }
                }


                log.info("Launching transport...");
                transport.launch(Transport.ThreadingModel.DEDICATED_THREADS);
                trainer.init(this.voidConfiguration, this.transport, storage, clipboard);

                initFinished.set(true);
            }
        }
    }

    /**
     * This method is available for debug purposes only
     *
     * @param mode
     */
    protected VoidParameterServer toggleManualMode(boolean mode) {
        manualMode.set(mode);
        return this;
    }

    /**
     * This method checks for designated role, according to local IP addresses and configuration passed into method
     *
     * @param voidConfiguration
     * @param localIPs
     * @return
     */
    protected Pair<NodeRole, String> getRole(@NonNull VoidConfiguration voidConfiguration,
                    @NonNull Collection<String> localIPs) {
        NodeRole result = NodeRole.CLIENT;

        for (String ip : voidConfiguration.getShardAddresses()) {
            String cleansed = ip.replaceAll(":.*", "");
            if (localIPs.contains(cleansed))
                return Pair.create(NodeRole.SHARD, ip);
        }

        if (voidConfiguration.getBackupAddresses() != null)
            for (String ip : voidConfiguration.getBackupAddresses()) {
                String cleansed = ip.replaceAll(":.*", "");
                if (localIPs.contains(cleansed))
                    return Pair.create(NodeRole.BACKUP, ip);
            }


        String sparkIp = null;


        if (sparkIp == null && voidConfiguration.getNetworkMask() != null) {
            NetworkOrganizer organizer = new NetworkOrganizer(voidConfiguration.getNetworkMask());
            sparkIp = organizer.getMatchingAddress();
        }

        // last resort here...
        if (sparkIp == null)
            sparkIp = System.getenv(ND4JEnvironmentVars.DL4J_VOID_IP);


        log.info("Got [{}] as sparkIp", sparkIp);
        if (sparkIp == null)
            throw new ND4JIllegalStateException("Can't get IP address for UDP communcation");

        // local IP from pair is used for shard only, so we don't care
        return Pair.create(result, sparkIp + ":" + voidConfiguration.getUnicastPort());
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
            log.info("Shutting down transport...");

            // we just sending out ShutdownRequestMessage
            //transport.sendMessage(new ShutdownRequestMessage());
            transport.shutdown();

            executor.shutdown();
            initFinished.set(false);
            initLocker.set(false);
            shutdownLocker.set(false);
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

                for (InterfaceAddress address : networkInterface.getInterfaceAddresses()) {
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


    // TODO: remove @NonNull check here
    protected void handleMessage(@NonNull VoidMessage message) {
        if (message == null) {
            //            log.info("sI_{} got null message", getShardIndex());
            return;
        }

        if (message.getTargetId() >= 0 && message.getTargetId() != shardIndex) {
            log.warn("sI_{}: Skipping message: [{}]; TargetIdx: [{}]", shardIndex, message.getClass().getSimpleName(),
                            message.getTargetId());
            return;
        }

        //      log.info("sI_{}: Processing message: [{}]", shardIndex, message.getClass().getSimpleName());

        message.attachContext(voidConfiguration, trainer, clipboard, transport, storage, nodeRole, shardIndex);
        message.processMessage();
    }

    /**
     * This method handles Shards initialization
     *
     * PLEASE NOTE: This method is blocking
     */
    // TODO: right now we support only columnar splits over tables
    public void initializeSeqVec(int vectorLength, int numWords, long seed, int columnsPerShard, boolean useHs,
                    boolean useNegSampling) {
        InitializationRequestMessage dim = new InitializationRequestMessage(vectorLength, numWords, seed, useHs,
                        useNegSampling, columnsPerShard);
        transport.sendMessage(dim);
    }

    /**
     * This method dispatches TrainingMessage to ParameterServer network
     *
     * PLEASE NOTE: This method is synchronized and *periodically* becomes blocking by design
     * @param message
     */
    public synchronized void execDistributed(@NonNull TrainingMessage message) {
        /**
         * Basically we should batch messages coming from different TrainingFunctions on spark executor side here.
         * So we pack them into batches, and send over the wire to selected Shard
         */
        Frame currentFrame;
        if ((currentFrame = frames.get(message.getClass().getSimpleName())) == null) {
            currentFrame = new Frame<>(BasicSequenceProvider.getInstance().getNextValue());
            frames.put(message.getClass().getSimpleName(), currentFrame);
        }

        currentFrame.stackMessage(message);

        // TODO: make this threshold variable
        if (currentFrame.size() >= 128) {
            transport.sendMessage(currentFrame);
            currentFrame = new Frame<>(BasicSequenceProvider.getInstance().getNextValue());
            frames.put(message.getClass().getSimpleName(), currentFrame);
        }

        //transport.sendMessage(message);
    }

    public void execDistributedImmediately(@NonNull TrainingMessage message) {
        transport.sendMessageToAllShards(message);
    }

    public void execDistributed(@NonNull Frame<? extends TrainingMessage> messages) {
        transport.sendMessage(messages);
    }


    public INDArray getVector(int rowIdx) {
        return getVector(WordVectorStorage.SYN_0, rowIdx);
    }

    /**
     * This method returns INDArray matching requested storageId value
     *
     * PLEASE NOTE: This method IS blocking
     *
     * @param rowIdx
     * @return
     */
    public INDArray getVector(@NonNull Integer key, int rowIdx) {
        /**
         * we create VoidMessage, send it, and block until it gets responded
         */

        VectorRequestMessage message = new VectorRequestMessage(key, rowIdx);

        MeaningfulMessage response = transport.sendMessageAndGetResponse(message);

        return response.getPayload();
    }

    /**
     * This method sends given message to all Shards
     *
     * @param message
     */
    public synchronized void sendMessageToAllShards(@NonNull VoidMessage message) {
        transport.sendMessageToAllShards(message);
    }

    /**
     * This method sends given message to all Clients
     *
     * @param message
     */
    public void sendMessageToAllClients(@NonNull VoidMessage message) {
        this.sendMessageToAllClients(message, null);
    }

    /**
     * This method sends given message to all Clients, excluding
     *
     * @param message
     */
    public synchronized void sendMessageToAllClients(@NonNull VoidMessage message, Long... exclusions) {
        transport.sendMessageToAllClients(message);
    }
}
