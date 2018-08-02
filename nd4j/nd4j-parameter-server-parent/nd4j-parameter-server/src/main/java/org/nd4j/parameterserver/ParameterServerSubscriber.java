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

package org.nd4j.parameterserver;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.beust.jcommander.Parameters;
import com.google.common.primitives.Ints;
import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.Unirest;
import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.agrona.CloseHelper;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.json.JSONObject;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.aeron.ipc.NDArrayHolder;
import org.nd4j.aeron.ipc.response.AeronNDArrayResponder;
import org.nd4j.aeron.ndarrayholder.InMemoryNDArrayHolder;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.parameterserver.model.MasterConnectionInfo;
import org.nd4j.parameterserver.model.ServerState;
import org.nd4j.parameterserver.model.SlaveConnectionInfo;
import org.nd4j.parameterserver.model.SubscriberState;
import org.nd4j.parameterserver.updater.ParameterServerUpdater;
import org.nd4j.parameterserver.updater.SoftSyncParameterUpdater;
import org.nd4j.parameterserver.updater.SynchronousParameterUpdater;
import org.nd4j.parameterserver.updater.storage.InMemoryUpdateStorage;
import org.nd4j.parameterserver.util.CheckSocket;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

/**
 * Subscriber main class for
 * the parameter
 * averaging server
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
@Data
@Parameters(separators = ",")
public class ParameterServerSubscriber implements AutoCloseable {

    private static Logger log = LoggerFactory.getLogger(ParameterServerSubscriber.class);

    @Parameter(names = {"-p", "--port"}, description = "The port to listen on for the daemon", arity = 1)
    private int port = 40123;
    @Parameter(names = {"-id", "--streamId"}, description = "The stream id to listen on", arity = 1)
    private int streamId = 10;
    @Parameter(names = {"-h", "--host"}, description = "Host for the server to bind to", arity = 1)
    private String host = "localhost";
    @Parameter(names = {"-d", "--deleteDirectoryOnStart"}, description = "Delete aeron directory on startup.",
                    arity = 1)
    private boolean deleteDirectoryOnStart = true;
    @Parameter(names = {"-m", "--master"}, description = "Whether this subscriber is a master node or not.", arity = 1)
    private boolean master = false;
    @Parameter(names = {"-pm", "--publishmaster"},
                    description = "Publish master url: host:port - this is for peer nodes needing to publish to another peer.",
                    arity = 1)
    private String publishMasterUrl = "localhost:40123";
    @Parameter(names = {"-md", "--mediadriverdirectory"},
                    description = "The media driver directory opName. This is for when the media driver is started as a separate process.",
                    arity = 1)
    private String mediaDriverDirectoryName;
    @Parameter(names = {"-sp", "--statusserverport"}, description = "The status server port, defaults to 9000.",
                    arity = 1)
    private int statusServerPort = 9000;
    @Parameter(names = {"-sh", "--statusserverhost"}, description = "The status host, defaults to localhost.",
                    arity = 1)
    private String statusServerHost = "localhost";
    @Parameter(names = {"-up", "--update"},
                    description = "The update opType for this parameter server. Defaults to sync. You can specify custom and use a jvm argument -Dorg.nd4j.parameterserver.updatetype=your.fully.qualified.class if you want to use a custom class. This must be able to be instantiated from an empty constructor though.",
                    arity = 1)
    private String updateTypeString = UpdateType.SYNC.toString().toLowerCase();

    private UpdateType updateType = UpdateType.SYNC;

    @Parameter(names = {"-s", "--shape"}, description = "The shape of the ndarray", arity = 1)
    private List<Integer> shape;
    @Parameter(names = {"-hbi", "--heartbeatinterval"}, description = "Heartbeat interval in ms", arity = 1)
    private int heartbeatMs = 1000;
    private ObjectMapper objectMapper = new ObjectMapper();
    private ScheduledExecutorService scheduledExecutorService;
    @Parameter(names = {"-u", "--updatesPerEpoch"}, description = "The number of updates per epoch", arity = 1,
                    required = true)
    private int updatesPerEpoch;


    /**
     * Specify a custom class as a jvm arg.
     * Note that this class must be a fully qualified classname
     */
    public final static String CUSTOM_UPDATE_TYPE = "org.nd4j.parameterserver.updatetype";

    /**
     * Update types are for
     * instantiating various kinds of update types
     */
    public enum UpdateType {
        HOGWILD, SYNC, TIME_DELAYED, SOFTSYNC, CUSTOM
    }



    private MediaDriver mediaDriver;
    private AeronNDArrayResponder responder;
    private AeronNDArraySubscriber subscriber;
    private NDArrayCallback callback;
    //alias for the callback where relevant
    private ParameterServerListener parameterServerListener;
    private Aeron aeron;
    private ScheduledExecutorService heartbeat;

    /**
     * Allow passing in a
     * media driver that already exists
     *
     * @param mediaDriver
     */
    public ParameterServerSubscriber(MediaDriver mediaDriver) {
        Preconditions.checkNotNull(mediaDriver);
        this.mediaDriver = mediaDriver;
    }



    /**
     * Return the current {@link SubscriberState}
     * of this subscriber
     *
     * @return the current state of this subscriber
     */
    public SubscriberState asState() {
        return SubscriberState.builder()
                        .parameterUpdaterStatus(parameterServerListener == null ? Collections.emptyMap()
                                        : parameterServerListener.getUpdater().status())
                        .isMaster(isMaster())
                        .connectionInfo(isMaster() ? masterConnectionInfo().toString()
                                        : slaveConnectionInfo().toString())
                        .isAsync(parameterServerListener.getUpdater().isAsync())
                        .isReady(parameterServerListener.getUpdater().isReady())
                        .totalUpdates(getResponder().getNdArrayHolder().totalUpdates()).streamId(streamId)
                        .serverState(subscriberLaunched() ? ServerState.STARTED.name().toLowerCase()
                                        : ServerState.STOPPED.name().toLowerCase())
                        .build();
    }

    /**
     * When this is a slave node
     * it returns the connection url for this node
     * and the associated master connection urls in the form of:
     * host:port:streamId
     *
     * @return the slave connection info
     */
    public SlaveConnectionInfo slaveConnectionInfo() {
        if (isMaster())
            throw new IllegalStateException("Unable to determine slave connection info. This is a master node");
        return SlaveConnectionInfo.builder().connectionUrl(subscriber.connectionUrl()).masterUrl(publishMasterUrl)
                        .build();

    }


    /**
     * When this is a master node,
     * it returns the connection url for this node,
     * it's slaves (if any exist) and the responder
     * connection url in the form of:
     * host:port:streamId
     *
     * @return the master connection info
     */
    public MasterConnectionInfo masterConnectionInfo() {
        if (!isMaster())
            throw new IllegalStateException("Unable to determine master connection info. This is a slave node");
        return MasterConnectionInfo.builder().connectionUrl(subscriber.connectionUrl())
                        .responderUrl(responder.connectionUrl()).slaveUrls(new ArrayList<>()).build();
    }

    /**
     * @param args
     */
    public void run(String[] args) {
        JCommander jcmdr = new JCommander(this);

        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            e.printStackTrace();
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            System.exit(1);
        }


        //ensure that the update opType is configured from the command line args
        updateType = UpdateType.valueOf(updateTypeString.toUpperCase());



        if (publishMasterUrl == null && !master)
            throw new IllegalStateException("Please specify a master url or set master to true");

        //allows passing in a media driver for things like unit tests
        //also ensure we don't use a media driver when a directory is specified
        //for a remote one
        if (mediaDriver == null && mediaDriverDirectoryName == null) {
            //length of array * sizeof(float)
            int ipcLength = ArrayUtil.prod(Ints.toArray(shape)) * 4;
            //must be a power of 2
            ipcLength *= 2;
            //padding for NDArrayMessage
            ipcLength += 64;
            //Length in bytes for the SO_RCVBUF, 0 means use OS default. This needs to be larger than Receiver Window.
            System.setProperty("aeron.socket.so_rcvbuf", String.valueOf(ipcLength));
            final MediaDriver.Context mediaDriverCtx = new MediaDriver.Context().threadingMode(ThreadingMode.DEDICATED)
                            .dirsDeleteOnStart(deleteDirectoryOnStart).termBufferSparseFile(false)
                            .ipcTermBufferLength(ipcLength).publicationTermBufferLength(ipcLength)
                            .maxTermBufferLength(ipcLength).conductorIdleStrategy(new BusySpinIdleStrategy())
                            .receiverIdleStrategy(new BusySpinIdleStrategy())
                            .senderIdleStrategy(new BusySpinIdleStrategy());

            mediaDriver = MediaDriver.launchEmbedded(mediaDriverCtx);
            //set the variable since we are using a media driver directly
            mediaDriverDirectoryName = mediaDriver.aeronDirectoryName();
            log.info("Using media driver directory " + mediaDriver.aeronDirectoryName());
        }

        if (aeron == null)
            this.aeron = Aeron.connect(getContext());


        if (master) {
            if (this.callback == null) {
                ParameterServerUpdater updater = null;
                //instantiate with shape instead of just length
                switch (updateType) {
                    case HOGWILD:
                        break;
                    case SYNC:
                        updater = new SynchronousParameterUpdater(new InMemoryUpdateStorage(),
                                        new InMemoryNDArrayHolder(Ints.toArray(shape)), updatesPerEpoch);
                        break;
                    case SOFTSYNC:
                        updater = new SoftSyncParameterUpdater();
                        break;
                    case TIME_DELAYED:
                        break;
                    case CUSTOM:
                        try {
                            updater = (ParameterServerUpdater) Class.forName(System.getProperty(CUSTOM_UPDATE_TYPE))
                                            .newInstance();
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                        break;
                    default:
                        throw new IllegalStateException("Illegal opType of updater");
                }

                callback = new ParameterServerListener(Ints.toArray(shape), updater);
                parameterServerListener = (ParameterServerListener) callback;

            }
            //start an extra daemon for responding to get queries
            ParameterServerListener cast = (ParameterServerListener) callback;
            responder = AeronNDArrayResponder.startSubscriber(aeron, host, port + 1, cast.getUpdater().ndArrayHolder(),
                            streamId + 1);
            log.info("Started responder on master node " + responder.connectionUrl());
        } else {
            String[] publishMasterUrlArr = publishMasterUrl.split(":");
            if (publishMasterUrlArr == null || publishMasterUrlArr.length < 2)
                throw new IllegalStateException("Please specify publish master url as host:port");

            callback = new PublishingListener(
                            String.format("aeron:udp?endpoint=%s:%s", publishMasterUrlArr[0], publishMasterUrlArr[1]),
                            Integer.parseInt(publishMasterUrlArr[2]), getContext());
        }

        log.info("Starting subscriber on " + host + ":" + port + " and stream " + streamId);
        AtomicBoolean running = new AtomicBoolean(true);

        //start a node
        subscriber = AeronNDArraySubscriber.startSubscriber(aeron, host, port, callback, streamId, running);

        while (!subscriber.launched()) {
            LockSupport.parkNanos(100000);
        }

        //send heartbeat to a status server. There will usually be 1 status server per master.
        //Only schedule this if a remote server is available.
        if (CheckSocket.remotePortTaken(statusServerHost, statusServerPort, 10000)) {
            scheduledExecutorService = Executors.newScheduledThreadPool(1);
            final AtomicInteger failCount = new AtomicInteger(0);
            scheduledExecutorService.scheduleAtFixedRate(() -> {
                try {
                    //
                    if (failCount.get() >= 3)
                        return;
                    SubscriberState subscriberState = asState();
                    JSONObject jsonObject = new JSONObject(objectMapper.writeValueAsString(subscriberState));
                    String url = String.format("http://%s:%d/updatestatus/%d", statusServerHost, statusServerPort,
                                    streamId);
                    HttpResponse<String> entity = Unirest.post(url).header("Content-Type", "application/json")
                                    .body(jsonObject).asString();
                } catch (Exception e) {
                    failCount.incrementAndGet();
                    if (failCount.get() >= 3) {
                        log.warn("Failed to send update, shutting down likely?", e);
                    }
                }
            }, 0, heartbeatMs, TimeUnit.MILLISECONDS);

        } else {
            log.info("No status server found. Will not send heartbeats. Specified host was " + statusServerHost
                            + " and port was " + statusServerPort);
        }


        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            close();

        }));

        //set the server for the status of the master and slave nodes
    }


    @Override
    public void close() {
        if (subscriber != null)
            CloseHelper.quietClose(subscriber);
        if (responder != null)
            CloseHelper.quietClose(responder);
        if (scheduledExecutorService != null)
            scheduledExecutorService.shutdown();
    }



    //get a context
    public Aeron.Context getContext() {
        Aeron.Context ctx = new Aeron.Context().publicationConnectionTimeout(-1)
                        .availableImageHandler(AeronUtil::printAvailableImage)
                        .unavailableImageHandler(AeronUtil::printUnavailableImage)
                        .aeronDirectoryName(mediaDriverDirectoryName).keepAliveInterval(100000)
                        .errorHandler(e -> log.error(e.toString(), e));
        return ctx;
    }

    /**
     * Get the master ndarray from the
     * internal {@link NDArrayHolder}
     *
     * @return the master ndarray
     */
    public INDArray getMasterArray() {
        return parameterServerListener.getUpdater().ndArrayHolder().get();
    }


    /**
     * Returns true if the subscriber is launched
     *
     * @return
     */
    public boolean subscriberLaunched() {
        return subscriber.launched();
    }

    public static void main(String[] args) {
        new ParameterServerSubscriber().run(args);
    }
}
