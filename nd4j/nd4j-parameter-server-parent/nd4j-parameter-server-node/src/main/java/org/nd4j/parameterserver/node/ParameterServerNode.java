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

package org.nd4j.parameterserver.node;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.agrona.CloseHelper;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.parameterserver.ParameterServerListener;
import org.nd4j.parameterserver.ParameterServerSubscriber;
import org.nd4j.parameterserver.status.play.InMemoryStatusStorage;
import org.nd4j.parameterserver.status.play.StatusServer;
import play.server.Server;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Integrated node for running
 * the parameter server.
 * This includes the status server,
 * media driver, and parameter server subscriber
 *
 * @author Adam Gibson
 */
@Slf4j
@NoArgsConstructor
@Data
public class ParameterServerNode implements AutoCloseable {
    private Server server;
    private ParameterServerSubscriber[] subscriber;
    private MediaDriver mediaDriver;
    private Aeron aeron;
    private int statusPort;
    private int numWorkers;



    /**
     *
     * @param mediaDriver the media driver to sue for communication
     * @param statusPort the port for the server status
     */
    public ParameterServerNode(MediaDriver mediaDriver, int statusPort) {
        this(mediaDriver, statusPort, Runtime.getRuntime().availableProcessors());

    }

    /**
     *
     * @param mediaDriver the media driver to sue for communication
     * @param statusPort the port for the server status
     */
    public ParameterServerNode(MediaDriver mediaDriver, int statusPort, int numWorkers) {
        this.mediaDriver = mediaDriver;
        this.statusPort = statusPort;
        this.numWorkers = numWorkers;
        subscriber = new ParameterServerSubscriber[numWorkers];

    }


    /**
     * Pass in the media driver used for communication
     * and a defualt status port of 9000
     * @param mediaDriver
     */
    public ParameterServerNode(MediaDriver mediaDriver) {
        this(mediaDriver, 9000);
    }

    /**
     * Run this node with the given args
     * These args are the same ones
     * that a {@link ParameterServerSubscriber} takes
     * @param args the arguments for the {@link ParameterServerSubscriber}
     */
    public void runMain(String[] args) {
        server = StatusServer.startServer(new InMemoryStatusStorage(), statusPort);
        if (mediaDriver == null)
            mediaDriver = MediaDriver.launchEmbedded();
        log.info("Started media driver with aeron directory " + mediaDriver.aeronDirectoryName());
        //cache a reference to the first listener.
        //The reason we do this is to share an updater and listener across *all* subscribers
        //This will create a shared pool of subscribers all updating the same "server".
        //This will simulate a shared pool but allow an accumulative effect of anything
        //like averaging we try.
        NDArrayCallback parameterServerListener = null;
        ParameterServerListener cast = null;
        for (int i = 0; i < numWorkers; i++) {
            subscriber[i] = new ParameterServerSubscriber(mediaDriver);
            //ensure reuse of aeron wherever possible
            if (aeron == null)
                aeron = Aeron.connect(getContext(mediaDriver));
            subscriber[i].setAeron(aeron);
            List<String> multiArgs = new ArrayList<>(Arrays.asList(args));
            if (multiArgs.contains("-id")) {
                int streamIdIdx = multiArgs.indexOf("-id") + 1;
                int streamId = Integer.parseInt(multiArgs.get(streamIdIdx)) + i;
                multiArgs.set(streamIdIdx, String.valueOf(streamId));
            } else if (multiArgs.contains("--streamId")) {
                int streamIdIdx = multiArgs.indexOf("--streamId") + 1;
                int streamId = Integer.parseInt(multiArgs.get(streamIdIdx)) + i;
                multiArgs.set(streamIdIdx, String.valueOf(streamId));
            }


            if (i == 0) {
                subscriber[i].run(multiArgs.toArray(new String[args.length]));
                parameterServerListener = subscriber[i].getCallback();
                cast = subscriber[i].getParameterServerListener();
            } else {
                //note that we set both the callback AND the listener here
                subscriber[i].setCallback(parameterServerListener);
                subscriber[i].setParameterServerListener(cast);
                //now run the callback initialized with this callback instead
                //in the run method it will use this reference instead of creating it
                //itself
                subscriber[i].run(multiArgs.toArray(new String[args.length]));
            }


        }

    }



    /**
     * Returns true if all susbcribers in the
     * subscriber pool have been launched
     * @return
     */
    public boolean subscriberLaunched() {
        boolean launched = true;
        for (int i = 0; i < numWorkers; i++) {
            launched = launched && subscriber[i].subscriberLaunched();
        }

        return launched;
    }


    /**
     * Stop the server
     * @throws Exception
     */
    @Override
    public void close() throws Exception {
        if (subscriber != null) {
            for (int i = 0; i < subscriber.length; i++) {
                if (subscriber[i] != null) {
                    subscriber[i].close();
                }
            }
        }
        if (server != null)
            server.stop();
        if (mediaDriver != null)
            CloseHelper.quietClose(mediaDriver);
        if (aeron != null)
            CloseHelper.quietClose(aeron);

    }



    private static Aeron.Context getContext(MediaDriver mediaDriver) {
        return new Aeron.Context()
                        .availableImageHandler(AeronUtil::printAvailableImage)
                        .unavailableImageHandler(AeronUtil::printUnavailableImage)
                        .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                        .errorHandler(e -> log.error(e.toString(), e));
    }


    public static void main(String[] args) {
        new ParameterServerNode().runMain(args);
    }

}
