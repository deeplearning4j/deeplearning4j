/* ******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

package org.deeplearning4j.ui.api;

import io.vertx.core.Promise;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.core.storage.StatsStorageRouter;
import org.deeplearning4j.core.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.ui.VertxUIServer;
import org.nd4j.common.function.Function;

import java.util.List;

/**
 * Interface for user interface server
 *
 * @author Alex Black
 */
public interface UIServer {

    /**
     * Get (and, initialize if necessary) the UI server. This synchronous function will wait until the server started.
     * Singleton pattern - all calls to getInstance() will return the same UI instance.
     *
     * @return UI instance for this JVM
     * @throws DL4JException if UI server failed to start;
     * if the instance has already started in a different mode (multi/single-session);
     * if interrupted while waiting for completion
     */
    static UIServer getInstance() throws DL4JException {
        if (VertxUIServer.getInstance() != null && !VertxUIServer.getInstance().isStopped()) {
            return VertxUIServer.getInstance();
        } else {
            return getInstance(false, null);
        }
    }

    /**
     * Get (and, initialize if necessary) the UI server. This synchronous function will wait until the server started.
     * Singleton pattern - all calls to getInstance() will return the same UI instance.
     *
     * @param multiSession         in multi-session mode, multiple training sessions can be visualized in separate browser tabs.
     *                             <br/>URL path will include session ID as a parameter, i.e.: /train becomes /train/:sessionId
     * @param statsStorageProvider function that returns a StatsStorage containing the given session ID.
     *                             <br/>Use this to auto-attach StatsStorage if an unknown session ID is passed
     *                             as URL path parameter in multi-session mode, or leave it {@code null}.
     * @return UI instance for this JVM
     * @throws DL4JException if UI server failed to start;
     * if the instance has already started in a different mode (multi/single-session);
     * if interrupted while waiting for completion
     */
    static UIServer getInstance(boolean multiSession, Function<String, StatsStorage> statsStorageProvider)
            throws DL4JException {
        return VertxUIServer.getInstance(null, multiSession, statsStorageProvider);
    }

    /**
     * Stop UIServer instance, if already running
     */
    static void stopInstance() throws Exception {
        VertxUIServer.stopInstance();
    }

    boolean isStopped();

    /**
     * Check if the instance initialized with one of the factory methods
     * ({@link #getInstance()} or {@link #getInstance(boolean, Function)}) is in multi-session mode
     *
     * @return {@code true} if the instance is in multi-session
     */
    boolean isMultiSession();

    /**
     * Get the address of the UI
     *
     * @return Address of the UI
     */
    String getAddress();

    /**
     * Get the current port for the UI
     */
    int getPort();

    /**
     * Attach the given StatsStorage instance to the UI, so the data can be visualized
     *
     * @param statsStorage StatsStorage instance to attach to the UI
     */
    void attach(StatsStorage statsStorage);

    /**
     * Detach the specified StatsStorage instance from the UI
     *
     * @param statsStorage StatsStorage instance to detach. If not attached: no op.
     */
    void detach(StatsStorage statsStorage);

    /**
     * Check whether the specified StatsStorage instance is attached to the UI instance
     *
     * @param statsStorage StatsStorage instance to attach
     * @return True if attached
     */
    boolean isAttached(StatsStorage statsStorage);

    /**
     * @return A list of all StatsStorage instances currently attached
     */
    List<StatsStorage> getStatsStorageInstances();

    /**
     * Enable the remote listener functionality, storing all data in memory, and attaching the instance to the UI.
     * Typically used with {@link RemoteUIStatsStorageRouter}, which will send information
     * remotely to this UI instance
     *
     * @see #enableRemoteListener(StatsStorageRouter, boolean)
     */
    void enableRemoteListener();

    /**
     * Enable the remote listener functionality, storing the received results in the specified StatsStorageRouter.
     * If the StatsStorageRouter is a {@link StatsStorage} instance, it may (optionally) be attached to the UI,
     * as if {@link #attach(StatsStorage)} was called on it.
     *
     * @param statsStorage StatsStorageRouter to post the received results to
     * @param attach       Whether to attach the given StatsStorage instance to the UI server
     */
    void enableRemoteListener(StatsStorageRouter statsStorage, boolean attach);

    /**
     * Disable the remote listener functionality (disabled by default)
     */
    void disableRemoteListener();

    /**
     * @return Whether the remote listener functionality is currently enabled
     */
    boolean isRemoteListenerEnabled();

    /**
     * Stop/shut down the UI server. This synchronous function should wait until the server is stopped.
     * @throws InterruptedException if the current thread is interrupted while waiting
     */
    void stop() throws InterruptedException;

    /**
     * Stop/shut down the UI server.
     * This asynchronous function should immediately return, and notify the callback {@link Promise} on completion:
     * either call {@link Promise#complete} or {@link io.vertx.core.Promise#fail}.
     * @param stopCallback callback {@link Promise} to notify on completion
     */
    void stopAsync(Promise<Void> stopCallback);

    /**
     * Get shutdown hook of UI server, that will stop the server when the Runtime is stopped.
     * You may de-register this shutdown hook with {@link Runtime#removeShutdownHook(Thread)},
     * and add your own hook with {@link Runtime#addShutdownHook(Thread)}
     * @return shutdown hook
     */
    static Thread getShutdownHook() {
        return VertxUIServer.getShutdownHook();
    };
}
