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

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.ui.VertxUIServer;
import org.nd4j.linalg.function.Function;

import java.util.List;

/**
 * Interface for user interface server
 *
 * @author Alex Black
 */
public interface UIServer {

    /**
     * Get (and, initialize if necessary) the UI server.
     * Singleton pattern - all calls to getInstance() will return the same UI instance.
     *
     * @return UI instance for this JVM
     * @throws RuntimeException if the instance has already started in a different mode (multi/single-session)
     */
    static UIServer getInstance() throws RuntimeException {
        return getInstance(false, null);
    }

    /**
     * Get (and, initialize if necessary) the UI server.
     * Singleton pattern - all calls to getInstance() will return the same UI instance.
     *
     * @param multiSession         in multi-session mode, multiple training sessions can be visualized in separate browser tabs.
     *                             <br/>URL path will include session ID as a parameter, i.e.: /train becomes /train/:sessionId
     * @param statsStorageProvider function that returns a StatsStorage containing the given session ID.
     *                             <br/>Use this to auto-attach StatsStorage if an unknown session ID is passed
     *                             as URL path parameter in multi-session mode, or leave it {@code null}.
     * @return UI instance for this JVM
     * @throws RuntimeException if the instance has already started in a different mode (multi/single-session)
     */
    static UIServer getInstance(boolean multiSession, Function<String, StatsStorage> statsStorageProvider) throws RuntimeException {
        return VertxUIServer.getInstance(null, multiSession, statsStorageProvider);
    }

    /**
     * Stop UIServer instance, if already running
     */
    static void stopInstance() {
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
     * Typically used with {@link org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter}, which will send information
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
     * Stop/shut down the UI server.
     */
    void stop();

}
