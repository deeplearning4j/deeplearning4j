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

package org.deeplearning4j.ui.api;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.ui.play.PlayUIServer;

import java.util.List;

/**
 * Interface for user interface server
 *
 * @author Alex Black
 */
public abstract class UIServer {

    private static UIServer uiServer;

    /**
     * Get (and, initialize if necessary) the UI server.
     * Singleton pattern - all calls to getInstance() will return the same UI instance.
     *
     * @return UI instance for this JVM
     */
    public static synchronized UIServer getInstance() {
        if (uiServer == null) {
            PlayUIServer playUIServer = new PlayUIServer();
            playUIServer.runMain(new String[] {"--uiPort", String.valueOf(PlayUIServer.DEFAULT_UI_PORT)});
            uiServer = playUIServer;
        }
        return uiServer;
    }

    /**
     * Get the address of the UI
     *
     * @return Address of the UI
     */
    public abstract String getAddress();

    /**
     * Get the current port for the UI
     */
    public abstract int getPort();

    /**
     * Attach the given StatsStorage instance to the UI, so the data can be visualized
     * @param statsStorage    StatsStorage instance to attach to the UI
     */
    public abstract void attach(StatsStorage statsStorage);

    /**
     * Detach the specified StatsStorage instance from the UI
     * @param statsStorage    StatsStorage instance to detach. If not attached: no op.
     */
    public abstract void detach(StatsStorage statsStorage);

    /**
     * Check whether the specified StatsStorage instance is attached to the UI instance
     *
     * @param statsStorage    StatsStorage instance to attach
     * @return True if attached
     */
    public abstract boolean isAttached(StatsStorage statsStorage);

    /**
     * @return A list of all StatsStorage instances currently attached
     */
    public abstract List<StatsStorage> getStatsStorageInstances();

    /**
     * Enable the remote listener functionality, storing all data in memory, and attaching the instance to the UI.
     * Typically used with {@link org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter}, which will send information
     * remotely to this UI instance
     *
     * @see #enableRemoteListener(StatsStorageRouter, boolean)
     */
    public abstract void enableRemoteListener();

    /**
     * Enable the remote listener functionality, storing the received results in the specified StatsStorageRouter.
     * If the StatsStorageRouter is a {@link StatsStorage} instance, it may (optionally) be attached to the UI,
     * as if {@link #attach(StatsStorage)} was called on it.
     *
     * @param statsStorage    StatsStorageRouter to post the received results to
     * @param attach          Whether to attach the given StatsStorage instance to the UI server
     */
    public abstract void enableRemoteListener(StatsStorageRouter statsStorage, boolean attach);

    /**
     * Disable the remote listener functionality (disabled by default)
     */
    public abstract void disableRemoteListener();

    /**
     * @return  Whether the remote listener functionality is currently enabled
     */
    public abstract boolean isRemoteListenerEnabled();

    /**
     * Stop/shut down the UI server.
     */
    public abstract void stop();

}
