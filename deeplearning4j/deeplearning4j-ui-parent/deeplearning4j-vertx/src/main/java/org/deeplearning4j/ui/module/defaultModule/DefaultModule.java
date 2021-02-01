/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.ui.module.defaultModule;

import io.netty.handler.codec.http.HttpResponseStatus;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.core.storage.StatsStorageEvent;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * Landing page - i.e., "/" route
 * @author Alex Black
 */
public class DefaultModule implements UIModule {
    private final boolean multiSession;

    public DefaultModule() {
        this(false);
    }

    /**
     *
     * @param multiSession multi-session mode
     */
    public DefaultModule(boolean multiSession) {
        this.multiSession = multiSession;
    }

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/", HttpMethod.GET,  (params, rc) -> rc.response()
                .putHeader("location", "/train" + (multiSession ? "" : "/overview"))
                .setStatusCode(HttpResponseStatus.FOUND.code())
                .end()
        );

        return Collections.singletonList(r);
    }

    @Override
    public void reportStorageEvents(Collection<StatsStorageEvent> events) {
        //Nop-op
    }

    @Override
    public void onAttach(StatsStorage statsStorage) {
        //No-op
    }

    @Override
    public void onDetach(StatsStorage statsStorage) {
        //No-op
    }

    @Override
    public List<I18NResource> getInternationalizationResources() {
        return Collections.emptyList();
    }
}
