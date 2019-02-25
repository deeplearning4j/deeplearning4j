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

package org.deeplearning4j.ui.module.defaultModule;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;
import org.nd4j.linalg.function.Supplier;

import java.io.File;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static play.mvc.Results.ok;
import static play.mvc.Results.redirect;

/**
 * Landing page - i.e., "/" route
 * @author Alex Black
 */
public class DefaultModule implements UIModule {
    private final boolean multiSession;
    private final Supplier<String> addressSupplier;

    public DefaultModule() {
        this(false, null);
    }

    /**
     *
     * @param multiSession multi-session mode
     * @param addressSupplier supplier for server address (server address in PlayUIServer gets initialized after modules)
     */
    public DefaultModule(boolean multiSession, Supplier<String> addressSupplier) {
        this.multiSession = multiSession;
        this.addressSupplier = addressSupplier;
    }

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        //TODO
        //        Route r = new Route("/", HttpMethod.GET, FunctionType.Supplier, () -> ok(org.deeplearning4j.ui.views.html.defaultPage.DefaultPage.apply()));
        Route r = multiSession ? new Route("/", HttpMethod.GET, FunctionType.Supplier,
                () -> ok("UI server is in multi-session mode. You can find a training session at "
                        + addressSupplier.get() + "/train/:sessionId (See console for attached training session ID.)"))
                : new Route("/", HttpMethod.GET, FunctionType.Supplier, () -> redirect("/train/overview"));

        return Collections.singletonList(r);
    }

    @Override
    public void reportStorageEvents(Collection<StatsStorageEvent> events) {

    }

    @Override
    public void onAttach(StatsStorage statsStorage) {

    }

    @Override
    public void onDetach(StatsStorage statsStorage) {

    }

    @Override
    public List<I18NResource> getInternationalizationResources() {
        return Collections.emptyList();
    }
}
