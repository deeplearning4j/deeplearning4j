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


package org.deeplearning4j.ui.module.remote;

import io.netty.handler.codec.http.HttpResponseStatus;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.common.config.DL4JClassLoading;
import org.deeplearning4j.core.storage.*;
import org.deeplearning4j.core.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;

import javax.xml.bind.DatatypeConverter;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 *
 * Used to receive UI updates remotely.
 * Used in conjunction with {@link RemoteUIStatsStorageRouter}, which posts to the UI.
 * UI information is then deserialized and routed to the specified StatsStorageRouter, which may (or may not)
 * be attached to the UI
 *
 * @author Alex Black
 */
@Slf4j
public class RemoteReceiverModule implements UIModule {

    private AtomicBoolean enabled = new AtomicBoolean(false);
    private StatsStorageRouter statsStorage;

    public void setEnabled(boolean enabled) {
        this.enabled.set(enabled);
        if (!enabled) {
            this.statsStorage = null;
        }
    }

    public boolean isEnabled() {
        return enabled.get() && this.statsStorage != null;
    }

    public void setStatsStorage(StatsStorageRouter statsStorage) {
        this.statsStorage = statsStorage;
    }

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/remoteReceive", HttpMethod.POST, (path, rc) -> this.receiveData(rc));
        return Collections.singletonList(r);
    }

    @Override
    public void reportStorageEvents(Collection<StatsStorageEvent> events) {
        //No op

    }

    @Override
    public void onAttach(StatsStorage statsStorage) {
        //No op
    }

    @Override
    public void onDetach(StatsStorage statsStorage) {
        //No op
    }

    @Override
    public List<I18NResource> getInternationalizationResources() {
        return Collections.emptyList();
    }

    private void receiveData(RoutingContext rc) {
        if (!enabled.get()) {
            rc.response().setStatusCode(HttpResponseStatus.FORBIDDEN.code())
                    .end("UI server remote listening is currently disabled. Use UIServer.getInstance().enableRemoteListener()");
            return;
        }

        if (statsStorage == null) {
            rc.response().setStatusCode(HttpResponseStatus.INTERNAL_SERVER_ERROR.code())
                    .end("UI Server remote listener: no StatsStorage instance is set/available to store results");
            return;
        }

        JsonObject jo = rc.getBodyAsJson();
        Map<String,Object> map = jo.getMap();
        String type = (String) map.get("type");
        String dataClass = (String) map.get("class");
        String data = (String) map.get("data");

        if (type == null || dataClass == null || data == null) {
            log.warn("Received incorrectly formatted data from remote listener (has type = " + (type != null)
                            + ", has data class = " + (dataClass != null) + ", has data = " + (data != null) + ")");
            rc.response().setStatusCode(HttpResponseStatus.BAD_REQUEST.code())
                    .end("Received incorrectly formatted data");
            return;
        }

        switch (type.toLowerCase()) {
            case "metadata":
                StorageMetaData meta = getMetaData(dataClass, data);
                if (meta != null) {
                    statsStorage.putStorageMetaData(meta);
                }
                break;
            case "staticinfo":
                Persistable staticInfo = getPersistable(dataClass, data);
                if (staticInfo != null) {
                    statsStorage.putStaticInfo(staticInfo);
                }
                break;
            case "update":
                Persistable update = getPersistable(dataClass, data);
                if (update != null) {
                    statsStorage.putUpdate(update);
                }
                break;
            default:

        }

        rc.response().end();
    }

    private StorageMetaData getMetaData(String dataClass, String content) {
        StorageMetaData meta;
        try {
            Class<?> clazz = DL4JClassLoading.loadClassByName(dataClass);
            if (StorageMetaData.class.isAssignableFrom(clazz)) {
                meta = clazz
                        .asSubclass(StorageMetaData.class)
                        .getDeclaredConstructor()
                        .newInstance();
            } else {
                log.warn("Skipping invalid remote data: class {} in not an instance of {}", dataClass,
                                StorageMetaData.class.getName());
                return null;
            }
        } catch (Exception e) {
            log.warn("Skipping invalid remote data: exception encountered for class {}", dataClass, e);
            return null;
        }

        try {
            byte[] bytes = DatatypeConverter.parseBase64Binary(content);
            meta.decode(bytes);
        } catch (Exception e) {
            log.warn("Skipping invalid remote UI data: exception encountered when deserializing data", e);
            return null;
        }

        return meta;
    }

    private Persistable getPersistable(String dataClass, String content) {
        Persistable persistable;
        try {
            Class<?> clazz = DL4JClassLoading.loadClassByName(dataClass);
            if (Persistable.class.isAssignableFrom(clazz)) {
                persistable = clazz
                        .asSubclass(Persistable.class)
                        .getDeclaredConstructor()
                        .newInstance();
            } else {
                log.warn("Skipping invalid remote data: class {} in not an instance of {}", dataClass,
                                Persistable.class.getName());
                return null;
            }
        } catch (Exception e) {
            log.warn("Skipping invalid remote UI data: exception encountered for class {}", dataClass, e);
            return null;
        }

        try {
            byte[] bytes = DatatypeConverter.parseBase64Binary(content);
            persistable.decode(bytes);
        } catch (Exception e) {
            log.warn("Skipping invalid remote data: exception encountered when deserializing data", e);
            return null;
        }

        return persistable;
    }
}