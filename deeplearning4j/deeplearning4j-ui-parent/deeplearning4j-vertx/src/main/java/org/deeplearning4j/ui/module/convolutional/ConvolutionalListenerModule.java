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

package org.deeplearning4j.ui.module.convolutional;

import io.vertx.core.buffer.Buffer;
import io.vertx.ext.web.RoutingContext;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.core.storage.Persistable;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.core.storage.StatsStorageEvent;
import org.deeplearning4j.core.storage.StatsStorageListener;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;
import org.deeplearning4j.ui.model.weights.ConvolutionListenerPersistable;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * Used for plotting results from the ConvolutionalIterationListener
 *
 * @author Alex Black
 */
@Slf4j
public class ConvolutionalListenerModule implements UIModule {

    private static final String TYPE_ID = "ConvolutionalListener";

    private StatsStorage lastStorage;
    private String lastSessionID;
    private String lastWorkerID;
    private long lastTimeStamp;

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/activations", HttpMethod.GET, (path, rc) -> rc.response().sendFile("templates/Activations.html"));
        Route r2 = new Route("/activations/data", HttpMethod.GET, (path, rc) -> this.getImage(rc));

        return Arrays.asList(r, r2);
    }

    @Override
    public synchronized void reportStorageEvents(Collection<StatsStorageEvent> events) {
        for (StatsStorageEvent sse : events) {
            if (TYPE_ID.equals(sse.getTypeID())
                            && sse.getEventType() == StatsStorageListener.EventType.PostStaticInfo) {
                if (sse.getTimestamp() > lastTimeStamp) {
                    lastStorage = sse.getStatsStorage();
                    lastSessionID = sse.getSessionID();
                    lastWorkerID = sse.getWorkerID();
                    lastTimeStamp = sse.getTimestamp();
                }
            }
        }
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

    private void getImage(RoutingContext rc) {
        if (lastTimeStamp > 0 && lastStorage != null) {
            Persistable p = lastStorage.getStaticInfo(lastSessionID, TYPE_ID, lastWorkerID);
            if (p instanceof ConvolutionListenerPersistable) {
                ConvolutionListenerPersistable clp = (ConvolutionListenerPersistable) p;
                BufferedImage bi = clp.getImg();
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                try {
                    ImageIO.write(bi, "png", baos);
                } catch (IOException e) {
                    log.warn("Error displaying image", e);
                }

                rc.response()
                        .putHeader("content-type", "image/png")
                        .end(Buffer.buffer(baos.toByteArray()));
            } else {
                rc.response()
                        .putHeader("content-type", "image/png")
                        .end(Buffer.buffer(new byte[0]));
            }
        } else {
            rc.response()
                    .putHeader("content-type", "image/png")
                    .end(Buffer.buffer(new byte[0]));
        }
    }
}
