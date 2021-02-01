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

package org.deeplearning4j.ui.module.tsne;

import io.netty.handler.codec.http.HttpResponseStatus;
import io.vertx.ext.web.FileUpload;
import io.vertx.ext.web.RoutingContext;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.core.storage.StatsStorageEvent;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class TsneModule implements UIModule {
    private static final String UPLOADED_FILE = "UploadedFile";

    private Map<String, List<String>> knownSessionIDs = Collections.synchronizedMap(new LinkedHashMap<>());
    private List<String> uploadedFileLines = null;

    public TsneModule() {
    }

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        Route r1 = new Route("/tsne", HttpMethod.GET, (path, rc) -> rc.response().sendFile("templates/Tsne.html"));
        Route r2 = new Route("/tsne/sessions", HttpMethod.GET, (path, rc) -> this.listSessions(rc));
        Route r3 = new Route("/tsne/coords/:sid", HttpMethod.GET, (path, rc) -> this.getCoords(path.get(0), rc));
        Route r4 = new Route("/tsne/upload", HttpMethod.POST, (path, rc) -> this.uploadFile(rc));
        Route r5 = new Route("/tsne/post/:sid", HttpMethod.POST, (path, rc) -> this.postFile(path.get(0), rc));
        return Arrays.asList(r1, r2, r3, r4, r5);
    }

    @Override
    public void reportStorageEvents(Collection<StatsStorageEvent> events) {
        //No-op
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

    private String asJson(Object o){
        try{
            return JsonMappers.getMapper().writeValueAsString(o);
        } catch (JsonProcessingException e){
            throw new RuntimeException("Error converting object to JSON", e);
        }
    }

    private void listSessions(RoutingContext rc) {
        List<String> list = new ArrayList<>(knownSessionIDs.keySet());
        if (uploadedFileLines != null) {
            list.add(UPLOADED_FILE);
        }
        rc.response()
                .putHeader("content-type", "application/json")
                .end(asJson(list));
    }

    private void getCoords(String sessionId, RoutingContext rc) {
        if (UPLOADED_FILE.equals(sessionId) && uploadedFileLines != null) {
            rc.response()
                    .putHeader("content-type", "application/json")
                    .end(asJson(uploadedFileLines));
        } else if (knownSessionIDs.containsKey(sessionId)) {
            rc.response().putHeader("content-type", "application/json")
                    .end(asJson(knownSessionIDs.get(sessionId)));
        } else {
            rc.response().end();
        }
    }

    private void uploadFile(RoutingContext rc) {
        postFile(null, rc);
    }

    private void postFile(String sid, RoutingContext rc) {
        Set<FileUpload> files = rc.fileUploads();
        if(files == null || files.isEmpty()){
            rc.response().end();
            return;
        }

        FileUpload u = files.iterator().next();
        File f = new File(u.uploadedFileName());
        List<String> lines;
        try {
            lines = FileUtils.readLines(f, StandardCharsets.UTF_8);
        } catch (IOException e) {
            rc.response().setStatusCode(HttpResponseStatus.BAD_REQUEST.code()).end("Could not read from uploaded file");
            return;
        }

        if(sid == null){
            uploadedFileLines = lines;
        } else {
            knownSessionIDs.put(sid, lines);
        }
        rc.response().end("File uploaded: " + u.fileName() + ", " + u.contentType());
    }
}
