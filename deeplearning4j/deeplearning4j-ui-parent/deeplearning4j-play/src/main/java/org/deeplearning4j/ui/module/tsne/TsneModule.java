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

package org.deeplearning4j.ui.module.tsne;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;
import play.libs.Json;
import play.mvc.Http;
import play.mvc.Result;
import play.mvc.Results;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static play.mvc.Controller.request;
import static play.mvc.Results.badRequest;
import static play.mvc.Results.ok;

/**
 * Created by Alex on 25/10/2016.
 */
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
        Route r1 = new Route("/tsne", HttpMethod.GET, FunctionType.Supplier,
                        () -> ok(org.deeplearning4j.ui.views.html.tsne.Tsne.apply()));
        Route r2 = new Route("/tsne/sessions", HttpMethod.GET, FunctionType.Supplier, this::listSessions);
        Route r3 = new Route("/tsne/coords/:sid", HttpMethod.GET, FunctionType.Function, this::getCoords);
        Route r4 = new Route("/tsne/upload", HttpMethod.POST, FunctionType.Supplier, this::uploadFile);
        Route r5 = new Route("/tsne/post/:sid", HttpMethod.POST, FunctionType.Function, this::postFile);
        return Arrays.asList(r1, r2, r3, r4, r5);
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

    private Result listSessions() {
        List<String> list = new ArrayList<>(knownSessionIDs.keySet());
        if (uploadedFileLines != null) {
            list.add(UPLOADED_FILE);
        }
        return Results.ok(Json.toJson(list));
    }

    private Result getCoords(String sessionId) {
        if (UPLOADED_FILE.equals(sessionId) && uploadedFileLines != null) {
            return Results.ok(Json.toJson(uploadedFileLines));
        } else if (knownSessionIDs.containsKey(sessionId)) {
            return Results.ok(Json.toJson(knownSessionIDs.get(sessionId)));
        } else {
            return Results.ok();
        }
    }

    private Result uploadFile() {
        Http.MultipartFormData body = request().body().asMultipartFormData();
        List<Http.MultipartFormData.FilePart> fileParts = body.getFiles();

        if (fileParts.isEmpty()) {
            return badRequest("No file uploaded");
        }

        Http.MultipartFormData.FilePart uploadedFile = fileParts.get(0);

        String fileName = uploadedFile.getFilename();
        String contentType = uploadedFile.getContentType();
        File file = uploadedFile.getFile();

        try {
            uploadedFileLines = FileUtils.readLines(file);
        } catch (IOException e) {
            return badRequest("Could not read from uploaded file");
        }

        return ok("File uploaded: " + fileName + ", " + contentType + ", " + file);
    }

    private Result postFile(String sid) {
        //        System.out.println("POST FILE CALLED: " + sid);
        Http.MultipartFormData body = request().body().asMultipartFormData();
        List<Http.MultipartFormData.FilePart> fileParts = body.getFiles();

        if (fileParts.isEmpty()) {
            //            System.out.println("**** NO FILE ****");
            return badRequest("No file uploaded");
        }

        Http.MultipartFormData.FilePart uploadedFile = fileParts.get(0);

        String fileName = uploadedFile.getFilename();
        String contentType = uploadedFile.getContentType();
        File file = uploadedFile.getFile();

        List<String> lines;
        try {
            // Set to uploadedFileLines as well, as the TSNE UI doesn't allow to properly select Sessions yet
            lines = uploadedFileLines = FileUtils.readLines(file);
        } catch (IOException e) {
            //            System.out.println("**** COULD NOT READ FILE ****");
            return badRequest("Could not read from uploaded file");
        }

        knownSessionIDs.put(sid, lines);


        return ok("File uploaded: " + fileName + ", " + contentType + ", " + file);
    }
}
