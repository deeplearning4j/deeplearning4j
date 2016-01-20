/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.ui.api;

import com.fasterxml.jackson.jaxrs.json.JacksonJsonProvider;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.ui.providers.ObjectMapperProvider;
import org.deeplearning4j.ui.uploads.FileResource;

import javax.ws.rs.*;
import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.Invocation;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * This class handles T-SNE coords upload (in tsv format), and provides output to the browser via JSON
 *
 * @author Adam Gibson
 */
@Path("/api")
@Produces(MediaType.APPLICATION_JSON)
public class ApiResource extends FileResource {
    // TODO: this list should be replaced with HistoryStorage
    private List<String> coords;
    private Client client = ClientBuilder.newClient().register(JacksonJsonProvider.class).register(new ObjectMapperProvider());

    /**
     * The file path for uploads
     *
     * @param filePath the file path for uploads
     */
    public ApiResource(String filePath) {
        super(filePath);
    }
    /**
     * The file path for uploads
     *
     */
    public ApiResource() {
        this(".");
    }




    @POST
    @Path("/update")
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(UrlResource resource) throws IOException {
        String content = client.target(resource.getUrl()).request(MediaType.TEXT_PLAIN_TYPE).get(String.class);
        List<String> testLines = IOUtils.readLines(new ByteArrayInputStream(content.getBytes()));
        this.coords = testLines;
        return Response.ok(coords).build();
    }

    @GET
    @Path("/coords")
    public Response coords() {
        /*
            TODO: here we should have ad-hoc for HistoryStorage.

            For T-SNE we'll probably have no real history though, so it's going to be plain common storage for both internally originated
            2D coordinates, and data uploaded by user
         */
        if(coords.isEmpty()) // TODO: actually we don't need that exception here, just show notification on page
            throw new IllegalStateException("Unable to get coordinates; empty list");

        return Response.ok(coords).build();
    }

    public void setPath(String path) throws IOException {
        coords = FileUtils.readLines(new File(path));
    }


    @Override
    public void handleUpload(File path) {
        /*
            TODO: this code should put new coords into HistoryStorage
         */
        List<String> testLines = null;
        try {
            testLines = FileUtils.readLines(path);
            this.coords = testLines;

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
