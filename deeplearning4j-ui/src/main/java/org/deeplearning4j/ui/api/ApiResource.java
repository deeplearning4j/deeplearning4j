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
import org.deeplearning4j.ui.providers.ObjectMapperProvider;

import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.Invocation;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * @author Adam Gibson
 */
@Path("/api")
@Produces(MediaType.APPLICATION_JSON)
public class ApiResource  {
    private List<String> coords;
    private Client client = ClientBuilder.newClient().register(JacksonJsonProvider.class).register(new ObjectMapperProvider());


    @POST
    @Path("/update")
    public Response update(UrlResource resource) {

        if(coords.isEmpty())
            throw new IllegalStateException("Unable to get coordinates; empty");
        List<String> list = client.target(resource.getUrl()).request(MediaType.TEXT_PLAIN_TYPE).get(List.class);
        this.coords = list;
        return Response.ok(coords).build();
    }

    @GET
    @Path("/coords")
    public Response coords() {

        if(coords.isEmpty())
            throw new IllegalStateException("Unable to get coordinates; empty");

        return Response.ok(coords).build();
    }

    public void setPath(String path) throws IOException {
        coords = FileUtils.readLines(new File(path));
    }


}
