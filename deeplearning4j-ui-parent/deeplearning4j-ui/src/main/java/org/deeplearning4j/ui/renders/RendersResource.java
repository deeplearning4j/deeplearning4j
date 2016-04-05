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

package org.deeplearning4j.ui.renders;

import org.apache.commons.compress.utils.IOUtils;

import javax.activation.MimetypesFileTypeMap;
import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.StreamingOutput;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collections;
import java.util.Map;

/**
 * Renders filters
 *
 * @author Adam Gibson
 */
@Path("/filters")
@Produces(MediaType.TEXT_HTML)
public class RendersResource {
    private String imagePath = "render.png";

    @POST
    @Path(("/update"))
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(PathUpdate path) {
        this.imagePath = path.getPath();
        return Response.ok(Collections.singletonMap("status","updated path")).build();
    }


    @GET
    @Path("/img")
    @Produces({"image/png"})
    public Response image() {
        if(imagePath == null) {
            throw new WebApplicationException(404);
        }

        final File f = new File(imagePath);

        if (!f.exists()) {
            throw new WebApplicationException(404);
        }

        return Response.ok().entity(new StreamingOutput() {
            @Override
            public void write(OutputStream output) throws IOException, WebApplicationException {
                FileInputStream fis = new FileInputStream(f);
                byte[] bytes = IOUtils.toByteArray(fis);
                fis.close();
                output.write(bytes);
                output.flush();


            }
        }).build();
    }







}
