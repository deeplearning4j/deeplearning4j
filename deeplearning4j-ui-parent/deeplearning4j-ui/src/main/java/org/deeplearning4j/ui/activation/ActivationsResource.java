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

package org.deeplearning4j.ui.activation;

import org.apache.commons.compress.utils.IOUtils;
import org.canova.api.util.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.activation.MimetypesFileTypeMap;
import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.StreamingOutput;
import java.io.*;
import java.util.Collections;

/**
 * Activation filters
 *
 * @author Adam Gibson
 */
@Path("/activations")
@Produces(MediaType.TEXT_HTML)
public class ActivationsResource {
    private String imagePath = "activations.png";

    private static Logger log = LoggerFactory.getLogger(ActivationsResource.class);


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
            imagePath = "activations.png";
        }

         File fx = new File(imagePath);


        if (!fx.exists()) {
            try {
                ClassPathResource resource = new ClassPathResource("/404.img");
                fx = resource.getFile();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        final File f = fx;


        return Response.ok().entity(new StreamingOutput(){
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
