package org.arbiter.optimize.ui.resources;

import org.arbiter.optimize.ui.components.RenderElements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.Collections;

@Path("/config")
@Produces(MediaType.APPLICATION_JSON)
public class ConfigResource {

    public static final Logger log = LoggerFactory.getLogger(ConfigResource.class);
    private RenderElements renderElements = new RenderElements();

    @GET
    public Response getConfig(){
        log.info("GET for config with current config: {}");
        return Response.ok(renderElements).build();
    }

    @POST
    @Path("/update")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(RenderElements renderElements){
        log.info("POST for config with new elements: {}",renderElements);
        this.renderElements = renderElements;
        return Response.ok(Collections.singletonMap("status", "ok")).build();
    }

}
