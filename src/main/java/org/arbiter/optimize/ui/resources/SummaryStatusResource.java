package org.arbiter.optimize.ui.resources;

import org.arbiter.optimize.ui.components.RenderElements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.Collections;

/**Summary stats: number of completed tasks etc
 */
@Path("/summary")
@Produces(MediaType.APPLICATION_JSON)
public class SummaryStatusResource {
    public static Logger log = LoggerFactory.getLogger(SummaryStatusResource.class);

    private RenderElements renderElements = new RenderElements();

    @GET
    public Response getStatus(){
        log.info("Get with elements: {}",renderElements);
        return Response.ok(renderElements).build();
    }

    @POST
    @Path("/update")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(RenderElements renderElements){
        log.info("Post with new elements: {}",renderElements);
        this.renderElements = renderElements;
        return Response.ok(Collections.singletonMap("status", "ok")).build();
    }

}
