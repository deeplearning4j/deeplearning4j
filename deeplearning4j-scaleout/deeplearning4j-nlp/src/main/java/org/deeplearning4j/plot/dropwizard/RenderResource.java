package org.deeplearning4j.plot.dropwizard;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

/**
 * Created by agibsonccc on 10/8/14.
 */
@Path("/render")
@Produces(MediaType.TEXT_HTML)
public class RenderResource {

    @GET
    public RenderView get() {
        return new RenderView();
    }

}
