package org.deeplearning4j.ui.defaults;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

/**
 * Container for the default root page, that's available for user when he open his browser and navigate to dl4j-ui
 *
 * @author raver119@gmail.com
 */
@Path("/")
@Produces(MediaType.TEXT_HTML)
public class DefaultResource {
    /**
     * So, basic idea here: main page, with links to available activities within this UIServer
     *
     */


    /**
     * This method produces default page, aka index page
     *
     * @return
     */
    @GET
    @Produces(MediaType.TEXT_HTML)
    public DefaultView get() {
        return new DefaultView();
    }

    @GET
    @Path("/whatsup")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getWhatsUp() {
        /*
         TODO: this method should return the list of available processes reported to this UIServer. And this response will be used on client-side
         Actual mechanism, probably, will use HistoryStorage, and it's ability to track active objects within
          */
        return Response.ok().build();
    }
}
