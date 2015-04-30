package org.deeplearning4j.ui.api;

import org.apache.commons.io.FileUtils;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
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
