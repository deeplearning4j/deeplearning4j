package org.deeplearning4j.ui.renders;

import javax.ws.rs.GET;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

/**
 * @author raver119@gmail.com
 */
public class RendersDropwiz extends RendersResource{
    @GET
    @Produces(MediaType.TEXT_HTML)
    public RenderView get() {
        return new RenderView();
    }
}
