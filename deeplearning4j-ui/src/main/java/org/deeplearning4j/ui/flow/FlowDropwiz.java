package org.deeplearning4j.ui.flow;

import javax.ws.rs.GET;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

/**
 * @author raver119@gmail.com
 */
public class FlowDropwiz extends FlowResource {

    @GET
    @Produces(MediaType.TEXT_HTML)
    public FlowView getView() {
        return new FlowView();
    }
}
