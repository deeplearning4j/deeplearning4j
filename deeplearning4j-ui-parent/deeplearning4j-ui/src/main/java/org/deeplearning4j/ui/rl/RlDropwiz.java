package org.deeplearning4j.ui.rl;

import javax.ws.rs.GET;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

/**
 * @author raver119@gmail.com
 */
public class RlDropwiz extends RlResource {

    @GET
    @Produces(MediaType.TEXT_HTML)
    public RlView getView() {
        return new RlView();
    }
}
