package org.deeplearning4j.ui.activation;

import javax.ws.rs.GET;

/**
 * @author raver119@gmail.com
 */
public class ActivationsDropwiz extends ActivationsResource {

    @GET
    public RenderView get() {
        return new RenderView();
    }

}
