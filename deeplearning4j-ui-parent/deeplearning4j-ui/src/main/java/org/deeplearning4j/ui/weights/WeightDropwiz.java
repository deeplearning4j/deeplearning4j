package org.deeplearning4j.ui.weights;

import io.dropwizard.views.View;

import javax.ws.rs.GET;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

/**
 * @author raver119@gmail.com
 */
public class WeightDropwiz extends WeightResource {
    @GET
    @Produces(MediaType.TEXT_HTML)
    public View get() {
        return new WeightView(path);
    }
}
