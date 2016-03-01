package org.deeplearning4j.ui.nearestneighbors.word2vec;

import io.dropwizard.views.View;

import javax.ws.rs.GET;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

/**
 * @author raver119@gmail.com
 */
public class NearestNeighborsDropwiz extends NearestNeighborsResource {

    public NearestNeighborsDropwiz(String string) {
        super(string);
    }

    @GET
    @Produces(MediaType.TEXT_HTML)
    public View get() {
        return new NearestNeighborsView();
    }
}
