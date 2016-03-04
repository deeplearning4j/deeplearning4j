package org.deeplearning4j.ui.tsne;

import io.dropwizard.views.View;

import javax.ws.rs.GET;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

/**
 * @author raver119@gmail.com
 */
public class TsneDropwiz extends TsneResource {

    /**
     * The file path for uploads
     *
     * @param filePath the file path for uploads
     */
    public TsneDropwiz(String filePath) {
        super(filePath);
    }

    @GET
    @Produces(MediaType.TEXT_HTML)
    public View get() {
        return new TsneView();
    }
}
