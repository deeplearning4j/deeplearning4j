package org.deeplearning4j.ui.defaults;

import javax.ws.rs.GET;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

/**
 * @author raver119@gmail.com
 */
public class DefaultDropwiz extends DefaultResource {

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

}
