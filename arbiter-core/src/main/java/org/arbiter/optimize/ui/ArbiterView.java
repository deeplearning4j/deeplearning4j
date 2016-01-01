package org.arbiter.optimize.ui;

import io.dropwizard.views.View;

import javax.ws.rs.GET;

/**
 * Created by Alex on 20/12/2015.
 */
public class ArbiterView extends View {

    protected ArbiterView() {
        super("arbiter.ftl");
    }

    @GET
    public String get(){
        return "test2";
    }
}
