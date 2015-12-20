package org.arbiter.optimize.report.web;

import io.dropwizard.views.View;

import javax.ws.rs.GET;

/**
 * Created by Alex on 20/12/2015.
 */
public class TestView extends View {

    protected TestView() {
        super("arbiterui.ftl");
    }

    @GET
    public String get(){
        return "test2";
    }
}
