package org.arbiter.optimize.report.web;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Created by Alex on 20/12/2015.
 */
public class TestRepresentation {

    private long id;

    private String content;

    public TestRepresentation() {
        // Jackson deserialization
    }

    public TestRepresentation(long id, String content) {
        this.id = id;
        this.content = content;
    }

    @JsonProperty
    public long getId() {
        return id;
    }

    @JsonProperty
    public String getContent() {
        return content;
    }

}
