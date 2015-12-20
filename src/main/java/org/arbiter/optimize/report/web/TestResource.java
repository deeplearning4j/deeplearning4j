package org.arbiter.optimize.report.web;

import com.codahale.metrics.annotation.Timed;
import com.google.common.base.Optional;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;
import java.util.concurrent.atomic.AtomicLong;

@Path("/arbiter-ui")
@Produces(MediaType.APPLICATION_JSON)
public class TestResource {

    private final String template;
    private final String defaultName;
    private final AtomicLong counter;

    public TestResource(String template, String defaultName) {
        this.template = template;
        this.defaultName = defaultName;
        this.counter = new AtomicLong();
    }

//    @GET
//    @Timed
//    public TestRepresentation sayHello(@QueryParam("name") Optional<String> name) {
//        final String value = String.format(template, name.or(defaultName));
//        return new TestRepresentation(counter.incrementAndGet(), value);
//    }

    @GET
    @Timed
    public TestRepresentation sayHello() {
        final String value = "TestText";    //String.format(template, name.or(defaultName));
        return new TestRepresentation(counter.incrementAndGet(), value);
    }

}
