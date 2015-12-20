package org.arbiter.optimize.report.web;

import io.dropwizard.Application;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import io.dropwizard.views.ViewBundle;

/**
 * Created by Alex on 20/12/2015.
 */
public class ArbiterUIServer extends Application<ArbiterUIConfig> {

    public static void main(String[] args) throws Exception {
        String[] str = new String[]{"server", "dropwizard.yml"};
        new ArbiterUIServer().run(str);
    }

    @Override
    public String getName() {
        return "arbiter-ui";
    }

    @Override
    public void initialize(Bootstrap<ArbiterUIConfig> bootstrap) {
        bootstrap.addBundle(new ViewBundle<ArbiterUIConfig>());
    }

    @Override
    public void run(ArbiterUIConfig configuration, Environment environment) {
//        final TestResource resource = new TestResource(
//                configuration.getTemplate(),
//                configuration.getDefaultName()
//        );
        final TestResource2 resource = new TestResource2();
        environment.jersey().register(resource);
    }
}
