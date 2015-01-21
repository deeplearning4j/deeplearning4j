package org.deeplearning4j.plot.dropwizard;

import io.dropwizard.Application;
import io.dropwizard.assets.AssetsBundle;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import io.dropwizard.views.ViewBundle;
import org.apache.commons.compress.utils.IOUtils;
import org.slf4j.Logger;
import org.springframework.core.io.ClassPathResource;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

/**
 *
 * @author Adam Gibson
 */
public class RenderApplication extends Application<ApiConfiguration> {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(RenderApplication.class);

    @Override
    public void initialize(Bootstrap<ApiConfiguration> apiConfigurationBootstrap) {
        apiConfigurationBootstrap.addBundle(new ViewBundle());
        apiConfigurationBootstrap.addBundle(new AssetsBundle());

    }

    @Override
    public void run(ApiConfiguration apiConfiguration, Environment environment) throws Exception {
        environment.jersey().register(new ApiResource("coords.csv"));
        environment.jersey().register(new RenderResource());


    }

    public static void main(String[] args) throws Exception {
        ClassPathResource resource = new ClassPathResource("/render/dropwizard.yml");
        InputStream is = resource.getInputStream();
        File tmpConfig = new File("dropwizard-render.yml");
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(tmpConfig));
        IOUtils.copy(is, bos);
        bos.flush();
        bos.close();
        is.close();
        tmpConfig.deleteOnExit();
        new RenderApplication().run(new String[]{
                "server", tmpConfig.getAbsolutePath()
        });
    }
}
