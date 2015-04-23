package org.deeplearning4j.ui;

import com.google.common.collect.ImmutableMap;
import io.dropwizard.Application;
import io.dropwizard.assets.AssetsBundle;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import io.dropwizard.views.ViewBundle;
import org.apache.commons.io.IOUtils;
import org.springframework.core.io.ClassPathResource;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

/**
 * @author Adam Gibson
 */
public class UiServer extends Application<UIConfiguration> {
    @Override
    public void run(UIConfiguration uiConfiguration, Environment environment) throws Exception {
        environment.jersey().register(new RenderResource());

    }

    @Override
    public void initialize(Bootstrap<UIConfiguration> bootstrap) {
        bootstrap.addBundle(new ViewBundle<UIConfiguration>(){
            @Override
            public ImmutableMap<String, ImmutableMap<String, String>> getViewConfiguration(
                    UIConfiguration arg0) {
                return ImmutableMap.of();
            }
        });

        bootstrap.addBundle(new AssetsBundle());
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
        new UiServer().run(new String[]{
                "server", tmpConfig.getAbsolutePath()
        });
    }
}
