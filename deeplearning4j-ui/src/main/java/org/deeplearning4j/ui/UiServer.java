package org.deeplearning4j.ui;

import com.google.common.collect.ImmutableMap;
import io.dropwizard.Application;
import io.dropwizard.assets.AssetsBundle;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import io.dropwizard.views.ViewBundle;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.ui.nearestneighbors.NearestNeighborsResource;
import org.deeplearning4j.ui.renders.RendersResource;
import org.deeplearning4j.ui.tsne.TsneResource;
import org.deeplearning4j.ui.uploads.FileResource;
import org.deeplearning4j.ui.weights.WeightResource;
import org.eclipse.jetty.servlets.CrossOriginFilter;
import org.glassfish.jersey.media.multipart.MultiPartFeature;
import org.springframework.core.io.ClassPathResource;

import javax.servlet.DispatcherType;
import javax.servlet.FilterRegistration;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.EnumSet;

/**
 * @author Adam Gibson
 */
public class UiServer extends Application<UIConfiguration> {

    private UIConfiguration conf;

    @Override
    public void run(UIConfiguration uiConfiguration, Environment environment) throws Exception {
        this.conf = uiConfiguration;
        environment.jersey().register(MultiPartFeature.class);
        environment.jersey().register(new TsneResource());
        environment.jersey().register(new NearestNeighborsResource(conf.getUploadPath()));
        environment.jersey().register(new WeightResource());
        environment.jersey().register(new RendersResource());
        configureCors(environment);
    }

    @Override
    public void initialize(Bootstrap<UIConfiguration> bootstrap) {
        bootstrap.addBundle(new ViewBundle<UIConfiguration>() {
            @Override
            public ImmutableMap<String, ImmutableMap<String, String>> getViewConfiguration(
                    UIConfiguration arg0) {
                return ImmutableMap.of();
            }
        });
        bootstrap.addBundle(new AssetsBundle());
    }


    public static void main(String[] args) throws Exception {
        ClassPathResource resource = new ClassPathResource("dropwizard.yml");
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



    private void configureCors(Environment environment) {
        FilterRegistration.Dynamic filter = environment.servlets().addFilter("CORS", CrossOriginFilter.class);
        filter.addMappingForUrlPatterns(EnumSet.allOf(DispatcherType.class), true, "/*");
        filter.setInitParameter(CrossOriginFilter.ALLOWED_METHODS_PARAM, "GET,PUT,POST,DELETE,OPTIONS");
        filter.setInitParameter(CrossOriginFilter.ALLOWED_ORIGINS_PARAM, "*");
        filter.setInitParameter(CrossOriginFilter.ACCESS_CONTROL_ALLOW_ORIGIN_HEADER, "*");
        filter.setInitParameter("allowedHeaders", "Content-Type,Authorization,X-Requested-With,Content-Length,Accept,Origin");
        filter.setInitParameter("allowCredentials", "true");
    }
}
