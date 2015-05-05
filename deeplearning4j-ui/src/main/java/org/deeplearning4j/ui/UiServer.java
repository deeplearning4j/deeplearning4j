/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

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
        environment.jersey().register(new TsneResource(conf.getUploadPath()));
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
        new UiServer().run("server", tmpConfig.getAbsolutePath());
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
