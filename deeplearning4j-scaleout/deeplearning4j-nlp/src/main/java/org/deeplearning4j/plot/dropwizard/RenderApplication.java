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

package org.deeplearning4j.plot.dropwizard;

import io.dropwizard.Application;
import io.dropwizard.assets.AssetsBundle;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import io.dropwizard.views.ViewBundle;

import org.apache.commons.compress.utils.IOUtils;


import com.google.common.collect.ImmutableMap;
import org.canova.api.util.ClassPathResource;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

/**
 *
 * @author Adam Gibson
 */
public class RenderApplication extends Application<ApiConfiguration> {

    @Override
    public void initialize(Bootstrap<ApiConfiguration> apiConfigurationBootstrap) {
        apiConfigurationBootstrap.addBundle(new ViewBundle<ApiConfiguration>(){
            @Override
            public ImmutableMap<String, ImmutableMap<String, String>> getViewConfiguration(
                ApiConfiguration arg0) {
                return ImmutableMap.of();
            }
        });
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
        new RenderApplication().run("server", tmpConfig.getAbsolutePath());
    }
}
