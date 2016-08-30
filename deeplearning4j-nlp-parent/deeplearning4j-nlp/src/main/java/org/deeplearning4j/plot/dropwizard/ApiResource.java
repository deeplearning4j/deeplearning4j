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

import com.google.common.collect.ImmutableMap;
import io.dropwizard.Application;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import io.dropwizard.views.ViewBundle;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.File;
import java.io.Serializable;
import java.util.List;

/**
 * Api Resource
 *
 * @author Adam Gibson
 */
@Deprecated
@Path("/api")
@Produces(MediaType.APPLICATION_JSON)

public class ApiResource extends Application<ApiConfiguration> implements Serializable {
    private static final Logger log = LoggerFactory.getLogger(ApiResource.class);
    private List<String> coords;


    public ApiResource(String coordPath) throws Exception {
        if(coordPath != null && new File(coordPath).exists()) {
            coords = FileUtils.readLines(new File(coordPath));
        }
    }

    @GET
    @Path("/coords")
    @Produces(MediaType.APPLICATION_JSON)
    public Response coords() {
        return Response.ok(coords).build();
    }

    /**
     * Initializes the application bootstrap.
     *
     * @param bootstrap the application bootstrap
     */
    @Override
    public void initialize(Bootstrap<ApiConfiguration> bootstrap) {
        bootstrap.addBundle(new ViewBundle<ApiConfiguration>(){
            @Override
            public ImmutableMap<String, ImmutableMap<String, String>> getViewConfiguration(
                ApiConfiguration arg0) {
                return ImmutableMap.of();
            }
        });

    }

    @Override
    public void run(ApiConfiguration configuration, Environment environment) throws Exception {

    }




}
