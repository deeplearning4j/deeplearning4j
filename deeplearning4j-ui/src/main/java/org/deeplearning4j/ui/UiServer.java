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

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.google.common.collect.ImmutableMap;
import io.dropwizard.Application;
import io.dropwizard.assets.AssetsBundle;
import io.dropwizard.jersey.jackson.JsonProcessingExceptionMapper;
import io.dropwizard.jetty.HttpConnectorFactory;
import io.dropwizard.lifecycle.ServerLifecycleListener;
import io.dropwizard.server.DefaultServerFactory;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import io.dropwizard.views.ViewBundle;
import org.apache.commons.io.IOUtils;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.ui.activation.ActivationsResource;
import org.deeplearning4j.ui.api.ApiResource;
import org.deeplearning4j.ui.defaults.DefaultResource;
import org.deeplearning4j.ui.exception.GenericExceptionMapper;
import org.deeplearning4j.ui.flow.FlowResource;
import org.deeplearning4j.ui.nearestneighbors.NearestNeighborsResource;
import org.deeplearning4j.ui.renders.RendersResource;
import org.deeplearning4j.ui.tsne.TsneResource;
import org.deeplearning4j.ui.weights.WeightResource;
import org.eclipse.jetty.server.Connector;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.servlets.CrossOriginFilter;
import org.glassfish.jersey.media.multipart.MultiPartFeature;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.jackson.VectorDeSerializer;
import org.nd4j.serde.jackson.VectorSerializer;

import javax.servlet.DispatcherType;
import javax.servlet.FilterRegistration;
import java.io.*;
import java.util.EnumSet;


/**
 * @author Adam Gibson
 */
public class UiServer extends Application<UIConfiguration> {
    private static UiServer INSTANCE;
    private UIConfiguration conf;
    private Environment env;
    public UiServer() {
        INSTANCE = this;
    }
    private int port;
    public int getPort(){ return port; }

    public static synchronized UiServer getInstance() throws Exception {
        if(INSTANCE == null) createServer();
        return INSTANCE;
    }

    public Environment getEnv() {
        return env;
    }

    @Override
    public void run(UIConfiguration uiConfiguration, Environment environment) throws Exception {
        this.conf = uiConfiguration;
        this.env = environment;

        //Workaround to dropwizard sometimes ignoring ports specified in dropwizard.yml
        int[] portsFromYml = getApplicationPortFromYml();
        if(portsFromYml[0] != -1 ) {
            ((HttpConnectorFactory) ((DefaultServerFactory) uiConfiguration.getServerFactory())
                    .getApplicationConnectors().get(0)).setPort(portsFromYml[0]);
        }
        if(portsFromYml[1] != -1 ) {
            ((HttpConnectorFactory) ((DefaultServerFactory) uiConfiguration.getServerFactory())
                    .getAdminConnectors().get(0)).setPort(portsFromYml[1]);
        }

        //Read the port that actually got assigned (needed for random ports i.e. port: 0 setting in yml)
        environment.lifecycle().addServerLifecycleListener(new ServerLifecycleListener() {
            @Override
            public void serverStarted(Server server) {
                for (Connector connector : server.getConnectors()) {
                    if (connector instanceof ServerConnector) {
                        ServerConnector serverConnector = (ServerConnector) connector;
                        if(!serverConnector.getName().toLowerCase().contains("application")) continue;
                        int port = serverConnector.getLocalPort();
                        try{
                            UiServer.getInstance().port = port;
                        }catch( Exception e ){
                            e.printStackTrace();
                        }
                    }
                }
            }
        });

        environment.jersey().register(MultiPartFeature.class);
        environment.jersey().register(new GenericExceptionMapper());
        environment.jersey().register(new JsonProcessingExceptionMapper());


        environment.jersey().register(new TsneResource(conf.getUploadPath()));
        environment.jersey().register(new NearestNeighborsResource(conf.getUploadPath()));
        environment.jersey().register(new DefaultResource());
        environment.jersey().register(new WeightResource());
        environment.jersey().register(new ActivationsResource());
        environment.jersey().register(new RendersResource());
        environment.jersey().register(new ApiResource());
        environment.jersey().register(new GenericExceptionMapper());
        environment.jersey().register(new FlowResource());
        environment.jersey().register(new org.deeplearning4j.ui.nearestneighbors.word2vec.NearestNeighborsResource(conf.getUploadPath()));

        environment.getObjectMapper().configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        environment.getObjectMapper().configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

        environment.getObjectMapper().registerModule(module());

        configureCors(environment);
    }

    @Override
    public void initialize(Bootstrap<UIConfiguration> bootstrap) {
        //custom serializers for the json serde
        bootstrap.getObjectMapper().registerModule(module());


        bootstrap.addBundle(new ViewBundle<UIConfiguration>() {
            @Override
            public ImmutableMap<String, ImmutableMap<String, String>> getViewConfiguration(
                    UIConfiguration arg0) {
                return ImmutableMap.of();
            }
        });
        bootstrap.addBundle(new AssetsBundle());
    }


    private SimpleModule module() {
        SimpleModule module = new SimpleModule();
        module.addSerializer(INDArray.class, new VectorSerializer());
        module.addDeserializer(INDArray.class, new VectorDeSerializer());
        return module;
    }

    public static void main(String[] args) throws Exception {
        createServer();
    }

    public static void createServer() throws Exception {
        ClassPathResource resource = new ClassPathResource("dropwizard.yml");
        File tmpConfig = resource.getFile();
        INSTANCE = new UiServer();
        INSTANCE.run("server", tmpConfig.getAbsolutePath());
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


    //Parse dropwizard.yml (if present on classpath) and parse the port specifications
    private int[] getApplicationPortFromYml(){
        int[] toReturn = {-1,-1};
        ClassPathResource resource = new ClassPathResource("dropwizard.yml");
        InputStream in = null;
        try {
             in = resource.getInputStream();
            if (in == null) return toReturn;   //Not found
        } catch (FileNotFoundException e) {
            return toReturn;
        }
        String s;
        try {
            s = IOUtils.toString(in);
        } catch(IOException e ){
            return toReturn;
        }

        String[] split = s.split("\n");
        int count = 0;
        for( String str : split ){
            if(!str.contains("port")) continue;
            System.out.println(str);
            String[] line = str.split("\\s+");
            for( String token : line ){
                try{
                    toReturn[count] = Integer.parseInt(token);
                    count++;
                }catch(NumberFormatException e ){ }
            }
            if(count == 2 ) return toReturn;
        }

        return toReturn;  //No port configuration?
    }
}
