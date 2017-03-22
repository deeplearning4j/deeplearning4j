/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.deeplearning4j.arbiter.optimize.ui;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.google.common.collect.ImmutableMap;
import io.dropwizard.Application;
import io.dropwizard.assets.AssetsBundle;
import io.dropwizard.jetty.HttpConnectorFactory;
import io.dropwizard.lifecycle.ServerLifecycleListener;
import io.dropwizard.server.DefaultServerFactory;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import io.dropwizard.views.ViewBundle;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.arbiter.optimize.runner.CandidateStatus;
import org.deeplearning4j.arbiter.optimize.ui.resources.*;
import org.deeplearning4j.arbiter.util.ClassPathResource;
import org.deeplearning4j.arbiter.util.WebUtils;
import org.deeplearning4j.ui.api.Component;
import org.eclipse.jetty.server.Connector;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Main class for the Arbiter UI
 */
@Slf4j
public class ArbiterUIServer extends Application<ArbiterUIConfig> {
    /* Design details: How the UI System and server actually works.
    UI system is web-based, running via a HTTP server. Java code posts information to server; Javascript (UI code in browser)
    periodically fetches this info and renders it on the page.

    Design utilizes a combination of the following:
        DropWizard: set of libraries. (Jetty server, Jackson for JSON, etc)
        FreeMarker: Java template library. Used to generate HTML (which actually does rendering)
        d3: Javascript library, used to render charts, etc (todo: not being used yet)

    How it works, at an overview level:
    - Single web page containing all info, but using collapseable elements to avoid information overload
       Basic layout of the webpage:
       - Summary results: number of queued/completed tasks, best model score & index, total runtime etc
         This is rendered in a basic table (todo, text only currently)
       - Optimization settings: hyperparameter optimization scheme (i.e., random search vs. Bayesian methods + settings)
         Plus details of the hyperparameter space for the model (i.e., set of valid configurations).
         This section is collapseable, and is collapsed by default.
       - Results for each model. Two aspects to this section: Sortable table + accordian
         Sortable table: just lists things like candidate ID, its status (complete/failed/running), score, start/end times etc.
           Table can be sorted by clicking on heading. Default is to sort by ID
         Accordian: each row of the table can be cliked on. Clicking expands the row, and shows lots of information about the
           candidate: its configuration, plus model-specific information (such as score vs. epoch for DL4J).
           Clicking again collapses the row.

    - BaseOptimizationRunner has a UIOptimizationRunnerStatusListener object. Called whenever something happens (task completion, etc)
        Creates a status update object, and passes this to UI server for async processing???

    - Information to be displayed is posted to the folowing addresses, in JSON format
        /lastUpdate     simple JSON, tracks when things were last updated. Loop on this, and update UI only when required
        /summary        summary results in JSON format -> table
        /config         optimization settings / details (hyperparameter space etc). JSON -> table
        /results        summary results for (non-accordian part) of results table. JSON -> table

    - Main web page code is in /resource/org/arbiter/optimize/report/web/arbiter.ftl -> HTML + JavaScript
        DropWizard/FreeMarker looks specifically for this path based on class in which "arbiter.ftl" is used
            http://www.dropwizard.io/0.9.1/docs/manual/views.html
        This operates on timed loop, every 1 second or so
        Loop: Fetches and parses JSON from /updateStatus. This information is used to determine what elements to update
          -> If no data has changed since the last rendering: do nothing
          -> Otherwise: Update only the page elements that need to be updated

    - How updates are actually executed:
        Updates are posted to /lastUpdate/update, /summary/update, /config/update, /results/update
        Format is JSON; POST to server is executed via the WebTarget.post(...) methods here
        JSON serialization is done automatically on java objects using Jackson
        These paths are set via the LastUpdateResource, SummaryStatusResource, ConfigResource
        An instance of each of these resources classes must be registered with Jersey

      TODO: Work out how to support cancelling of tasks from UI
     */

    private static ArbiterUIServer instance;
    private Client client = ClientProvider.getClient();

    private AtomicLong lastSummaryUpdateTime = new AtomicLong(0);
    private AtomicLong lastConfigUpdateTime = new AtomicLong(0);
    private AtomicLong lastResultsUpdateTime = new AtomicLong(0);

    private int port;

    private WebTarget targetLastUpdateStatus; //= client.target("http://localhost:8080/lastUpdate/update");
    private WebTarget targetSummaryStatusUpdate; // = client.target("http://localhost:8080/summary/update");
    private WebTarget targetConfigUpdate; // = client.target("http://localhost:8080/config/update");
    private WebTarget targetResultsUpdate; // = client.target("http://localhost:8080/results/update");

    public int getPort(){
        return port;
    }

    public static boolean isRunning(){
        return instance != null;
    }

    public static synchronized ArbiterUIServer getInstance() {
        if(instance == null){
            File f = null;
            try{
                f = new ClassPathResource("arbiterdropwizard.yml").getFile();
            }catch(Exception e){
                //Didn't find arbiterdropwizard.yml -> look next for dropwizard.yml
            }
            if(f == null){
                try{
                    f = new ClassPathResource("dropwizard.yml").getFile();
                }catch(Exception e){
                    throw new RuntimeException("Could not find dropwizard configuration for UI: could not find arbiterdropwizard.yml or dropwizard.yml on classpath");
                }
            }

            instance = new ArbiterUIServer();
            String[] str = new String[]{"server", f.getAbsolutePath()};
            try{
                instance.run(str);
            }catch(Exception e){
                instance = null;
                throw new RuntimeException(e);
            }
            WebUtils.tryOpenBrowser("http://localhost:" + instance.port + "/arbiter", log);
        }
        return instance;
    }

    protected ArbiterUIServer(){
        super();
        log.info("Arbiter UI Server: Starting");
    }

    @Override
    public String getName() {
        return "arbiter-ui";
    }

    @Override
    public void initialize(Bootstrap<ArbiterUIConfig> bootstrap) {
        bootstrap.addBundle(new ViewBundle<ArbiterUIConfig>() {
            @Override
            public ImmutableMap<String, ImmutableMap<String, String>> getViewConfiguration(
                    ArbiterUIConfig arg0) {
                return ImmutableMap.of();
            }
        });
        bootstrap.addBundle(new AssetsBundle());
    }

    @Override
    public void run(ArbiterUIConfig configuration, Environment environment) {
        //Workaround to dropwizard sometimes ignoring ports specified in dropwizard.yml
        int[] portsFromYml = getApplicationPortFromYml();
        if(portsFromYml[0] != -1 ) {
            ((HttpConnectorFactory) ((DefaultServerFactory) configuration.getServerFactory())
                    .getApplicationConnectors().get(0)).setPort(portsFromYml[0]);
        }
        if(portsFromYml[1] != -1 ) {
            ((HttpConnectorFactory) ((DefaultServerFactory) configuration.getServerFactory())
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
                            ArbiterUIServer.getInstance().port = port;
                        }catch( Exception e ){
                            e.printStackTrace();
                        }
                    }
                }
            }
        });

        final ArbiterUIResource resource = new ArbiterUIResource();
        environment.jersey().register(resource);

        //Register our resources
        environment.jersey().register(new LastUpdateResource());
        environment.jersey().register(new SummaryStatusResource());
        environment.jersey().register(new ConfigResource());
        environment.jersey().register(new SummaryResultsResource());
        environment.jersey().register(new CandidateResultsResource());

        environment.getObjectMapper().configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        environment.getObjectMapper().configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    }

    public void updateStatus(Component component){
        String str = "";
        try{
            str = new ObjectMapper().writeValueAsString(component);
        }catch (Exception e){

        }

        if(targetSummaryStatusUpdate == null) targetSummaryStatusUpdate = client.target("http://localhost:" + port + "/summary/update");

        Response response = targetSummaryStatusUpdate.request(MediaType.TEXT_PLAIN).accept(MediaType.APPLICATION_JSON)
                .post(Entity.entity(str, MediaType.TEXT_PLAIN));

        log.trace("Status update response: {}", response);
        log.trace("Posted summary status update: {}", component);
        lastSummaryUpdateTime.set(System.currentTimeMillis());

        updateStatusTimes();
    }

    private void updateStatusTimes(){
        if(targetLastUpdateStatus == null) targetLastUpdateStatus = client.target("http://localhost:" + port + "/lastUpdate/update");
        UpdateStatus updateStatus = new UpdateStatus(lastSummaryUpdateTime.get(),lastConfigUpdateTime.get(),lastResultsUpdateTime.get());
        targetLastUpdateStatus.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON)
                .post(Entity.entity(updateStatus, MediaType.APPLICATION_JSON));
        log.trace("Posted new update times: {}", updateStatus);
    }


    public void updateOptimizationSettings(Component component){
        if(targetConfigUpdate == null) targetConfigUpdate = client.target("http://localhost:" + port + "/config/update");

        String str = "";
        try{
            str = new ObjectMapper().writeValueAsString(component);
        }catch (Exception e){

        }

        targetConfigUpdate.request(MediaType.TEXT_PLAIN).accept(MediaType.APPLICATION_JSON)
                .post(Entity.entity(str, MediaType.TEXT_PLAIN));

        log.trace("Posted optimization settings update: {}", component);

        lastConfigUpdateTime.set(System.currentTimeMillis());

        updateStatusTimes();
    }

    public void updateResults(Collection<CandidateStatus> status){
        List<CandidateStatus> list = new ArrayList<>(status);
        Collections.sort(list, new Comparator<CandidateStatus>() {
            @Override
            public int compare(CandidateStatus o1, CandidateStatus o2) {
                return Integer.compare(o1.getIndex(), o2.getIndex());
            }
        });

        //Post update:
        if(targetResultsUpdate == null) targetResultsUpdate = client.target("http://localhost:" + port + "/results/update");
        targetResultsUpdate.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON)
                .post(Entity.entity(list, MediaType.APPLICATION_JSON));
        log.trace("Posted new results: {}", list);
        lastResultsUpdateTime.set(System.currentTimeMillis());

        updateStatusTimes();
    }

    //Parse dropwizard.yml (if present on classpath) and parse the port specifications
    private int[] getApplicationPortFromYml(){
        int[] toReturn = {-1,-1};
        ClassPathResource resource = new ClassPathResource("dropwizard.yml");
        InputStream in;
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
            if(str.matches("^\\s*#(.|\n|\r)*")) continue;    //Ignore comment lines
            if(!str.contains("port")) continue;
            String[] line = str.split("\\s+");
            for( String token : line ){
                try{
                    toReturn[count] = Integer.parseInt(token);
                    count++;
                }
                catch(NumberFormatException e ){ }
            }
            if(count == 2 ) return toReturn;
        }

        return toReturn;  //No port configuration?
    }

}
