/*
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
package org.arbiter.optimize.ui;

import io.dropwizard.Application;
import io.dropwizard.assets.AssetsBundle;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import io.dropwizard.views.ViewBundle;
import org.arbiter.optimize.runner.CandidateStatus;
import org.arbiter.optimize.ui.components.RenderElements;
import org.arbiter.optimize.ui.resources.*;
import org.arbiter.util.WebUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

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

    - OptimizationRunner has a UIOptimizationRunnerStatusListener object. Called whenever something happens (task completion, etc)
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

    - Elements to be rendered in the various sections of the webpage: this is customizable.
        This is what the RenderableComponent classes are for: they define a set of things we want to render in the page.
        For example, text, line charts, tables etc. The idea is that for any platform using Arbiter, we might want to
        display a set of arbitrary objects as the status of each model/candidate.
        A set of commonly used elements (text, tables, line charts, histograms etc) are (or will be) provided here.
        TODO: Make this fully extensible/generic (i.e., able to render arbitrary objects via user-provided Javascript code)

      TODO: Work out how exactly to do full details (when accordian row is expanded)
        Maybe: Just do a request given the ID of the model? UI server then extracts/generates the required JSON
      TODO: Work out how to support cancelling of tasks from UI

     */

    private static final Logger log = LoggerFactory.getLogger(ArbiterUIServer.class);
    private static final ArbiterUIServer instance = new ArbiterUIServer();
    private Client client = ClientProvider.getClient();

    private AtomicLong lastSummaryUpdateTime = new AtomicLong(0);
    private AtomicLong lastConfigUpdateTime = new AtomicLong(0);
    private AtomicLong lastResultsUpdateTime = new AtomicLong(0);

    private WebTarget targetLastUpdateStatus = client.target("http://localhost:8080/lastUpdate/update");
    private WebTarget targetSummaryStatusUpdate = client.target("http://localhost:8080/summary/update");
    private WebTarget targetConfigUpdate = client.target("http://localhost:8080/config/update");
    private WebTarget targetResultsUpdate = client.target("http://localhost:8080/results/update");


    public static void main(String[] args) throws Exception {
        String[] str = new String[]{"server", "dropwizard.yml"};
        new ArbiterUIServer().run(str);
        WebUtils.tryOpenBrowser("http://localhost:8080/arbiter", log);    //TODO don't hardcode
    }

    public ArbiterUIServer(){
        super();
        log.info("Arbiter UI Server: Starting");
    }

    @Override
    public String getName() {
        return "arbiter-ui";
    }

    @Override
    public void initialize(Bootstrap<ArbiterUIConfig> bootstrap) {
        bootstrap.addBundle(new ViewBundle<ArbiterUIConfig>());
        bootstrap.addBundle(new AssetsBundle());
    }

    @Override
    public void run(ArbiterUIConfig configuration, Environment environment) {
        final ArbiterUIResource resource = new ArbiterUIResource();
        environment.jersey().register(resource);

        //Register our resources
        environment.jersey().register(new LastUpdateResource());
        environment.jersey().register(new SummaryStatusResource());
        environment.jersey().register(new ConfigResource());
        environment.jersey().register(new SummaryResultsResource());
        environment.jersey().register(new CandidateResultsResource());
    }

    public void updateStatus(RenderElements elements){
        Response response = targetSummaryStatusUpdate.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON)
                .post(Entity.entity(elements, MediaType.APPLICATION_JSON));
        log.info("Status update response: {}", response);
        log.info("Posted summary status update: {}", elements);
        lastSummaryUpdateTime.set(System.currentTimeMillis());

        updateStatusTimes();
    }

    private void updateStatusTimes(){
        UpdateStatus updateStatus = new UpdateStatus(lastSummaryUpdateTime.get(),lastConfigUpdateTime.get(),lastResultsUpdateTime.get());
        targetLastUpdateStatus.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON)
                .post(Entity.entity(updateStatus, MediaType.APPLICATION_JSON));
        log.info("Posted new update times: {}", updateStatus);
    }


    public void updateOptimizationSettings(RenderElements elements){
        targetConfigUpdate.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON)
                .post(Entity.entity(elements, MediaType.APPLICATION_JSON));
        log.info("Posted optimization settings update: {}", elements);

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
        targetResultsUpdate.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON)
                .post(Entity.entity(list, MediaType.APPLICATION_JSON));
        log.info("Posted new results: {}", list);
        lastResultsUpdateTime.set(System.currentTimeMillis());

        updateStatusTimes();
    }

}
