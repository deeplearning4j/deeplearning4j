package org.arbiter.optimize.report.web;

import io.dropwizard.Application;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import io.dropwizard.views.ViewBundle;

/**
 * Created by Alex on 20/12/2015.
 */
public class ArbiterUIServer extends Application<ArbiterUIConfig> {
    /* Design details: How the UI System and server actually works.
    UI system is web-based, running via a HTTP s
    Using the combination of the following things:
        DropWizard: set of libraries. (Jetty server, Jackson for JSON, etc)
        FreeMarker: Java template library. Used to generate HTML (which actually does rendering)
        d3: Javascript library, used to render charts, etc

    How it works, at an overview level:
    - Single web page containing all info, but using collapseable elements to avoid information overload
       Basic layout of the webpage:
       - Summary results: number of queued/completed tasks, best model score & index, total runtime etc
         This is rendered in a basic table, always displayed
       - Optimization settings: hyperparameter optimization scheme (i.e., random search vs. Bayesian methods + settings)
         Plus details of the hyperparameter space for the model (i.e., set of valid configurations).
         This section is collapseable, and is collapsed by default.
       - Results for each model. Two aspects to this section: Sortable table + accordian
         Sortable table: just lists things like candidate ID, its status (complete/failed/running), score, start/end times etc.
           Table can be sorted by clicking on heading. Default is to sort by ID
         Accordian: each row of the table can be cliked on. Clicking expands the row, and shows lots of information about the
           candidate: its configuration, plus model-specific information (such as score vs. epoch for DL4J).
           Clicking again collapses the row.

    - OptimizationRunner has a UIStatusListener object. Called whenever something happens (task completion, etc)
        Creates a status update object, and passes this to UI server for async processing???

    - Information to be displayed is posted to the folowing addresses, in JSON format
        /updateStatus   simple JSON, tracks when things were last updated. Loop on this, and update UI only when required
        /summary        summary results in JSON format -> table
        /config         optimization settings / details (hyperparameter space etc). JSON -> table
        /results        summary results for (non-accordian part) of results table. JSON -> table

    - Main web page code is in /resource/org/arbiter/optimize/report/web/arbiterui.ftl -> HTML + JavaScript
        This operates on timed loop, every 1 second or so
        Loop: Fetches and parses JSON from /updateStatus. This information is used to determine what elements to update
          -> If no data has changed since the last rendering: do nothing
          -> Otherwise: Update only the page elements that need to be updated

      TODO: Work out how exactly to do full details (when accordian row is expanded)
        Maybe: Just do a request given the ID of the model? UI server then extracts/generates the required JSON
      TODO: Work out encoding scheme for arbitrary objects in JSON (i.e., render a graph, a table, an image or whatever for each candidate)
        Then work out what it is and insert it into page as required. Plu
      TODO: Work out how to support cancelling of tasks from UI

     */



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
