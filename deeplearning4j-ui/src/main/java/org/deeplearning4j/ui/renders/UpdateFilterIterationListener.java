package org.deeplearning4j.ui.renders;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.PlotFilters;
import org.deeplearning4j.plot.iterationlistener.PlotFiltersIterationListener;
import org.deeplearning4j.ui.UiServer;
import org.deeplearning4j.ui.UiUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class UpdateFilterIterationListener implements IterationListener {
    private static final Logger log = LoggerFactory.getLogger(UpdateFilterIterationListener.class);
    private Client client = ClientBuilder.newClient();
    private WebTarget target;
    private PlotFiltersIterationListener listener;
    private int iterations = 1;
    private boolean openBrowser;
    private boolean firstIteration = true;
    private String path;

    /**
     * Initializes with the variables to render filters for
     * @param variables the variables ot render filters for
     * @param iterations the number of iterations to update on
     */
    public UpdateFilterIterationListener(PlotFilters filters,List<String> variables,int iterations) {
        this(filters,variables,iterations,true,"filters");
    }
    public UpdateFilterIterationListener(PlotFilters filters,List<String> variables,int iterations, boolean openBrowser,
                                         String subPath ) {
        int port = -1;
        try{
            UiServer server = UiServer.getInstance();
            port = server.getPort();
        }catch(Exception e){
            log.error("Error initializing UI server",e);
            throw new RuntimeException(e);
        }
        target = client.target("http://localhost:" + port ).path(subPath).path("update");
        listener = new PlotFiltersIterationListener(filters,variables,0);
        this.iterations = iterations;
        this.openBrowser = openBrowser;
        path = "http://localhost:" + port + "/" + subPath;
    }

    @Override
    public boolean invoked() {
        return false;
    }

    @Override
    public void invoke() {

    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if(iteration % iterations == 0) {
            PathUpdate update = new PathUpdate();
            //update the weights
            listener.iterationDone(model, iteration);
            //ensure path is set
            update.setPath(listener.getOutputFile().getPath());
            //ensure the server is hooked up with the path
            target.request(MediaType.APPLICATION_JSON).post(Entity.entity(update, MediaType.APPLICATION_JSON));
            if(openBrowser && firstIteration){
                UiUtils.tryOpenBrowser(path, log);
                firstIteration = false;
            }
        }

    }
}
