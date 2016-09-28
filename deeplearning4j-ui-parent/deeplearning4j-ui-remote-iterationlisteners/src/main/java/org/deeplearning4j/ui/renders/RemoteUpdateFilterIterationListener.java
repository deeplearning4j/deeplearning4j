package org.deeplearning4j.ui.renders;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.PlotFilters;
import org.deeplearning4j.plot.iterationlistener.PlotFiltersIterationListener;
import org.deeplearning4j.ui.UiConnectionInfo;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import java.util.List;

/**
 *
 * Updates the filters
 * in the ui for rendering
 * @author Adam Gibson
 */
public class RemoteUpdateFilterIterationListener implements IterationListener {
    private static final Logger log = LoggerFactory.getLogger(RemoteUpdateFilterIterationListener.class);
    private Client client = ClientBuilder.newClient();
    private WebTarget target;
    private PlotFiltersIterationListener listener;
    private int iterations = 1;
    private boolean openBrowser;
    private boolean firstIteration = true;
    private String path;


    public RemoteUpdateFilterIterationListener(PlotFilters filters, List<String> variables, int iterations, UiConnectionInfo uiConnectionInfo,
                                               String subPath) {
        target = client.target(uiConnectionInfo.getAddress()).path(subPath).path("update");
        listener = new PlotFiltersIterationListener(filters,variables,0);
        this.iterations = iterations;
        path = uiConnectionInfo.getFullAddress() + "/" + subPath;
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
            if(firstIteration) {
                firstIteration = false;
            }
        }

    }
}
