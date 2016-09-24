package org.deeplearning4j.ui.activation;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.iterationlistener.ActivationMeanIterationListener;

import org.deeplearning4j.ui.UiConnectionInfo;


import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;

/**
 * @author Adam Gibson
 */
public class RemoteUpdateActivationIterationListener implements IterationListener {
    private Client client = ClientBuilder.newClient();
    private WebTarget target;
    private ActivationMeanIterationListener listener;
    private int iterations = 1;
    private boolean firstIteration = true;
    private String path;
    private UiConnectionInfo uiConnectionInfo;


    public RemoteUpdateActivationIterationListener(int iterations, UiConnectionInfo uiConnectionInfo,String subPath) {
        int port = -1;
        this.uiConnectionInfo = uiConnectionInfo;
        listener = new ActivationMeanIterationListener(iterations);
        this.iterations = iterations;
        path = uiConnectionInfo.getPath() + "/" + subPath;
        target = client.target("http://localhost:" + port).path(subPath).path("update");
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
