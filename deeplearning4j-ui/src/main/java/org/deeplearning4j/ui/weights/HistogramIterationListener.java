package org.deeplearning4j.ui.weights;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;

/**
 *
 * A histogram iteration listener that
 * updates the weights of the model
 * with a web based ui.
 *
 * @author Adam Gibson
 */
public class HistogramIterationListener implements IterationListener {

    private Client client = ClientBuilder.newClient();
    private WebTarget target = client.target("http://localhost:8080").path("weights").path("update");


    @Override
    public void iterationDone(Model model, int iteration) {
        ModelAndGradient g = new ModelAndGradient(model);
        target.request(MediaType.APPLICATION_JSON).post(Entity.entity(g,MediaType.APPLICATION_JSON));

    }
}
