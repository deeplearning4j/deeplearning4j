package org.deeplearning4j.ui.weights;


import com.fasterxml.jackson.jaxrs.json.JacksonJsonProvider;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.providers.ObjectMapperProvider;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 *
 * A histogram iteration listener that
 * updates the weights of the model
 * with a web based ui.
 *
 * @author Adam Gibson
 */
public class HistogramIterationListener implements IterationListener {

    private Client client = ClientBuilder.newClient().register(JacksonJsonProvider.class).register(new ObjectMapperProvider());
    private WebTarget target;
    private int iterations = 1;

    public HistogramIterationListener(int iterations) {
        this.iterations = iterations;
        target = client.target("http://localhost:8080").path("weights").path("update");
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
            Map<String,INDArray> grad = model.gradient().gradientForVariable();
            Map<String,INDArray> newGrad = new LinkedHashMap<>();
            for(Map.Entry<String,INDArray> entry : grad.entrySet() ){
                newGrad.put("param_" + entry.getKey(),entry.getValue().dup());
                //CSS identifier can't start with digit http://www.w3.org/TR/CSS21/syndata.html#value-def-identifier
            }

            Map<String,INDArray> params = model.paramTable();
            Map<String,INDArray> newParams = new LinkedHashMap<>();
            for(Map.Entry<String,INDArray> entry : params.entrySet()) {
                newParams.put("param_" + entry.getKey(),entry.getValue().dup());
                //dup() because params might be a view
            }

            ModelAndGradient g = new ModelAndGradient();
            g.setGradients(newGrad);
            g.setParameters(newParams);
            g.setScore(model.score());

            Response resp = target.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON).post(Entity.entity(g,MediaType.APPLICATION_JSON));
            System.out.println(resp);
        }


    }
}
