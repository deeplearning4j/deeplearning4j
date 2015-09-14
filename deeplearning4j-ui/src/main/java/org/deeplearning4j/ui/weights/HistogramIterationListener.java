package org.deeplearning4j.ui.weights;


import com.fasterxml.jackson.jaxrs.json.JacksonJsonProvider;
import io.dropwizard.server.DefaultServerFactory;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.UiServer;
import org.deeplearning4j.ui.UiUtils;
import org.deeplearning4j.ui.providers.ObjectMapperProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
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
    private static final Logger log = LoggerFactory.getLogger(HistogramIterationListener.class);
    private Client client = ClientBuilder.newClient().register(JacksonJsonProvider.class).register(new ObjectMapperProvider());
    private WebTarget target;
    private int iterations = 1;
    private ArrayList<Double> scoreHistory = new ArrayList<>();
    private boolean openBrowser;
    private boolean firstIteration = true;
    private String path;
    private String subPath;

    public HistogramIterationListener(int iterations) {
        this(iterations,true,"weights");
    }
    public HistogramIterationListener(int iterations, boolean openBrowser, String subPath){
        int port = -1;
        try{
            UiServer server = UiServer.getInstance();
            port = server.getPort();
        }catch(Exception e){
            log.error("Error initializing UI server",e);
            throw new RuntimeException(e);
        }

        this.iterations = iterations;
        target = client.target("http://localhost:" + port ).path(subPath).path("update");
        this.openBrowser = openBrowser;
        this.path = "http://localhost:" + port + "/" + subPath;
        this.subPath = subPath;

        System.out.println("UI Histogram: " + this.path);
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
                newParams.put("param_" + entry.getKey(), entry.getValue().dup());
                //dup() because params might be a view
            }

            double score = model.score();
            scoreHistory.add(score);

            ModelAndGradient g = new ModelAndGradient();
            g.setGradients(newGrad);
            g.setParameters(newParams);
            g.setScore(score);
            g.setScores(scoreHistory);
            g.setPath(subPath);

            Response resp = target.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON).post(Entity.entity(g,MediaType.APPLICATION_JSON));
            log.debug("{}",resp);

            if(openBrowser && firstIteration){
                UiUtils.tryOpenBrowser(path,log);
                firstIteration = false;
            }
        }


    }
}
