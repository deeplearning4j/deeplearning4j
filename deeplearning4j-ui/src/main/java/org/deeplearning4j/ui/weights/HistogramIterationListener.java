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
import java.util.*;

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
    private List<Map<String,List<Double>>> meanMagHistoryParams = new ArrayList<>();    //1 map per layer; keyed by new param name
    private List<Map<String,List<Double>>> meanMagHistoryUpdates = new ArrayList<>();
    private Map<String,Integer> layerNameIndexes = new HashMap<>();
    private List<String> layerNames = new ArrayList<>();
    private int layerNameIndexesCount = 0;
    private boolean openBrowser;
    private boolean firstIteration = true;
    private String path;
    private String subPath;

    public HistogramIterationListener(int iterations) {
        this(iterations, true, "weights");
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

            if(meanMagHistoryParams.size() == 0){
                //Initialize:
                int maxLayerIdx = -1;
                for(String s : grad.keySet()){
                    maxLayerIdx = Math.max(maxLayerIdx,indexFromString(s));
                }
                if(maxLayerIdx == -1 ) maxLayerIdx = 0;
                for( int i = 0; i <= maxLayerIdx; i++ ){
                    meanMagHistoryParams.add(new LinkedHashMap<String,List<Double>>());
                    meanMagHistoryUpdates.add(new LinkedHashMap<String,List<Double>>());
                }
            }

            //Process gradients: duplicate + calculate and store mean magnitudes
            Map<String,INDArray> newGrad = new LinkedHashMap<>();
            for(Map.Entry<String,INDArray> entry : grad.entrySet() ){
                String param = entry.getKey();
                String newName;
                if(Character.isDigit(param.charAt(0))) newName = "param_" + param;
                else newName = param;
                newGrad.put(newName,entry.getValue().dup());
                //CSS identifier can't start with digit http://www.w3.org/TR/CSS21/syndata.html#value-def-identifier

                //Work out layer index:
                Map<String,List<Double>> map = meanMagHistoryUpdates.get(indexFromString(param));
                List<Double> list = map.get(newName);
                if(list==null){
                    list = new ArrayList<>();
                    map.put(newName,list);
                }
                double meanMag = entry.getValue().norm1Number().doubleValue() / entry.getValue().length();
                list.add(meanMag);
            }

            //Process parameters: duplicate + calculate and store mean magnitudes
            Map<String,INDArray> params = model.paramTable();
            Map<String,INDArray> newParams = new LinkedHashMap<>();
            for(Map.Entry<String,INDArray> entry : params.entrySet()) {
                String param = entry.getKey();
                String newName;
                if(Character.isDigit(param.charAt(0))) newName = "param_" + param;
                else newName = param;


                newParams.put(newName, entry.getValue().dup());
                //dup() because params might be a view

                Map<String,List<Double>> map = meanMagHistoryParams.get(indexFromString(param));
                List<Double> list = map.get(newName);
                if(list==null){
                    list = new ArrayList<>();
                    map.put(newName,list);
                }
                double meanMag = entry.getValue().norm1Number().doubleValue() / entry.getValue().length();
                list.add(meanMag);
            }

            double score = model.score();
            scoreHistory.add(score);

            ModelAndGradient g = new ModelAndGradient();
            g.setGradients(newGrad);
            g.setParameters(newParams);
            g.setScore(score);
            g.setScores(scoreHistory);
            g.setPath(subPath);
            g.setUpdateMagnitudes(meanMagHistoryUpdates);
            g.setParamMagnitudes(meanMagHistoryParams);
            g.setLayerNames(layerNames);
            g.setLastUpdateTime(System.currentTimeMillis());

            Response resp = target.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON).post(Entity.entity(g,MediaType.APPLICATION_JSON));
            log.debug("{}",resp);

            if(openBrowser && firstIteration){
                UiUtils.tryOpenBrowser(path,log);
                firstIteration = false;
            }
        }
    }

    private int indexFromString(String str) {
        int underscore = str.indexOf("_");
        if (underscore == -1) {
            if (!layerNameIndexes.containsKey(str)) {
                layerNames.add(str);
                layerNameIndexes.put(str, layerNameIndexesCount++);
            }
            return layerNameIndexes.get(str);
        } else {
            String subStr = str.substring(0,underscore);
            if(!layerNameIndexes.containsKey(subStr)){
                layerNames.add(subStr);
                layerNameIndexes.put(subStr,layerNameIndexesCount++);
            }
            return layerNameIndexes.get(subStr);
        }
    }
}
