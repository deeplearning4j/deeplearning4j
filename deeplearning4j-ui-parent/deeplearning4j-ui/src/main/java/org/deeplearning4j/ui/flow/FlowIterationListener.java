package org.deeplearning4j.ui.flow;

import com.fasterxml.jackson.jaxrs.json.JacksonJsonProvider;
import lombok.NonNull;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.UiConnectionInfo;
import org.deeplearning4j.ui.UiServer;
import org.deeplearning4j.ui.UiUtils;
import org.deeplearning4j.ui.flow.beans.Description;
import org.deeplearning4j.ui.flow.beans.LayerInfo;
import org.deeplearning4j.ui.flow.beans.ModelInfo;
import org.deeplearning4j.ui.providers.ObjectMapperProvider;
import org.glassfish.jersey.filter.LoggingFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.UriBuilder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This IterationListener is suited for general model performance/architecture overview
 *
 * PLEASE NOTE: WORK IN PROGRESS, DO NOT USE IT UNLESS YOU HAVE TO
 * @author raver119@gmail.com
 */
public class FlowIterationListener implements IterationListener {
    // TODO: basic auth should be considered here as well
    private String remoteAddr;
    private int remotePort;
    private String login;
    private String password;
    private int frequency = 1;
    private boolean firstIteration = true;
    private String path;
    private UiConnectionInfo connectionInfo;

    private static final List<String> colors = Collections.unmodifiableList(Arrays.asList("#9966ff", "#ff9933", "#ffff99", "#3366ff", "#0099cc", "#669999", "#66ffff"));

    private Client client = ClientBuilder.newClient().register(JacksonJsonProvider.class).register(new ObjectMapperProvider());
    private WebTarget target;

    private static Logger log = LoggerFactory.getLogger(FlowIterationListener.class);

    /**
     * Creates IterationListener and keeps it detached from any UiServer instances
     */
    protected FlowIterationListener() {
        // please keep this constructor protected
    }

    /**
     * Creates IterationListener and attaches it local UiServer instance
     *
     * @param frequency update frequency
     */
    public FlowIterationListener(int frequency) {
        this("localhost", 0, frequency);
    }

    /**
     *  Creates IterationListener and attaches it to specified remote UiServer instance
     *
     * @param address remote UiServer address
     * @param port remote UiServer port
     * @param frequency update frequency
     */
    public FlowIterationListener(@NonNull String address, int port, int frequency) {
        this.remoteAddr = address;
        this.remotePort = port;
        this.frequency = frequency;
        UiConnectionInfo info = null;

        if (address.equals("localhost") || address.equals("127.0.0.1") || address.isEmpty()) {
            try {
                this.remoteAddr = "localhost";
                this.remotePort = UiServer.getInstance().getPort();
                info = UiServer.getInstance().getConnectionInfo();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        setup(info);
    }


    /**
     * Creates IterationListener and attaches it to specified remote UiServer instance
     *
     * @param login Login for HTTP Basic auth
     * @param password Password for HTTP Basic auth
     * @param address remote UiServer address
     * @param port remote UiServer port
     * @param frequency update frequency
     */
    public FlowIterationListener(@NonNull String login, @NonNull String password, @NonNull String address, int port, int frequency) {
        this(address, port, frequency);
        this.connectionInfo.setLogin(login);
        this.connectionInfo.setPassword(password);
        this.login = login;
        this.password = password;
    }

    public FlowIterationListener(@NonNull UiConnectionInfo connectionInfo, int frequency) {
        setup(connectionInfo);
    }

    private void setup(@NonNull UiConnectionInfo connectionInfo) {
        // TODO: add auth option

        this.connectionInfo = connectionInfo;

        java.util.logging.Logger logger =  java.util.logging.Logger.getGlobal();
        login = null;
        password = null;
       // client.register(new LoggingFilter(logger, true));
        if (login == null || password == null) target = client.target(connectionInfo.getFirstPart()).path(connectionInfo.getSecondPart("flow")).path("state").queryParam("sid", connectionInfo.getSessionId());

        this.path = connectionInfo.getFullAddress("flow");

        log.info("Flow UI address: " + this.path);
    }

    /**
     * Get if listener invoked
     */
    @Override
    public boolean invoked() {
        return false;
    }

    /**
     * Change invoke to true
     */
    @Override
    public void invoke() {

    }

    /**
     * Event listener for each iteration
     *
     * @param model     the model iterating
     * @param iteration the iteration
     */
    @Override
    public synchronized void iterationDone(Model model, int iteration) {
        if (iteration % frequency == 0) {
        /*
            Basic plan:
                1. We should detect, if that's CompGraph or MultilayerNetwork. However the actual difference will be limited to number of non-linear connections.
                2. Network structure should be converted to JSON
                3. Params for each node should be packed to JSON as well
                4. For specific cases (like CNN) binary data should be wrapped into base64
                5. For arrays/params gzip could be used (to be investigated)
                ......
                Later, on client side, this JSON should be parsed and rendered. So, proper object structure to be considered.
         */

            // On first pass we just build list of layers. However, for MultiLayerNetwork first pass is the last pass, since we know connections in advance
            ModelInfo info = buildModelInfo(model);

            // add info about inputs


        /*
            as soon as model info is built, we need to define color scheme based on number of unique nodes
         */

            // send ModelInfo to UiServer
            Response resp = target.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON).post(Entity.entity(info, MediaType.APPLICATION_JSON));
        //    log.info("ModelInfo:" + Entity.entity(info, MediaType.APPLICATION_JSON));
            log.debug("Response: " + resp);
        /*
            TODO: it would be nice to send updates of nodes as well
         */

            if(firstIteration){
                try {
                    UiUtils.tryOpenBrowser(path, log);
                } catch (Exception e) {
                    ;
                }
                firstIteration = false;
            }
        }
    }

    /**
     * This method returns all Layers connected to the currentInput
     *
     * @param vertices
     * @param currentInput
     * @param currentY
     * @return
     */
    protected  List<LayerInfo> flattenToY(ModelInfo model, GraphVertex[] vertices, List<String> currentInput, int currentY) {
        List<LayerInfo> results = new ArrayList<>();
        int x = 0;
        for (int v = 0; v < vertices.length; v++) {
            GraphVertex vertex = vertices[v];
            VertexIndices[] indices = vertex.getInputVertices();

            if (indices != null) for (int i = 0; i < indices.length; i++) {
                GraphVertex cv = vertices[indices[i].getVertexIndex()];
                String inputName = cv.getVertexName();

                for (String input: currentInput) {
                    if (inputName.equals(input)) {
                        // we have match for Vertex
                    //    log.info("Vertex: " + vertex.getVertexName() + " has Input: " + input);
                        try {
                            LayerInfo info = model.getLayerInfoByName(vertex.getVertexName());
                            if (info == null) info = getLayerInfo(vertex.getLayer(), x, currentY, 121);
                            info.setName(vertex.getVertexName());

                            // special case here: vertex isn't a layer
                            if (vertex.getLayer() == null) {
                                info.setLayerType(vertex.getClass().getSimpleName());
                            }
                            if (info.getName().endsWith("-merge")) info.setLayerType("MERGE");
                            if (model.getLayerInfoByName(vertex.getVertexName()) == null) {
                                x++;
                                model.addLayer(info);
                                results.add(info);
                            }

                            // now we should map connections
                            LayerInfo connection = model.getLayerInfoByName(input);
                            if (connection != null) {
                                connection.addConnection(info);
                              //  log.info("Adding connection ["+ connection.getName()+"] -> ["+ info.getName()+"]");
                            } else {
                                // the only reason to have null here, is direct input connection
                                //connection.addConnection(0,0);
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }
        return results;
    }

    protected ModelInfo buildModelInfo(Model model) {
        ModelInfo modelInfo = new ModelInfo();
        if (model instanceof ComputationGraph) {
            ComputationGraph graph = (ComputationGraph) model;
            /*
                we assume that graph starts on input. every layer connected to input - is on y1
                every layer connected to y1, is on y2 etc.
              */
            List<String> inputs = graph.getConfiguration().getNetworkInputs();
            // now we need to add inputs as y0 nodes
            int x = 0;
            for (String input: inputs) {
                LayerInfo info = new LayerInfo();
                info.setId(0);
                info.setName(input);
                info.setY(0);
                info.setX(x);
                info.setLayerType("INPUT");
                info.setDescription(new Description());
                info.getDescription().setMainLine("Model input");
                modelInfo.addLayer(info);
                x++;
            }

            GraphVertex[] vertices = graph.getVertices();

            // filling grid in LTR/TTB direction
            List<String> needle = new ArrayList<>();


            // we assume that max row can't be higher then total number of vertices
            for (int y = 1; y < vertices.length; y++) {
                if (needle.isEmpty()) needle.addAll(inputs);

                /*
                    for each grid row we look for nodes, that are connected to previous layer
                */
                List<LayerInfo> layersForGridY =  flattenToY(modelInfo, vertices, needle, y);

                needle.clear();
                for (LayerInfo layerInfo: layersForGridY) {
                    needle.add(layerInfo.getName());
                }
                if (needle.isEmpty()) break;
            }

        } else if (model instanceof MultiLayerNetwork) {
            MultiLayerNetwork network = (MultiLayerNetwork) model;

            // manually adding input layer
            LayerInfo info = new LayerInfo();
            info.setId(0);
            info.setName("Input");
            info.setY(0);
            info.setX(0);
            info.setLayerType("INPUT");
            info.setDescription(new Description());
            info.getDescription().setMainLine("Model input");
            info.addConnection(0, 1);
            modelInfo.addLayer(info);

            // entry 0 is reserved for inputs
            int y = 1;

            // for MLN x value is always 0
            final int x = 0;
            for (Layer layer: network.getLayers()) {
                LayerInfo layerInfo = getLayerInfo(layer, x, y, y);
                // since it's MLN, we know connections in advance as curLayer + 1
                layerInfo.addConnection(x, y+1);
                modelInfo.addLayer(layerInfo);
                y++;
            }

            LayerInfo layerInfo = modelInfo.getLayerInfoByCoords(x, y - 1);
            layerInfo.dropConnections();

        }// else throw new IllegalStateException("Model ["+model.getClass().getCanonicalName()+"] doesn't looks like supported one.");

        // find layers without connections, and mark them as output layers
        for (LayerInfo layerInfo: modelInfo.getLayers()) {
            if (layerInfo.getConnections().size() == 0) layerInfo.setLayerType("OUTPUT");
        }

        // now we apply colors to distinct layer types
        AtomicInteger cnt = new AtomicInteger(0);
        for (String layerType: modelInfo.getLayerTypes()) {
            String curColor = colors.get(cnt.getAndIncrement());
            if (cnt.get() >= colors.size()) cnt.set(0);
            for (LayerInfo layerInfo: modelInfo.getLayersByType(layerType)) {
                if (layerType.equals("INPUT")) {
                    layerInfo.setColor("#99ff66");
                } else if (layerType.equals("OUTPUT")) {
                    layerInfo.setColor("#e6e6e6");
                } else {
                    layerInfo.setColor(curColor);
                }
            }
        }
        return modelInfo;
    }

    private LayerInfo getLayerInfo(Layer layer, int x, int y, int order) {
        LayerInfo info = new LayerInfo();


        // set coordinates
        info.setX(x);
        info.setY(y);

        // if name was set, we should grab it
        try {
            info.setName(layer.conf().getLayer().getLayerName());
        } catch (Exception e) {
            ;
        }
        if (info.getName() == null || info.getName().isEmpty()) info.setName("unnamed");

        // unique layer id required here
        info.setId(order);

        // set layer description according to layer params
        Description description = new Description();
        info.setDescription(description);

        // set layer type
        try {
            info.setLayerType(layer.getClass().getSimpleName());
        } catch (Exception e) {
            info.setLayerType("n/a");
            return info;
        }


        StringBuilder mainLine = new StringBuilder();
        StringBuilder subLine = new StringBuilder();

    //    log.info("Layer: " + info.getName() + " class: " + layer.getClass().getSimpleName());

        if (layer.type().equals(Layer.Type.CONVOLUTIONAL)) {
            org.deeplearning4j.nn.conf.layers.ConvolutionLayer layer1 = (org.deeplearning4j.nn.conf.layers.ConvolutionLayer) layer.conf().getLayer();
            mainLine.append("K: " + Arrays.toString(layer1.getKernelSize()) + " S: " + Arrays.toString(layer1.getStride()) + " P: " + Arrays.toString(layer1.getPadding()));
            subLine.append("nIn/nOut: [" + layer1.getNIn() + "/" + layer1.getNOut() + "]");
        } else if (layer.conf().getLayer() instanceof FeedForwardLayer) {
            org.deeplearning4j.nn.conf.layers.FeedForwardLayer layer1 = (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) layer.conf().getLayer();
            mainLine.append("nIn/nOut: [" + layer1.getNIn() + "/" + layer1.getNOut() + "]");
            subLine.append(info.getLayerType());
        } else {
                // TODO: Introduce Layer.Type.OUTPUT
                if (layer instanceof BaseOutputLayer) {
                    mainLine.append("Outputs: [" + ((org.deeplearning4j.nn.conf.layers.BaseOutputLayer)layer.conf().getLayer()).getNOut()+ "]");
                }
        }

        subLine.append(" A: [").append(layer.conf().getLayer().getActivationFunction()).append("]");

        description.setMainLine(mainLine.toString());
        description.setSubLine(subLine.toString());

        return info;
    }
}
