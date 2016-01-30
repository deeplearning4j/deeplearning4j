package org.deeplearning4j.ui.flow;

import lombok.NonNull;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.UiServer;
import org.deeplearning4j.ui.flow.beans.Description;
import org.deeplearning4j.ui.flow.beans.LayerInfo;
import org.deeplearning4j.ui.flow.beans.ModelInfo;

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
        this.login = login;
        this.password = password;
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
        ModelInfo modelInfo = new ModelInfo();
        if (model instanceof ComputationGraph) {
            ComputationGraph graph = (ComputationGraph) model;

            // keep x0, y0 reserved for inputs


        } else if (model instanceof MultiLayerNetwork) {
            MultiLayerNetwork network = (MultiLayerNetwork) model;

            // entry 0 is reserved for inputs
            int cnt = 1;
            for (Layer layer: network.getLayers()) {
                modelInfo.addLayer(getLayerInfo(layer, 1, cnt, cnt));
                cnt++;
            }
        } else throw new IllegalStateException("Model ["+model.getClass().getCanonicalName()+"] doesn't looks like supported one.");

        // add info about inputs


        /*
            as soon as model info is built, we need to define color scheme based on number of unique nodes
         */
    }

    private LayerInfo getLayerInfo(Layer layer, int x, int y, int order) {
        LayerInfo info = new LayerInfo();


        // set coordinates
        info.setX(x);
        info.setY(y);

        // if name was set, we should grab it
        info.setName(layer.conf().getLayer().getLayerName());

        // unique layer id required here
        info.setId(order);

        // set layer type
        info.setLayerType(layer.type());

        // set layer description according to layer params
        Description description = new Description();


        info.setDescription(description);
        return info;
    }
}
