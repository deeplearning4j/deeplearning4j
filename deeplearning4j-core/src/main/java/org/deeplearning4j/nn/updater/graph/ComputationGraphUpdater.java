package org.deeplearning4j.nn.updater.graph;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

public class ComputationGraphUpdater {

    private final Updater[] layerUpdaters;
    private final Map<String,Updater> layerUpdatersMap;

    public ComputationGraphUpdater(ComputationGraph graph){
        layerUpdaters = new Updater[graph.getNumLayers()];
        layerUpdatersMap = new HashMap<>();
        //TODO make this more efficient
        GraphVertex[] vertices = graph.getVertices();

        int i=0;
        for (GraphVertex vertex : vertices) {
            if (!vertex.hasLayer()) continue;
            Layer layer = vertex.getLayer();
            Updater u = UpdaterCreator.getUpdater(layer);
            layerUpdaters[i++] = u;
            layerUpdatersMap.put(vertex.getVertexName(),u);
        }
    }

    public void update(ComputationGraph graph, Gradient gradient, int iteration, int batchSize ){

        Map<String,Gradient> layerGradients = new HashMap<>();

        //TODO user may create a name with underscore character -> will mess this up (just not allowing underscore characters would be bad too)
        //For computationGraph: expect
        for(Map.Entry<String,INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
            String key = gradientPair.getKey();
            int idx = key.indexOf("_");
            if( idx == -1 ) throw new IllegalStateException("Invalid key: ComputationGraph Gradient key does not have layer separator: \""+key+"\"");

            String layerName = key.substring(0,idx);

            Gradient g = layerGradients.get(layerName);
            if(g == null){
                g = new DefaultGradient();
                layerGradients.put(layerName,g);
            }

            String newKey = key.substring(idx + 1);
            g.setGradientFor(newKey, gradientPair.getValue());
        }

//        for( int i = 0; i < layerUpdaters.length; i++ ) {
        for(Map.Entry<String,Gradient> entry : layerGradients.entrySet() ){
            String layerName = entry.getKey();
            layerUpdatersMap.get(layerName).update(graph.getLayer(layerName),entry.getValue(),iteration,batchSize);

            //Gradients may be replaced by BaseUpdater.update()
            for( Map.Entry<String, INDArray> entry2 : layerGradients.get(layerName).gradientForVariable().entrySet() ){
                gradient.setGradientFor(entry.getKey()+"_"+entry2.getKey(), entry2.getValue());
            }
        }
    }

}
