package org.deeplearning4j.nn.updater.graph;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class ComputationGraphUpdater {

    private final Updater[] layerUpdaters;
//    private final Map<String,Updater> layerUpdatersMap;
    private final Map<String,Integer> layerUpdatersMap;

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
            layerUpdaters[i] = u;
            layerUpdatersMap.put(vertex.getVertexName(),i);
            i++;
        }
    }

    private ComputationGraphUpdater(int size, Map<String,Integer> layerUpdatersMap){
        layerUpdaters = new Updater[size];
        this.layerUpdatersMap = layerUpdatersMap;
    }

    public void update(ComputationGraph graph, Gradient gradient, int iteration, int batchSize ){
        Map<String,Gradient> layerGradients = new HashMap<>();

        //TODO user may create a name with underscore character -> will mess this up (just not allowing underscore characters would be bad too)
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

        for(Map.Entry<String,Gradient> entry : layerGradients.entrySet() ){
            String layerName = entry.getKey();
            int updaterIdx = layerUpdatersMap.get(layerName);
            layerUpdaters[updaterIdx].update(graph.getLayer(layerName),entry.getValue(),iteration,batchSize);


            //Gradients may be replaced by BaseUpdater.update()
            for( Map.Entry<String, INDArray> entry2 : layerGradients.get(layerName).gradientForVariable().entrySet() ){
                gradient.setGradientFor(entry.getKey()+"_"+entry2.getKey(), entry2.getValue());
            }
        }
    }

    public Aggregator getAggregator(boolean addThis){
        Aggregator aggregator = new Aggregator();
        if(addThis) aggregator.aggregate(this);
        return aggregator;
    }

    public static class Aggregator implements Serializable {

        private UpdaterAggregator[] aggregators;
        private Map<String,Integer> layerNamesMap;

        public void aggregate(ComputationGraphUpdater updater){
            if(aggregators == null){
                aggregators = new UpdaterAggregator[updater.layerUpdaters.length];
                for( int i=0; i<updater.layerUpdaters.length; i++ ){
                    aggregators[i] = updater.layerUpdaters[i].getAggregator(true);
                }
                layerNamesMap = new HashMap<>(updater.layerUpdatersMap);
            } else {
                if(updater.layerUpdaters == null) return;
                for( int i=0; i<aggregators.length; i++ ){
                    aggregators[i].aggregate(updater.layerUpdaters[i]);
                }
            }
        }

        public void merge(Aggregator aggregator){
            if(aggregators == null){
                aggregators = aggregator.aggregators;
            } else {
                if (aggregator.aggregators != null) {
                    for( int i=0; i<aggregators.length; i++ ){
                        aggregators[i].merge(aggregator.aggregators[i]);
                    }
                }
            }
        }

        public ComputationGraphUpdater getUpdater(){
            ComputationGraphUpdater updater = new ComputationGraphUpdater(aggregators.length,layerNamesMap);
            for( int i=0; i<aggregators.length; i++ ){
                updater.layerUpdaters[i] = aggregators[i].getUpdater();
            }
            return updater;
        }
    }

}
