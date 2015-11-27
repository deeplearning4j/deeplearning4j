package org.deeplearning4j.nn.updater;

import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/**MultiLayerUpdater: Gradient updater for MultiLayerNetworks.
 * Expects backprop gradients for all layers to be in single Gradient object,
 * keyed by "0_b", "1_w" etc., as per MultiLayerNetwork.backward()
 */
public class MultiLayerUpdater implements Updater {
	
	private final Updater[] layerUpdaters; 
	
	public MultiLayerUpdater( MultiLayerNetwork network ){
		Layer[] layers = network.getLayers();
		layerUpdaters = new Updater[layers.length];
		for( int i=0; i<layers.length; i++ ){
			layerUpdaters[i] = UpdaterCreator.getUpdater(layers[i]);
		}
	}

	@Override
	public void update(Layer layer, Gradient gradient, int iteration, int batchSize) {
		MultiLayerNetwork mln = (MultiLayerNetwork)layer;
		
		Gradient[] layerGradients = new Gradient[layerUpdaters.length];
		for( int i=0; i<layerGradients.length; i++ ) layerGradients[i] = new DefaultGradient();
		
		for(Map.Entry<String,INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
			String key = gradientPair.getKey();
			int idx = key.indexOf("_");
			if( idx == -1 ) throw new IllegalStateException("Invalid key: MuliLayerNetwork Gradient key does not have layer separator: \""+key+"\"");
			int layerIdx = Integer.parseInt(key.substring(0, idx));
			
			String newKey = key.substring(idx+1);
			layerGradients[layerIdx].gradientForVariable().put(newKey, gradientPair.getValue());
        }
		
		for( int i=0; i<layerUpdaters.length; i++ ){
			layerUpdaters[i].update(mln.getLayer(i), layerGradients[i], iteration, batchSize);
			//Gradients may be replaced by BaseUpdater.update()
			for( Map.Entry<String, INDArray> entry : layerGradients[i].gradientForVariable().entrySet() ){
				gradient.setGradientFor(i+"_"+entry.getKey(), entry.getValue());
			}
		}
	}

}
