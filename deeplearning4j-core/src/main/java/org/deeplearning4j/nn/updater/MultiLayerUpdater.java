package org.deeplearning4j.nn.updater;

import java.util.Map;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;

/**MultiLayerUpdater: Gradient updater for MultiLayerNetworks.
 * Expects backprop gradients for all layers to be in single Gradient object,
 * keyed by "0_b", "1_w" etc., as per MultiLayerNetwork.backward()
 */
@EqualsAndHashCode
public class MultiLayerUpdater implements Updater {
	private final Updater[] layerUpdaters;
	
	public MultiLayerUpdater( MultiLayerNetwork network ){
		Layer[] layers = network.getLayers();
		layerUpdaters = new Updater[layers.length];
		for( int i=0; i<layers.length; i++ ){
			layerUpdaters[i] = UpdaterCreator.getUpdater(layers[i]);
		}
	}

	public MultiLayerUpdater(MultiLayerUpdater updater){
		layerUpdaters = new Updater[updater.layerUpdaters.length];
		for( int i=0; i<updater.layerUpdaters.length; i++ ){
			layerUpdaters[i] = updater.layerUpdaters[i].clone();
		}
	}

	private MultiLayerUpdater(int size){
		layerUpdaters = new Updater[size];
	}

	@Override
	public void update(Layer layer, Gradient gradient, int iteration, int batchSize) {
		MultiLayerNetwork mln = (MultiLayerNetwork)layer;
		
		Gradient[] layerGradients = new Gradient[layerUpdaters.length];
		for( int i=0; i<layerGradients.length; i++)
			layerGradients[i] = new DefaultGradient();
		
		for(Map.Entry<String,INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
			String key = gradientPair.getKey();
			int idx = key.indexOf("_");
			if( idx == -1 ) throw new IllegalStateException("Invalid key: MuliLayerNetwork Gradient key does not have layer separator: \""+key+"\"");
			int layerIdx = Integer.parseInt(key.substring(0, idx));
			
			String newKey = key.substring(idx + 1);
			layerGradients[layerIdx].gradientForVariable().put(newKey, gradientPair.getValue());
        }
		
		for( int i = 0; i < layerUpdaters.length; i++ ) {
			layerUpdaters[i].update(mln.getLayer(i), layerGradients[i], iteration, batchSize);
			//Gradients may be replaced by BaseUpdater.update()
			for( Map.Entry<String, INDArray> entry : layerGradients[i].gradientForVariable().entrySet() ){
				gradient.setGradientFor(i+"_"+entry.getKey(), entry.getValue());
			}
		}
	}

	@Override
	public UpdaterAggregator getAggregator(boolean addThis) {
		MultiLayerUpdaterAggregator ag = new MultiLayerUpdaterAggregator();
		if(addThis) ag.aggregate(this);
		return ag;
	}

	protected static class MultiLayerUpdaterAggregator implements UpdaterAggregator {

		private UpdaterAggregator[] aggregators;

		@Override
		public void aggregate(Updater updater) {
			MultiLayerUpdater mlu = (MultiLayerUpdater)updater;
			if (aggregators == null) {
				aggregators = new UpdaterAggregator[mlu.layerUpdaters.length];
				for( int i=0; i<aggregators.length; i++ ){
					aggregators[i] = mlu.layerUpdaters[i].getAggregator(true);
				}
			} else {
				if(mlu.layerUpdaters == null) return;
				for( int i=0; i<aggregators.length; i++ ){
					aggregators[i].aggregate(mlu.layerUpdaters[i]);
				}
			}
		}

		@Override
		public void merge(UpdaterAggregator aggregator) {
			MultiLayerUpdaterAggregator mlua = (MultiLayerUpdaterAggregator)aggregator;
			if(aggregators == null){
				aggregators = mlua.aggregators;
			} else {
				if (mlua.aggregators != null) {
					for(int i=0; i<aggregators.length; i++ ){
						aggregators[i].merge(mlua.aggregators[i]);
					}
				}
			}
		}

		@Override
		public Updater getUpdater() {
			MultiLayerUpdater multiLayerUpdater = new MultiLayerUpdater(aggregators.length);
			for( int i=0; i<aggregators.length; i++ ){
				multiLayerUpdater.layerUpdaters[i] = aggregators[i].getUpdater();
			}
			return multiLayerUpdater;
		}
	}

	@Override
	public Updater clone(){
		return new MultiLayerUpdater(this);
	}
}
