package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.GradientUpdaterAggregator;

public class NoOpUpdater extends BaseUpdater {
	private NoOpGradientUpdater updater;
	
	@Override
	public void init() {

	}

	@Override
	public GradientUpdater init(String variable, INDArray gradient, Layer layer) {
		GradientUpdater updater = updaterForVariable.get(variable);
        if (updater == null) {
            updater = new NoOpGradientUpdater();
            updaterForVariable.put(variable,updater);
        }
        return updater;
	}

	private static class NoOpGradientUpdater implements GradientUpdater {
		@Override
		public void update(Object... args) {
			//No op
		}

		@Override
		public INDArray getGradient(INDArray gradient, int iteration) {
			return gradient;
		}

		@Override
		public GradientUpdaterAggregator getAggregator(boolean addThis) {
			return new NoOpUpdaterAggregator();
		}
	}

	@Override
	public UpdaterAggregator getAggregator(boolean addThis){
		return new NoOpAggregator();
	}

	protected static class NoOpAggregator extends BaseUpdater.UpdaterAggregatorImpl {
		@Override
		public Updater getUpdater() {
			return setUpdaterState(new NoOpUpdater());
		}
	}

	private static class NoOpUpdaterAggregator implements GradientUpdaterAggregator{
		@Override
		public GradientUpdater getUpdater() {
			return new NoOpGradientUpdater();
		}

		@Override
		public void aggregate(GradientUpdater updater) {
			//No op
		}

		@Override
		public GradientUpdaterAggregator combine(GradientUpdaterAggregator other) {
			return this;
		}
	}


}
