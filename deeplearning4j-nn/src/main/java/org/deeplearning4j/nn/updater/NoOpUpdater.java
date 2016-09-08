package org.deeplearning4j.nn.updater;

import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.GradientUpdaterAggregator;

/**
 * @deprecated As of 0.6.0. Use {@link LayerUpdater instead}
 */
@Deprecated
public class NoOpUpdater extends BaseUpdater {
	private NoOpGradientUpdater updater;
	
	@Override
	public void init() {

	}

	@Override
	public GradientUpdater init(String variable, Layer layer) {
		GradientUpdater updater = updaterForVariable.get(variable);
        if (updater == null) {
            updater = new NoOpGradientUpdater();
            updaterForVariable.put(variable,updater);
        }
        return updater;
	}

	@EqualsAndHashCode
	private static class NoOpGradientUpdater implements GradientUpdater {
		@Override
		public int stateSizeForInputSize(int inputSize) {
			return 0;
		}

		@Override
		public void setStateViewArray(INDArray viewArray, int[] shape, char order, boolean initialize) {
			//No op
		}

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

	@EqualsAndHashCode
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
