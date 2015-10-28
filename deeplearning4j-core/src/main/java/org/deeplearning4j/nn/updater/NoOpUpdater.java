package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;

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

		}

		@Override
		public INDArray getGradient(INDArray gradient, int iteration) {
			return gradient;
		}

	}
}
