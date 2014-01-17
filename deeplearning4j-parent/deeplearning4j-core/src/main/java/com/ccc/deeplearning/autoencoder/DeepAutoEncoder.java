package com.ccc.deeplearning.autoencoder;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;

public class DeepAutoEncoder implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3571832097247806784L;
	private BaseMultiLayerNetwork encoder;
	private BaseMultiLayerNetwork decoder;
	private Object[] trainingParams;

	public DeepAutoEncoder(BaseMultiLayerNetwork encoder,Object[] trainingParams) {
		this.encoder = encoder;
		this.trainingParams = trainingParams;
	}


	public void train(DoubleMatrix input,DoubleMatrix labels,double lr) {
		encoder.trainNetwork(input, labels, trainingParams);
		decoder = new BaseMultiLayerNetwork.Builder<>().withClazz(encoder.getClass()).buildEmpty();
		decoder.asDecoder(encoder);
		DoubleMatrix encoderInput = encoder.predict(input);
		DoubleMatrix encoderLabels = input;
		decoder.trainNetwork(encoderInput, encoderLabels, trainingParams);

	}


	public DoubleMatrix encode(DoubleMatrix input) {
		return encoder.predict(input);
	}

	public DoubleMatrix decode(DoubleMatrix input) {
		return decoder.predict(input);
	}



}
