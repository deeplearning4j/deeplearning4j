package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm.matrix.jblas;


import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseNeuralNetwork;
import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;


/**
 * Restricted Boltzmann Machine
 * @author Adam Gibson
 *
 */
public class RBM extends BaseNeuralNetwork {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6189188205731511957L;
	private static Logger log = LoggerFactory.getLogger(RBM.class);

	public RBM() {}
	
	public RBM(int nVisible, int nHidden, DoubleMatrix W, DoubleMatrix hbias,
			DoubleMatrix vbias, RandomGenerator rng) {
		super(nVisible, nHidden, W, hbias, vbias, rng);
	}


	public RBM(DoubleMatrix input, int n_visible, int n_hidden, DoubleMatrix W,
			DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng) {
		super(input, n_visible, n_hidden, W, hbias, vbias, rng);
	}


	public void contrastiveDivergence(double learningRate,int k,DoubleMatrix input) {
		if(input != null)
			this.input = input;
		Pair<DoubleMatrix,DoubleMatrix> probHidden = this.sampleHiddenGivenVisible(this.input);
        
		DoubleMatrix chainStart = probHidden.getSecond();
		Pair<Pair<DoubleMatrix,DoubleMatrix>,Pair<DoubleMatrix,DoubleMatrix>> matrices = null;
		DoubleMatrix nvMeans = null;
		DoubleMatrix nvSamples = null;
		DoubleMatrix nhMeans = null;
		DoubleMatrix nhSamples = null;
		for(int i = 0; i < k; i++) {


			if(i == 0) 
				matrices = gibbhVh(chainStart);
			else
				matrices = gibbhVh(nhSamples);

			nvMeans = matrices.getFirst().getFirst();
			nvSamples = matrices.getFirst().getSecond();
			nhMeans = matrices.getSecond().getFirst();
			nhSamples = matrices.getSecond().getSecond();
		}

		DoubleMatrix inputTimesPhSample =  this.input.transpose().mmul(probHidden.getSecond());
		DoubleMatrix nvSamplesTTimesNhMeans = nvSamples.transpose().mmul(nhMeans);
		DoubleMatrix diff = inputTimesPhSample.sub(nvSamplesTTimesNhMeans);
		DoubleMatrix wAdd = diff.mul(learningRate);

		W = W.add(wAdd);


		DoubleMatrix  vBiasAdd = MatrixUtil.mean(this.input.sub(nvSamples), 0);
		vBias = vBiasAdd.mul(learningRate);


		DoubleMatrix hBiasAdd = MatrixUtil.mean(probHidden.getSecond().sub(nhMeans), 0);


		hBiasAdd = hBiasAdd.mul(learningRate);

		hBias = hBias.add(hBiasAdd);
	}


	public double getReConstructionCrossEntropy() {
		DoubleMatrix preSigH = input.mmul(W).add(hBias);
		DoubleMatrix sigH = MatrixUtil.sigmoid(preSigH);

		DoubleMatrix preSigV = sigH.mmul(W.transpose()).add(vBias);
		DoubleMatrix sigV = MatrixUtil.sigmoid(preSigV);


		DoubleMatrix logSigV = MatrixFunctions.log(sigV);
		DoubleMatrix oneMinusSigV = DoubleMatrix.ones(sigV.rows,sigV.columns).sub(sigV);

		DoubleMatrix logOneMinusSigV = MatrixFunctions.log(oneMinusSigV);
		DoubleMatrix inputTimesLogSigV = input.mul(logSigV);


		DoubleMatrix oneMinusInput = DoubleMatrix.ones(input.rows,input.columns).min(input);

		DoubleMatrix crossEntropyMatrix = MatrixUtil.mean(inputTimesLogSigV.add(oneMinusInput).mul(logOneMinusSigV).rowSums(),1);

		return -crossEntropyMatrix.mean();
	}


	public DoubleMatrix reconstruct(DoubleMatrix v) {
		DoubleMatrix h = MatrixUtil.sigmoid(v.mmul(W).addRowVector(hBias));
		DoubleMatrix reconstructedV = MatrixUtil.sigmoid(h.mmul(W.transpose()).addRowVector(vBias));
		return reconstructedV;
	}


	public Pair<DoubleMatrix,DoubleMatrix> sampleHiddenGivenVisible(DoubleMatrix v) {
		DoubleMatrix h1Mean = propUp(v);
		DoubleMatrix h1Sample = MatrixUtil.binomial(h1Mean, 1, rng);
		return new Pair<DoubleMatrix,DoubleMatrix>(h1Mean,h1Sample);

	}

	public Pair<Pair<DoubleMatrix,DoubleMatrix>,Pair<DoubleMatrix,DoubleMatrix>> gibbhVh(DoubleMatrix h) {
		Pair<DoubleMatrix,DoubleMatrix> v1MeanAndSample = this.sampleVGivenH(h);
		DoubleMatrix vSample = v1MeanAndSample.getSecond();
		Pair<DoubleMatrix,DoubleMatrix> h1MeanAndSample = this.sampleHiddenGivenVisible(vSample);
		return new Pair<>(v1MeanAndSample,h1MeanAndSample);
	}



	public Pair<DoubleMatrix,DoubleMatrix> sampleVGivenH(DoubleMatrix h) {
		DoubleMatrix v1Mean = propDown(h);
		DoubleMatrix v1Sample = MatrixUtil.binomial(v1Mean, 1, rng);
		return new Pair<DoubleMatrix,DoubleMatrix>(v1Mean,v1Sample);
	}


	public DoubleMatrix propUp(DoubleMatrix v) {
		DoubleMatrix preSig = v.mmul(W);
		preSig = preSig.addRowVector(hBias);
		return MatrixUtil.sigmoid(preSig);

	}

	public DoubleMatrix propDown(DoubleMatrix h) {
		DoubleMatrix preSig = h.mmul(W.transpose()).addRowVector(vBias);
		return MatrixUtil.sigmoid(preSig);

	}


	public static class Builder extends BaseNeuralNetwork.Builder<RBM> {
		public Builder() {
			this.clazz =  RBM.class;
		}

	}


}
