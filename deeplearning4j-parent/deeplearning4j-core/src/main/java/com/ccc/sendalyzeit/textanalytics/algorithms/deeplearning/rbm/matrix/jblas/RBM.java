package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm.matrix.jblas;


import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseNeuralNetwork;
import static com.ccc.sendalyzeit.textanalytics.util.MatrixUtil.*;


/**
 * Restricted Boltzmann Machine
 * 
 * 
 * Based on Hinton et al.'s work 
 * 
 * Great reference:
 * http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/239
 * 
 * 
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

	/**
	 * Trains till global minimum is found.
	 * @param learningRate
	 * @param k
	 * @param input
	 */
	public void trainTillConvergence(double learningRate,int k,DoubleMatrix input) {
		if(input != null)
			this.input = input;
		double score = getReConstructionCrossEntropy();
		boolean done = false;
		int numTimesExceeded = 0;

		while(!done) {
			DoubleMatrix W = this.W.dup();
			DoubleMatrix hBias = this.hBias.dup();
			DoubleMatrix vBias = this.vBias.dup();
			contrastiveDivergence(learningRate,k,input);
			double currScore = getReConstructionCrossEntropy();
			if(currScore <= score)	{
				//increase patience
				numTimesExceeded = 0;
				double diff = Math.abs(currScore - score);
				if(diff <= 0.000001) {
					done = true;
					score = currScore;
					log.info("Converged on cost " + currScore);
					break;
				}
				else
					score = currScore;
				log.info("Found new reconstruction entropy " + score);
			}
			else if(currScore > score) {
			
				if(numTimesExceeded >= 5) {
					done = true;
					log.info("Converged on score " + score + " due to greater entropy after the last training batch");
					this.W = W;
					this.hBias = hBias;
					this.vBias = vBias;
					break;
				}
				else {
					numTimesExceeded++;
					int diff = 5 - numTimesExceeded;
					log.info("Entropy exceeded going to iterate " + diff + " more times to search for possible local minima, otherwise converging.");
					if(numTimesExceeded >= 5) {
						done = true;
						log.info("Converged on score " + score + " due to greater entropy after the last training batch");
						this.W = W;
						this.hBias = hBias;
						this.vBias = vBias;
						break;
					}
				}
				
			}

		}
	}

	/**
	 * Contrastive divergence revolves around the idea 
	 * of approximating the log likelihood around x1(input) with repeated sampling.
	 * Given is an energy based model: the higher k is (the more we sample the model)
	 * the more we lower the energy (increase the likelihood of the model)
	 * 
	 * and lower the likelihood (increase the energy) of the hidden samples.
	 * 
	 * Other insights:
	 *    CD - k involves keeping the first k samples of a gibbs sampling of the model.
	 *    
	 * @param learningRate the learning rate to scale by
	 * @param k the number of iterations to do
	 * @param input the input to sample from
	 */
	public void contrastiveDivergence(double learningRate,int k,DoubleMatrix input) {
		if(input != null)
			this.input = input;

		/*
		 * Cost and updates dictionary.
		 * This is the update rules for weights and biases
		 */
		Pair<DoubleMatrix,DoubleMatrix> probHidden = sampleHiddenGivenVisible(input);

		/*
		 * Start the gibbs sampling.
		 */
		DoubleMatrix chainStart = probHidden.getSecond();

		/*
		 * Note that at a later date, we can explore alternative methods of 
		 * storing the chain transitions for different kinds of sampling
		 * and exploring the search space.
		 */
		Pair<Pair<DoubleMatrix,DoubleMatrix>,Pair<DoubleMatrix,DoubleMatrix>> matrices = null;
		//negative visble means or expected values
		DoubleMatrix nvMeans = null;
		//negative value samples
		DoubleMatrix nvSamples = null;
		//negative hidden means or expected values
		DoubleMatrix nhMeans = null;
		//negative hidden samples
		DoubleMatrix nhSamples = null;

		/*
		 * K steps of gibbs sampling. THis is the positive phase of contrastive divergence.
		 * 
		 * There are 4 matrices being computed for each gibbs sampling.
		 * The samples from both the positive and negative phases and their expected values or averages.
		 * 
		 */

		for(int i = 0; i < k; i++) {


			if(i == 0) 
				matrices = gibbhVh(chainStart);
			else
				matrices = gibbhVh(nhSamples);
			//get the cost updates for sampling in the chain after k iterations
			nvMeans = matrices.getFirst().getFirst();
			nvSamples = matrices.getFirst().getSecond();
			nhMeans = matrices.getSecond().getFirst();
			nhSamples = matrices.getSecond().getSecond();
		}


		DoubleMatrix wAdd = input.transpose().mmul(probHidden.getSecond()).sub(nvSamples.transpose().mmul(nhMeans)).mul(learningRate);
		//update rule
		W = W.add(wAdd);
		
		regularizeWeights(input.rows, learningRate);
		//update rule: the expected values of the input - the negative samples adjusted by the learning rate
		DoubleMatrix  vBiasAdd = mean(input.sub(nvSamples), 0);
		vBias = vBiasAdd.mul(learningRate);


		//update rule: the expected values of the hidden input - the negative hidden  means adjusted by the learning rate

		DoubleMatrix hBiasAdd = mean(probHidden.getSecond().sub(nhMeans), 0);

		hBiasAdd = hBiasAdd.mul(learningRate);

		hBias = hBias.add(hBiasAdd);
	}





	/**
	 * Binomial sampling of the hidden values given visible
	 * @param v the visible values
	 * @return a binomial distribution containing the expected values and the samples
	 */
	public Pair<DoubleMatrix,DoubleMatrix> sampleHiddenGivenVisible(DoubleMatrix v) {
		DoubleMatrix h1Mean = propUp(v);
		DoubleMatrix h1Sample = binomial(h1Mean, 1, rng);
		return new Pair<DoubleMatrix,DoubleMatrix>(h1Mean,h1Sample);

	}

	/**
	 * Gibbs sampling step: hidden ---> visible ---> hidden
	 * @param h the hidden input
	 * @return the expected values and samples of both the visible samples given the hidden
	 * and the new hidden input and expected values
	 */
	public Pair<Pair<DoubleMatrix,DoubleMatrix>,Pair<DoubleMatrix,DoubleMatrix>> gibbhVh(DoubleMatrix h) {
		Pair<DoubleMatrix,DoubleMatrix> v1MeanAndSample = sampleVGivenH(h);
		DoubleMatrix vSample = v1MeanAndSample.getSecond();
		Pair<DoubleMatrix,DoubleMatrix> h1MeanAndSample = sampleHiddenGivenVisible(vSample);
		return new Pair<>(v1MeanAndSample,h1MeanAndSample);
	}


	/**
	 * Guess the visible values given the hidden
	 * @param h
	 * @return
	 */
	public Pair<DoubleMatrix,DoubleMatrix> sampleVGivenH(DoubleMatrix h) {
		DoubleMatrix v1Mean = propDown(h);
		DoubleMatrix v1Sample = binomial(v1Mean, 1, rng);
		return new Pair<>(v1Mean,v1Sample);
	}


	public DoubleMatrix propUp(DoubleMatrix v) {
		DoubleMatrix preSig = v.mmul(W);
		preSig = preSig.addRowVector(hBias);
		return sigmoid(preSig);

	}

	/**
	 * Propagates hidden down to visible
	 * @param h the hidden layer
	 * @return the approximated output of the hidden layer
	 */
	public DoubleMatrix propDown(DoubleMatrix h) {
		DoubleMatrix preSig = h.mmul(W.transpose()).addRowVector(vBias);
		return sigmoid(preSig);

	}

	/**
	 * Reconstructs the visible input.
	 * A reconstruction is a propdown of the reconstructed hidden input.
	 * @param v the visible input
	 * @return the reconstruction of the visible input
	 */
	@Override
	public DoubleMatrix reconstruct(DoubleMatrix v) {
		//reconstructed: propUp ----> hidden propDown to reconstruct
		return propDown(propUp(v));
	}

	public static class Builder extends BaseNeuralNetwork.Builder<RBM> {
		public Builder() {
			clazz =  RBM.class;
		}

	}


}
