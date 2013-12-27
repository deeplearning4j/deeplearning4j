package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm.matrix.jblas;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseNeuralNetwork;
import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;


/**
 * Continuous Restricted Boltzmann Machine
 * @author Adam Gibson
 *
 */
public class CRBM extends RBM {

	/**
	 * 
	 */
	private static final long serialVersionUID = 598767790003731193L;


	public CRBM() {}
	
	
	
	public CRBM(DoubleMatrix input, int nVisible, int nHidden, DoubleMatrix W,
			DoubleMatrix hBias, DoubleMatrix vBias, RandomGenerator rng) {
		super(input, nVisible, nHidden, W, hBias, vBias, rng);
	}
	
	
	
	public CRBM(int n_visible, int n_hidden, DoubleMatrix W,
			DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng) {
		super(n_visible, n_hidden, W, hbias, vbias, rng);
	}


	@Override
	public DoubleMatrix propDown(DoubleMatrix h) {
		DoubleMatrix preAct = h.mmul(W.transpose());
	    preAct = preAct.addRowVector(vBias);
	    return preAct;
	}
	
	@Override
	public Pair<DoubleMatrix, DoubleMatrix> sampleVGivenH(DoubleMatrix h) {
		DoubleMatrix aH = propDown(h);
		DoubleMatrix en = MatrixFunctions.exp(aH.neg());
		DoubleMatrix ep = MatrixFunctions.exp(aH);
		
		DoubleMatrix oneMinusEn = MatrixUtil.oneMinus(en);
		DoubleMatrix oneDivAh = MatrixUtil.oneDiv(aH);
		DoubleMatrix diff  = oneMinusEn.sub(oneDivAh);
		
		DoubleMatrix v1Mean = MatrixUtil.oneDiv(diff);
		
		UniformRealDistribution uDist = new UniformRealDistribution(rng,0,1);
		DoubleMatrix U = new DoubleMatrix(v1Mean.rows,v1Mean.columns);
		for(int i = 0; i < U.rows; i++)
			for(int j = 0; j < U.columns; j++) 
				U.put(i,j,uDist.sample());
		
		DoubleMatrix oneMinusEp = MatrixUtil.oneMinus(ep);
		DoubleMatrix uTimesOneMinusEp = U.mul(oneMinusEp);
		DoubleMatrix preLog = MatrixUtil.oneMinus(uTimesOneMinusEp);
		DoubleMatrix v1Sample = MatrixFunctions.log(preLog).div(aH);
		
		
		return new Pair<DoubleMatrix,DoubleMatrix>(v1Mean,v1Sample);
			
		
	    
	}
	
	
	public static class Builder extends BaseNeuralNetwork.Builder<CRBM> {
		public Builder() {
			this.clazz = CRBM.class;
		}
	}
	

	
}
