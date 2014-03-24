package org.deeplearning4j.util;

/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
http://www.cs.umass.edu/~mccallum/mallet
This software is provided under the terms of the Common Public License,
version 1.0, as published by http://www.opensource.org.  For further
information, see the file `LICENSE' included with this distribution. */


/** 
@author Aron Culotta <a href="mailto:culotta@cs.umass.edu">culotta@cs.umass.edu</a>
*/

/**
	 Numerical Recipes in C: p.385. lnsrch. A simple backtracking line
	 search. No attempt at accurately finding the true minimum is
	 made. The goal is only to ensure that BackTrackLineSearch will
	 return a position of higher value.
*/

import java.util.logging.Level;
import java.util.logging.Logger;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import cc.mallet.optimize.InvalidOptimizableException;

//"Line Searches and Backtracking", p385, "Numeric Recipes in C"

public class VectorizedBackTrackLineSearch implements LineOptimizerMatrix
{
	private static Logger logger = Logger.getLogger(VectorizedBackTrackLineSearch.class.getName());
	
	OptimizableByGradientValueMatrix function;
	
	public VectorizedBackTrackLineSearch (OptimizableByGradientValueMatrix optimizable) {
		this.function = optimizable;
	}

	final int maxIterations = 100;
	final double stpmax = 100;
	final double EPS = 3.0e-12;

	// termination conditions: either
	//   a) abs(delta x/x) < REL_TOLX for all coordinates
	//   b) abs(delta x) < ABS_TOLX for all coordinates
	//   c) sufficient function increase (uses ALF)
	private double relTolx = 1e-7;
	private double absTolx = 1e-4; // tolerance on absolute value difference
	final double ALF = 1e-4;


	/** 
	 * Sets the tolerance of relative diff in function value.
	 *  Line search converges if <tt>abs(delta x / x) < tolx</tt>
	 *  for all coordinates. */
	public void setRelTolx (double tolx) { relTolx = tolx; }        

	/** 
	 * Sets the tolerance of absolute diff in function value.
	 *  Line search converges if <tt>abs(delta x) < tolx</tt>
	 *  for all coordinates. */
	public void setAbsTolx (double tolx) { absTolx = tolx; }        

	// initialStep is ignored.  This is b/c if the initial step is not 1.0,
	//   it sometimes confuses the backtracking for reasons I don't 
	//   understand.  (That is, the jump gets LARGER on iteration 1.)

	// returns fraction of step size (alam) if found a good step
	// returns 0.0 if could not step in direction
	public double optimize (DoubleMatrix line, double initialStep)
	{
		DoubleMatrix g, x, oldParameters;
		double slope, newSlope, temp, test, alamin, alam, alam2, tmplam;
		double rhs1, rhs2, a, b, disc, oldAlam;
		double f, fold, f2;
		g = new DoubleMatrix(function.getNumParameters()); // gradient
		x = new DoubleMatrix(function.getNumParameters()); // parameters
		oldParameters = new DoubleMatrix(function.getNumParameters());
		x = function.getParameters ();
		System.arraycopy (x, 0, oldParameters, 0, x.length);
		g = function.getValueGradient();
		alam2 = tmplam = 0.0; 
		f2 = fold = function.getValue();
		if (logger.isLoggable(Level.FINE)) {
			logger.fine ("ENTERING BACKTRACK\n");
			logger.fine("Entering BackTrackLnSrch, value="+fold+",\ndirection.oneNorm:"
					+	line.norm1() + "  direction.infNorm:"+ Math.max(Double.NEGATIVE_INFINITY,MatrixFunctions.abs(line).max()));
		}
		assert (!MatrixUtil.isNaN(g));
		double sum = line.norm1();
		if(sum > stpmax) {
			logger.warning("attempted step too big. scaling: sum="+sum+
					", stpmax="+stpmax);
			line.muli(stpmax / sum);
		}
		//dot product
		newSlope = slope = g.mmul(line).sum();
		logger.fine("slope="+slope);

		if (slope<0) 
			throw new InvalidOptimizableException ("Slope = " + slope + " is negative");
		if (slope == 0)
			throw new InvalidOptimizableException ("Slope = " + slope + " is zero");

		// find maximum lambda
		// converge when (delta x) / x < REL_TOLX for all coordinates.
		//  the largest step size that triggers this threshold is
		//  precomputed and saved in alamin
		test = 0.0;
		for(int i=0; i<oldParameters.length; i++) {
			temp = Math.abs(line.get(i)) /
			Math.max(Math.abs(oldParameters.get(i)), 1.0);
			if(temp > test) test = temp;
		}

		alamin = relTolx/test;
		alam  = 1.0;
		oldAlam = 0.0;
		int iteration = 0;
		// look for step size in direction given by "line"
		for(iteration=0; iteration < maxIterations; iteration++) {
			// x = oldParameters + alam*line
			// initially, alam = 1.0, i.e. take full Newton step
			logger.fine("BackTrack loop iteration "+iteration+": alam="+
					alam+" oldAlam="+oldAlam);
			logger.fine ("before step, x.1norm: " + x.norm1() +
					"\nalam: " + alam + "\noldAlam: " + oldAlam);
			assert(alam != oldAlam) : "alam == oldAlam";
			x.addi(line.mul(alam - oldAlam));  // step
			
			logger.fine ("after step, x.1norm: " + x.norm1());

			// check for convergence 
			//convergence on delta x
			if ((alam < alamin) || smallAbsDiff (oldParameters, x)) {
//				if ((alam < alamin)) {
				function.setParameters(oldParameters);
				f = function.getValue();
				logger.warning("EXITING BACKTRACK: Jump too small (alamin="+alamin+"). Exiting and using xold. Value="+f);
				return 0.0;
			}

			function.setParameters(x);
			oldAlam = alam;
			f = function.getValue();

			logger.fine("value="+f);

			// sufficient function increase (Wolf condition)
			if(f >= fold+ALF*alam*slope) { 

				logger.fine("EXITING BACKTRACK: value="+f);

				if (f<fold) 
					throw new IllegalStateException
					("Function did not increase: f=" + f + 
							" < " + fold + "=fold");				
				return alam;
			}
			// if value is infinite, i.e. we've
			// jumped to unstable territory, then scale down jump
			else if(Double.isInfinite(f) || Double.isInfinite(f2)) {
				logger.warning ("Value is infinite after jump " + oldAlam + ". f="+f+", f2="+f2+". Scaling back step size...");
				tmplam = .2 * alam;					
				if(alam < alamin) { //convergence on delta x
					function.setParameters(oldParameters);
					f = function.getValue();
					logger.warning("EXITING BACKTRACK: Jump too small. Exiting and using xold. Value="+f);
					return 0.0;
				}
			}
			else { // backtrack
				if(alam == 1.0) // first time through
					tmplam = -slope / (2.0 * ( f - fold - slope ));
				else {
					rhs1 = f-fold-alam*slope;
					rhs2 = f2-fold-alam2*slope;
					assert((alam - alam2) != 0): "FAILURE: dividing by alam-alam2. alam="+alam;
					a = (rhs1/(alam*alam)-rhs2/( alam2* alam2 ))/(alam-alam2);
					b = (-alam2*rhs1/(alam*alam)+ alam * rhs2/( alam2 *  alam2))/(alam - alam2);
					if(a == 0.0) 
						tmplam = -slope / (2.0 * b);
					else {
						disc = b * b - 3.0 * a * slope;
						if(disc < 0.0) {
							tmplam = .5 * alam;
						}
						else if (b <= 0.0)
							tmplam = (-b+Math.sqrt(disc))/(3.0 * a );
						else tmplam = -slope/(b + Math.sqrt(disc));
					}
					if (tmplam > .5*alam)
						tmplam = .5*alam;    // lambda <= .5 lambda_1
				}
			}
			alam2 = alam;
			f2 = f;
			logger.fine("tmplam:"+tmplam);
			alam = Math.max(tmplam, .1*alam);  // lambda >= .1*Lambda_1						
		}
		if(iteration >= maxIterations) 
			throw new IllegalStateException ("Too many iterations.");
		return 0.0;
	}

	// returns true iff we've converged based on absolute x difference 
	private boolean smallAbsDiff (DoubleMatrix x, DoubleMatrix xold)
	{
		for (int i = 0; i < x.length; i++) {
			if (Math.abs (x.get(i) - x.get(i)) > absTolx) {
				return false;
			}
		}
		return true;
	}

}

