package org.deeplearning4j.optimize.solvers;

/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
http://www.cs.umass.edu/~mccallum/mallet
This software is provided under the terms of the Common Public License,
version 1.0, as published by http://www.opensource.org.  For further
information, see the file `LICENSE' included with this distribution. */


/**
 @author Aron Culotta <a href="mailto:culotta@cs.umass.edu">culotta@cs.umass.edu</a>

 Adapted from mallet with original authors above.
 Modified to be a vectorized version that uses jblas matrices
 for computation rather than the mallet ops.

 */

/**
 Numerical Recipes in C: p.385. lnsrch. A simple backtracking line
 search. No attempt at accurately finding the true minimum is
 made. The goal is only to ensure that BackTrackLineSearch will
 return a position of higher value.

 @author Adam Gibson


 */

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.deeplearning4j.optimize.api.OptimizableByGradientValueMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


//"Line Searches and Backtracking", p385, "Numeric Recipes in C"

public class VectorizedBackTrackLineSearchMinimum
{
    private static Logger logger = LoggerFactory.getLogger(VectorizedBackTrackLineSearchMinimum.class.getName());

    OptimizableByGradientValueMatrix function;

    public VectorizedBackTrackLineSearchMinimum(OptimizableByGradientValueMatrix optimizable) {
        this.function = optimizable;
    }

    final int maxIterations = 100;
    double stpmax = 100;
    final double EPS = 3.0e-12;

    // termination conditions: either
    //   a) abs(delta x/x) < REL_TOLX for all coordinates
    //   b) abs(delta x) < ABS_TOLX for all coordinates
    //   c) sufficient function increase (uses ALF)
    private double relTolx = 1e-10;
    private double absTolx = 1e-4; // tolerance on absolute value difference
    final double ALF = 1e-4;

    public void setStpmax(double stpmax) {
        this.stpmax = stpmax;
    }


    public double getStpmax() {
        return stpmax;
    }

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
    public double optimize (INDArray line, int lineSearchIteration,INDArray g,INDArray params)
    {
        INDArray x, oldParameters;
        double slope, test, alamin, alam, alam2, tmplam;
        double rhs1, rhs2, a, b, disc, oldAlam;
        double f, fold, f2;
        oldParameters = params.dup();


        alam2 = tmplam = 0.0;
        f2 = fold = function.getValue();
        if (logger.isDebugEnabled()) {
            logger.trace ("ENTERING BACKTRACK\n");
            logger.trace("Entering BackTrackLnSrch, value=" + fold +",\ndirection.oneNorm:"
                    +	line.norm1(Integer.MAX_VALUE) + "  direction.infNorm:"+ FastMath.max(Double.NEGATIVE_INFINITY, (double) Transforms.abs(line).max(Integer.MAX_VALUE).element()));
        }

        LinAlgExceptions.assertValidNum(g);
        double sum = (double) line.norm2(Integer.MAX_VALUE).element();
//        if(sum > stpmax) {
//            logger.warn("attempted step too big. scaling: sum= " + sum +
//                    ", stpmax= "+ stpmax);
//            line.muli(stpmax / sum);
//        }

        //dot product
        slope = Nd4j.getBlasWrapper().dot(g, line);
        logger.debug("slope = " + slope);

        // find maximum lambda
        // converge when (delta x) / x < REL_TOLX for all coordinates.
        //  the largest step size that triggers this threshold is
        //  precomputed and saved in alamin
        INDArray maxOldParams = Nd4j.create(line.length());
        for(int i = 0;i < line.length(); i++) {
            maxOldParams.putScalar(i,Math.max(Math.abs((double) oldParameters.getScalar(i).element()), 1.0));

        }

        INDArray testMatrix = Transforms.abs(line).div(maxOldParams);

        test = (double) testMatrix.max(Integer.MAX_VALUE).element();


        alamin = relTolx / test;

        alam  = 1.0;
        oldAlam = 0.0;
        int iteration = 0;
        // look for step size in direction given by "line"
        for(iteration = 0; iteration < maxIterations; iteration++) {
            function.setCurrentIteration(lineSearchIteration);
            // x = oldParameters + alam*line
            // initially, alam = 1.0, i.e. take full Newton step
            logger.trace("BackTrack loop iteration " + iteration +" : alam="+
                    alam+" oldAlam=" + oldAlam);
            logger.trace ("before step, x.1norm: " + params.norm1(Integer.MAX_VALUE) +
                    "\nalam: " + alam + "\noldAlam: " + oldAlam);
            assert(alam != oldAlam) : "alam == oldAlam";



            params.addi(line.mul(alam - oldAlam));  // step

            logger.debug ("after step, x.1norm: " + params.norm1(Integer.MAX_VALUE));

            // check for convergence
            //convergence on delta x
            if ((alam < alamin) || smallAbsDiff (oldParameters, params)) {
                //				if ((alam < alamin)) {
                function.setParameters(oldParameters);
                f = function.getValue();
                logger.trace("EXITING BACKTRACK: Jump too small (alamin="+ alamin + "). Exiting and using xold. Value=" + f);
                return 0.0;
            }

            function.setParameters(params);
            oldAlam = alam;
            f = function.getValue();

            logger.debug("value = " + f);

            // sufficient function increase (Wolf condition)
            if(f >= fold + ALF * alam * slope) {

                logger.debug("EXITING BACKTRACK: value=" + f);

                if (f<fold)
                    throw new IllegalStateException
                            ("Function did not increase: f=" + f +
                                    " < " + fold + "=fold");
                return alam;
            }


            // if value is infinite, i.e. we've
            // jumped to unstable territory, then scale down jump
            else if(Double.isInfinite(f) || Double.isInfinite(f2)) {
                logger.warn ("Value is infinite after jump " + oldAlam + ". f="+ f +", f2=" + f2 + ". Scaling back step size...");
                tmplam = .2 * alam;
                if(alam < alamin) { //convergence on delta x
                    function.setParameters(oldParameters);
                    f = function.getValue();
                    logger.warn("EXITING BACKTRACK: Jump too small. Exiting and using xold. Value="+ f );
                    return 0.0;
                }
            }
            else { // backtrack
                if(alam == 1.0) // first time through
                    tmplam = -slope / (2.0 * ( f - fold - slope ));
                else {
                    rhs1 = f - fold - alam * slope;
                    rhs2 = f2 - fold - alam2 * slope;
                    assert((alam - alam2) != 0): "FAILURE: dividing by alam-alam2. alam="+alam;
                    a = ( rhs1 / (FastMath.pow(alam, 2)) - rhs2 / ( FastMath.pow(alam2, 2) )) / (alam-alam2);
                    b = ( -alam2* rhs1/( alam* alam ) + alam * rhs2 / ( alam2 *  alam2 )) / ( alam - alam2);
                    if(a == 0.0)
                        tmplam = -slope / (2.0 * b);
                    else {
                        disc = b * b - 3.0 * a * slope;
                        if(disc < 0.0) {
                            tmplam = .5 * alam;
                        }
                        else if (b <= 0.0)
                            tmplam = (-b + FastMath.sqrt(disc))/(3.0 * a );
                        else
                            tmplam = -slope / (b + FastMath.sqrt(disc));
                    }
                    if (tmplam > .5 * alam)
                        tmplam = .5 * alam;    // lambda <= .5 lambda_1
                }
            }

            alam2 = alam;
            f2 = f;
            logger.debug("tmplam:" + tmplam);
            alam = Math.max(tmplam, .1*alam);  // lambda >= .1*Lambda_1

        }

        if(iteration >= maxIterations)
            throw new IllegalStateException ("Too many iterations.");
        return 0.0;
    }

    // returns true iff we've converged based on absolute x difference
    private boolean smallAbsDiff (INDArray x, INDArray xold)
    {

        for (int i = 0; i < x.length(); i++) {
            double comp = Math.abs ((double) x.getScalar(i).sub(xold.getScalar(i)).element());
            if ( comp > absTolx) {
                return false;
            }
        }
        return true;
    }

}

