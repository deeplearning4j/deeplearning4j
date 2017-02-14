package org.deeplearning4j.arbiter.optimize.distribution;

import org.apache.commons.math3.distribution.*;

/**
 * Distribution utils for Apache Commons math distributions - which don't provide equals, hashcode, toString methods,
 * don't implement serializable etc.
 * Which makes unit testing etc quite difficult.
 *
 * @author Alex Black
 */
public class DistributionUtils {

    private DistributionUtils(){ }


    public static boolean distributionsEqual(RealDistribution a, RealDistribution b){
        if(a.getClass() != b.getClass()) return false;
        Class<?> c = a.getClass();
        if(c == BetaDistribution.class){
            BetaDistribution ba = (BetaDistribution)a;
            BetaDistribution bb = (BetaDistribution)b;

            return ba.getAlpha() == bb.getAlpha() && ba.getBeta() == bb.getBeta();
        } else if( c == CauchyDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == ChiSquaredDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == ExponentialDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == FDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == GammaDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == LevyDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == LogNormalDistribution.class ){
            LogNormalDistribution la = (LogNormalDistribution)a;
            LogNormalDistribution lb = (LogNormalDistribution)b;
            return la.getScale() == lb.getScale() && la.getShape() == lb.getShape();
        } else if( c == NormalDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == ParetoDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == TDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == TriangularDistribution.class ){
            throw new UnsupportedOperationException("Not yet implemented");
        } else if( c == UniformRealDistribution.class ){
            UniformRealDistribution ua = (UniformRealDistribution)a;
            UniformRealDistribution ub = (UniformRealDistribution)b;
            return ua.getSupportLowerBound() == ub.getSupportLowerBound() && ua.getSupportUpperBound() == ub.getSupportUpperBound();
        } else if( c == WeibullDistribution.class ) {
            throw new UnsupportedOperationException("Not yet implemented");
        } else {
            throw new UnsupportedOperationException("Unknown or not supported RealDistribution: " + c);
        }
    }

}
