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

    private DistributionUtils() {}


    public static boolean distributionsEqual(RealDistribution a, RealDistribution b) {
        if (a.getClass() != b.getClass())
            return false;
        Class<?> c = a.getClass();
        if (c == BetaDistribution.class) {
            BetaDistribution ba = (BetaDistribution) a;
            BetaDistribution bb = (BetaDistribution) b;

            return ba.getAlpha() == bb.getAlpha() && ba.getBeta() == bb.getBeta();
        } else if (c == CauchyDistribution.class) {
            CauchyDistribution ca = (CauchyDistribution) a;
            CauchyDistribution cb = (CauchyDistribution) b;
            return ca.getMedian() == cb.getMedian() && ca.getScale() == cb.getScale();
        } else if (c == ChiSquaredDistribution.class) {
            ChiSquaredDistribution ca = (ChiSquaredDistribution) a;
            ChiSquaredDistribution cb = (ChiSquaredDistribution) b;
            return ca.getDegreesOfFreedom() == cb.getDegreesOfFreedom();
        } else if (c == ExponentialDistribution.class) {
            ExponentialDistribution ea = (ExponentialDistribution) a;
            ExponentialDistribution eb = (ExponentialDistribution) b;
            return ea.getMean() == eb.getMean();
        } else if (c == FDistribution.class) {
            FDistribution fa = (FDistribution) a;
            FDistribution fb = (FDistribution) b;
            return fa.getNumeratorDegreesOfFreedom() == fb.getNumeratorDegreesOfFreedom()
                            && fa.getDenominatorDegreesOfFreedom() == fb.getDenominatorDegreesOfFreedom();
        } else if (c == GammaDistribution.class) {
            GammaDistribution ga = (GammaDistribution) a;
            GammaDistribution gb = (GammaDistribution) b;
            return ga.getShape() == gb.getShape() && ga.getScale() == gb.getScale();
        } else if (c == LevyDistribution.class) {
            LevyDistribution la = (LevyDistribution) a;
            LevyDistribution lb = (LevyDistribution) b;
            return la.getLocation() == lb.getLocation() && la.getScale() == lb.getScale();
        } else if (c == LogNormalDistribution.class) {
            LogNormalDistribution la = (LogNormalDistribution) a;
            LogNormalDistribution lb = (LogNormalDistribution) b;
            return la.getScale() == lb.getScale() && la.getShape() == lb.getShape();
        } else if (c == NormalDistribution.class) {
            NormalDistribution na = (NormalDistribution) a;
            NormalDistribution nb = (NormalDistribution) b;
            return na.getMean() == nb.getMean() && na.getStandardDeviation() == nb.getStandardDeviation();
        } else if (c == ParetoDistribution.class) {
            ParetoDistribution pa = (ParetoDistribution) a;
            ParetoDistribution pb = (ParetoDistribution) b;
            return pa.getScale() == pb.getScale() && pa.getShape() == pb.getShape();
        } else if (c == TDistribution.class) {
            TDistribution ta = (TDistribution) a;
            TDistribution tb = (TDistribution) b;
            return ta.getDegreesOfFreedom() == tb.getDegreesOfFreedom();
        } else if (c == TriangularDistribution.class) {
            TriangularDistribution ta = (TriangularDistribution) a;
            TriangularDistribution tb = (TriangularDistribution) b;
            return ta.getSupportLowerBound() == tb.getSupportLowerBound()
                            && ta.getSupportUpperBound() == tb.getSupportUpperBound() && ta.getMode() == tb.getMode();
        } else if (c == UniformRealDistribution.class) {
            UniformRealDistribution ua = (UniformRealDistribution) a;
            UniformRealDistribution ub = (UniformRealDistribution) b;
            return ua.getSupportLowerBound() == ub.getSupportLowerBound()
                            && ua.getSupportUpperBound() == ub.getSupportUpperBound();
        } else if (c == WeibullDistribution.class) {
            WeibullDistribution wa = (WeibullDistribution) a;
            WeibullDistribution wb = (WeibullDistribution) b;
            return wa.getShape() == wb.getShape() && wa.getScale() == wb.getScale();
        } else if (c == LogUniformDistribution.class ){
            LogUniformDistribution lu_a = (LogUniformDistribution)a;
            LogUniformDistribution lu_b = (LogUniformDistribution)b;
            return lu_a.getMin() == lu_b.getMin() && lu_a.getMax() == lu_b.getMax();
        } else {
            throw new UnsupportedOperationException("Unknown or not supported RealDistribution: " + c);
        }
    }

    public static boolean distributionEquals(IntegerDistribution a, IntegerDistribution b) {
        if (a.getClass() != b.getClass())
            return false;
        Class<?> c = a.getClass();

        if (c == BinomialDistribution.class) {
            BinomialDistribution ba = (BinomialDistribution) a;
            BinomialDistribution bb = (BinomialDistribution) b;
            return ba.getNumberOfTrials() == bb.getNumberOfTrials()
                            && ba.getProbabilityOfSuccess() == bb.getProbabilityOfSuccess();
        } else if (c == GeometricDistribution.class) {
            GeometricDistribution ga = (GeometricDistribution) a;
            GeometricDistribution gb = (GeometricDistribution) b;
            return ga.getProbabilityOfSuccess() == gb.getProbabilityOfSuccess();
        } else if (c == HypergeometricDistribution.class) {
            HypergeometricDistribution ha = (HypergeometricDistribution) a;
            HypergeometricDistribution hb = (HypergeometricDistribution) b;
            return ha.getPopulationSize() == hb.getPopulationSize()
                            && ha.getNumberOfSuccesses() == hb.getNumberOfSuccesses()
                            && ha.getSampleSize() == hb.getSampleSize();
        } else if (c == PascalDistribution.class) {
            PascalDistribution pa = (PascalDistribution) a;
            PascalDistribution pb = (PascalDistribution) b;
            return pa.getNumberOfSuccesses() == pb.getNumberOfSuccesses()
                            && pa.getProbabilityOfSuccess() == pb.getProbabilityOfSuccess();
        } else if (c == PoissonDistribution.class) {
            PoissonDistribution pa = (PoissonDistribution) a;
            PoissonDistribution pb = (PoissonDistribution) b;
            return pa.getMean() == pb.getMean();
        } else if (c == UniformIntegerDistribution.class) {
            UniformIntegerDistribution ua = (UniformIntegerDistribution) a;
            UniformIntegerDistribution ub = (UniformIntegerDistribution) b;
            return ua.getSupportUpperBound() == ub.getSupportUpperBound()
                            && ua.getSupportUpperBound() == ub.getSupportUpperBound();
        } else if (c == ZipfDistribution.class) {
            ZipfDistribution za = (ZipfDistribution) a;
            ZipfDistribution zb = (ZipfDistribution) b;
            return za.getNumberOfElements() == zb.getNumberOfElements() && za.getExponent() == zb.getNumberOfElements();
        } else {
            throw new UnsupportedOperationException("Unknown or not supported IntegerDistribution: " + c);
        }

    }
}
