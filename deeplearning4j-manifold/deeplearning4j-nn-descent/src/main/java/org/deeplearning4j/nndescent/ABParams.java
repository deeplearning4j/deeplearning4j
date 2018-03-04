package org.deeplearning4j.nndescent;

import lombok.Builder;
import org.apache.commons.math3.analysis.UnivariateVectorFunction;
import org.apache.commons.math3.analysis.differentiation.FiniteDifferencesDifferentiator;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.fitting.leastsquares.MultivariateJacobianFunction;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.indexing.conditions.GreaterThan;
import org.nd4j.linalg.indexing.conditions.GreaterThanOrEqual;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;

@Builder
public class ABParams {
    private double spread;
    private double minDistance;

    public INDArray[] solve() {
        INDArray xv = Nd4j.linspace(0,spread * 3,300);
        INDArray yv = Nd4j.zeros(xv.shape());
        int[] indices = BooleanIndexing.chooseFrom(new INDArray[]{xv}, Arrays.asList(0.0),new ArrayList<>(),new GreaterThan()).data().asInt();
        yv.put(new INDArrayIndex[]{new SpecifiedIndex(indices)},1.0);
        indices = BooleanIndexing.chooseFrom(new INDArray[]{xv}, Arrays.asList(minDistance),new ArrayList<>(),new GreaterThanOrEqual()).data().asInt();
        INDArray distAssign = Transforms.exp(
                xv.get(new SpecifiedIndex(indices)).sub(minDistance).divi(spread).negi());
        yv.put(new INDArrayIndex[]{new SpecifiedIndex(indices)},distAssign);
        LevenbergMarquardtOptimizer levenbergMarquardtOptimizer = new LevenbergMarquardtOptimizer();
        double[] params = xv.data().asDouble();
        double[] target = yv.data().asDouble();
        final FiniteDifferencesDifferentiator finiteDifferencesDifferentiator = new FiniteDifferencesDifferentiator(params.length,1e-3);
        LeastSquaresOptimizer.Optimum optimum = levenbergMarquardtOptimizer.optimize(
                new LeastSquaresBuilder()
                        .start(params)
                        .target(params).maxEvaluations(10).maxIterations(100)
                        .model(new MultivariateJacobianFunction() {
                            @Override
                            public Pair<RealVector, RealMatrix> value(RealVector point) {
                                INDArray point2 = Nd4j.create(point.toArray());
                                //        return 1.0 / (1.0 + a * x ** (2 * b))
                                INDArray xPow = Transforms.pow(point2,2 * minDistance).muli(spread).addi(1.0).rdivi(1.0);
                                /**
                                 * https://github.com/apache/commons-math/blob/master/src/main/java/org/apache/commons/math4/analysis/interpolation/BicubicInterpolator.java
                                 */
                                return new Pair<>(new ArrayRealVector(xPow.data().asDouble()),null);
                            }
                        })
                        .build());
        return new INDArray[] {Nd4j.create(params),Nd4j.create(optimum.getCovariances(0.0).getData())};


    }


}
