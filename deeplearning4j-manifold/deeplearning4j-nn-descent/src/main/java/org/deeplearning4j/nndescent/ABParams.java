package org.deeplearning4j.nndescent;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.apache.commons.math3.analysis.UnivariateVectorFunction;
import org.apache.commons.math3.analysis.differentiation.FiniteDifferencesDifferentiator;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.fitting.leastsquares.MultivariateJacobianFunction;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.nd4j.finitedifferences.TwoPointApproximation;
import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.indexing.conditions.GreaterThan;
import org.nd4j.linalg.indexing.conditions.GreaterThanOrEqual;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.max;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.bytedeco.javacpp.cminpack.*;
@Builder
public class ABParams {
    private double spread;
    private double minDistance;

    public INDArray[] solve() {
        INDArray xv = Nd4j.linspace(0,spread * 3,300);
        INDArray yv = Nd4j.zeros(xv.shape());
        INDArray xvLtMinDist = xv.lt(minDistance);

        yv = yv.putWhereWithMask(xvLtMinDist,1.0);
        INDArray xvGteMinDist = xv.getWhere(minDistance,new GreaterThanOrEqual());
        INDArray xvGteMinDistMinusMinDist = xvGteMinDist.sub(minDistance);
        INDArray neg = xvGteMinDistMinusMinDist.neg();
        INDArray divSpread = neg.div(spread);

        INDArray toPut = exp(divSpread);
        INDArray xvGteMask = xv.gte(minDistance);
        yv = yv.putWhereWithMask(xvGteMask,toPut);



        Function<INDArray,INDArray> f = new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray indArray) {
                INDArray add =  pow(indArray.mul(spread),2 * minDistance);
                INDArray arr =  add
                        .addi(1).rdivi(1.0);
                return arr;
            }
        };

        final  INDArray yvRef = yv;

       // LevenbergMarquardtOptimizer levenbergMarquardtOptimizer = new LevenbergMarquardtOptimizer();
        double[] params = xv.data().asDouble();
        double[] start =  {1,1};

        Func func = new Func(xv,yv);
        DoublePointer doublePointer = new DoublePointer(start);
        double fTol = 1.49012e-08;
        double xTol = 1.49012e-08;
        double gTol = 0.0;
        int maxFev = 600;
        double epsfcn = 2.220446049250313e-16;
        double factor = 100;
        INDArray fVec = Nd4j.create(xv.shape());
        DoublePointer fVecPointer = (DoublePointer) fVec.data().pointer();
        IntPointer ipvt = new IntPointer(new int[start.length]);
        DataBuffer ipvtBuff = Nd4j.createBuffer(new int[]{start.length}, DataBuffer.Type.INT);
        INDArray ipvtArr = Nd4j.create(ipvtBuff);
        IntPointer nFev = new IntPointer(new int[1]);
        INDArray fjac = Nd4j.create(new int[] {start.length,start.length});
        DoublePointer fJacPointer = (DoublePointer) fjac.data().pointer();
        lmdif(func,doublePointer,
                xv.length(),
                params.length,
                doublePointer,
                fVecPointer,
                fTol,
                xTol,
                gTol,
                maxFev,
                epsfcn,
                null,
                1,
                factor,
                0,
                nFev,
                fJacPointer,
                0,
                ipvt,
                new DoublePointer(),
                new DoublePointer(),
                new DoublePointer(),
                new DoublePointer(),
                new DoublePointer());
   /*     final FiniteDifferencesDifferentiator finiteDifferencesDifferentiator = new FiniteDifferencesDifferentiator(params.length,1e-3);
        LeastSquaresOptimizer.Optimum optimum = levenbergMarquardtOptimizer.optimize(
                new LeastSquaresBuilder()
                        .start(start)
                        .target(params).maxEvaluations(100).maxIterations(100)
                        .model(new MultivariateJacobianFunction() {
                            @Override
                            public Pair<RealVector, RealMatrix> value(RealVector point) {
                                INDArray point2 = Nd4j.create(point.toArray());
                                //        return 1.0 / (1.0 + a * x ** (2 * b))
                                INDArray xPow = Transforms.pow(point2,2 * minDistance).muli(spread).addi(1.0).rdivi(1.0);
                                INDArray derivative = TwoPointApproximation.approximateDerivative(f,point2,null,yvRef,
                                        Nd4j.create(new double[] {Float.MIN_VALUE
                                                ,Float.MAX_VALUE}));


                                *//**
                                 * https://github.com/apache/commons-math/blob/master/src/main/java/org/apache/commons/math4/analysis/interpolation/BicubicInterpolator.java
                                 *//*
                                return new Pair<>(new ArrayRealVector(xPow.data().asDouble()),new Array2DRowRealMatrix(derivative.toDoubleMatrix()));
                            }
                        })
                        .build());*/

        /**
         *  perm = take(eye(n), ipvt - 1, 0)
         r = triu(transpose(fjac)[:n, :])
         R = dot(r, perm)
         try:
         cov_x = inv(dot(transpose(R), R))
         except (LinAlgError, ValueError):
         pass
         return (retval[0], cov_x) + retval[1:-1] + (mesg, info)

         */

        /**
         * Note that retval above is already pointers we have access to
         */



        return null;
       // return new INDArray[] {Nd4j.create(params),Nd4j.create(optimum.getCovariances(0.0).getData())};


    }


    @AllArgsConstructor
    public static class Func extends cminpack_func_mn {
        private INDArray xData,yData;
        @Override
        public int call(Pointer p, int m, int n, DoublePointer x, DoublePointer fvec, int iflag) {
            int i;
            DoubleIndexer xIdx = DoubleIndexer.create(x.capacity(n));
            double a = xIdx.get(0);
            double b = xIdx.get(1);
            INDArray add =  pow(xData.mul(a),2 * b);
            INDArray arr =  add
                    .addi(1).rdivi(1.0);
            INDArray diff = arr.sub(yData);

            DoubleIndexer fvecIdx = DoubleIndexer.create(fvec.capacity(m));
            for (i = 0; i < diff.length(); ++i) {
                fvecIdx.put(i, diff.getDouble(i));
            }



            return 0;
        }
    }



}
