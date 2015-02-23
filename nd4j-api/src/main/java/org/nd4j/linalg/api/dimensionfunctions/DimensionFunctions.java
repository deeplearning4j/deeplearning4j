/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.dimensionfunctions;

import com.google.common.base.Function;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Dimension wise functions
 *
 * @author Adam Gibson
 */
public class DimensionFunctions {
    public static Function<INDArray, INDArray> normmax(final int dimension) {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                return input.normmax(dimension);
            }
        };
    }

    public static Function<INDArray, INDArray> norm2(final int dimension) {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                return input.norm2(dimension);
            }
        };
    }

    public static Function<INDArray, INDArray> norm1(final int dimension) {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                return input.norm1(dimension);
            }
        };
    }


    public static Function<INDArray, INDArray> sum(final int dimension) {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {

                return input.sum(dimension);
            }
        };
    }

    public static Function<INDArray, INDArray> var(final int dimension) {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {

                return input.var(dimension);
            }
        };
    }

    public static Function<INDArray, INDArray> std(final int dimension) {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {

                return input.std(dimension);
            }
        };
    }

    public static Function<INDArray, INDArray> prod(final int dimension) {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                return input.prod(dimension);
            }
        };
    }

    public static Function<INDArray, INDArray> cumsum(final int dimension) {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                return input.cumsum(dimension);
            }
        };
    }


    public static Function<INDArray, INDArray> mean(final int dimension) {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                return input.mean(dimension);
            }
        };
    }

    public static Function<INDArray, INDArray> min(final int dimension) {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                return input.min(dimension);
            }
        };
    }

    public static Function<INDArray, INDArray> max(final int dimension) {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                return input.max(dimension);
            }
        };
    }


    public static Function<INDArray, INDArray> norm2() {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                Accumulation acc = Nd4j.getExecutioner().execAndReturn(new Norm2(input));
                return input instanceof IComplexNDArray ? Nd4j.scalar(acc.currentResultComplex()) : Nd4j.scalar(acc.currentResult().doubleValue());
            }
        };
    }

    public static Function<INDArray, INDArray> norm1() {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                Accumulation acc = Nd4j.getExecutioner().execAndReturn(new Norm1(input));
                return input instanceof IComplexNDArray ? Nd4j.scalar(acc.currentResultComplex()) : Nd4j.scalar(acc.currentResult().doubleValue());
            }
        };
    }


    public static Function<INDArray, INDArray> sum() {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                Accumulation acc = Nd4j.getExecutioner().execAndReturn(new Sum(input));
                return input instanceof IComplexNDArray ? Nd4j.scalar(acc.currentResultComplex()) : Nd4j.scalar(acc.currentResult().doubleValue());
            }
        };
    }

    public static Function<INDArray, INDArray> var() {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                Accumulation acc = Nd4j.getExecutioner().execAndReturn(new Variance(input));
                return input instanceof IComplexNDArray ? Nd4j.scalar(acc.currentResultComplex()) : Nd4j.scalar(acc.currentResult().doubleValue());
            }
        };
    }

    public static Function<INDArray, INDArray> std() {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                Accumulation acc = Nd4j.getExecutioner().execAndReturn(new StandardDeviation(input));
                return input instanceof IComplexNDArray ? Nd4j.scalar(acc.currentResultComplex()) : Nd4j.scalar(acc.currentResult().doubleValue());
            }
        };
    }

    public static Function<INDArray, INDArray> prod() {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                Accumulation acc = Nd4j.getExecutioner().execAndReturn(new Prod(input));
                return input instanceof IComplexNDArray ? Nd4j.scalar(acc.currentResultComplex()) : Nd4j.scalar(acc.currentResult().doubleValue());
            }
        };
    }

    public static Function<INDArray, INDArray> cumsum() {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                double s = 0.0;
                for (int i = 0; i < input.length(); i++) {
                    if (input.data().dataType() == (DataBuffer.FLOAT))
                        s += input.getDouble(i);
                    else
                        s += input.getDouble(i);
                    input.putScalar(i, s);
                }

                return input;
            }
        };
    }


    public static Function<INDArray, INDArray> mean() {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                Accumulation acc = Nd4j.getExecutioner().execAndReturn(new Mean(input));
                return input instanceof IComplexNDArray ? Nd4j.scalar(acc.currentResultComplex()) : Nd4j.scalar(acc.currentResult().doubleValue());
            }
        };
    }

    public static Function<INDArray, INDArray> min() {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                Accumulation acc = Nd4j.getExecutioner().execAndReturn(new Min(input));
                return input instanceof IComplexNDArray ? Nd4j.scalar(acc.currentResultComplex()) : Nd4j.scalar(acc.currentResult().doubleValue());
            }
        };
    }

    public static Function<INDArray, INDArray> max() {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                Accumulation acc = Nd4j.getExecutioner().execAndReturn(new Max(input));
                return input instanceof IComplexNDArray ? Nd4j.scalar(acc.currentResultComplex()) : Nd4j.scalar(acc.currentResult().doubleValue());
            }
        };
    }

    public static Function<INDArray, INDArray> normmax() {
        return new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray input) {
                Accumulation acc = Nd4j.getExecutioner().execAndReturn(new NormMax(input));
                return input instanceof IComplexNDArray ? Nd4j.scalar(acc.currentResultComplex()) : Nd4j.scalar(acc.currentResult().doubleValue());
            }
        };
    }


}