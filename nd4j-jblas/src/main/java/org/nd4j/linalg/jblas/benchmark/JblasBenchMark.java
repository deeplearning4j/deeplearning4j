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

package org.nd4j.linalg.jblas.benchmark;

import org.jblas.DoubleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.benchmark.TimeOperations;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 8/20/14.
 */
public class JblasBenchMark {


    public static void main(String[] args) {
        INDArray n = Nd4j.linspace(1, 100000, 100000).reshape(50000, 2);
        TimeOperations ops = new TimeOperations(n, 1000);
        ops.run();

        DoubleMatrix linspace = DoubleMatrix.linspace(1, 100000, 100000).reshape(50000, 2);
        DoubleMatrix linspace2 = DoubleMatrix.linspace(1, 100000, 100000).reshape(50000, 2);

        long timeDiff = 0;

        for (int i = 0; i < 1000; i++) {
            long before = System.currentTimeMillis();
            linspace.mul(linspace2);
            long after = System.currentTimeMillis();
            timeDiff += Math.abs(after - before);
        }

        System.out.println("Took on avg " + (timeDiff / 1000) + " milliseconds");


    }


}
