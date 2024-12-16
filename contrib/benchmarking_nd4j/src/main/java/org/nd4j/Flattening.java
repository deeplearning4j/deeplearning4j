
/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class Flattening {

    @State(Scope.Thread)
    public static class SetupState {
        public INDArray small_c = org.nd4j.linalg.factory.Nd4j.create(new int[]{1<<10, 1<<10}, 'c');
        public INDArray small_f = org.nd4j.linalg.factory.Nd4j.create(new int[]{1<<10, 1<<10}, 'f');
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void toFlattened_CC_Small(SetupState state) throws IOException {
        org.nd4j.linalg.factory.Nd4j.toFlattened('c', state.small_c);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void toFlattened_CF_Small(SetupState state) throws IOException {
        org.nd4j.linalg.factory.Nd4j.toFlattened('f', state.small_c);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void toFlattened_FF_Small(SetupState state) throws IOException {
        org.nd4j.linalg.factory.Nd4j.toFlattened('f', state.small_f);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void toFlattened_FC_Small(SetupState state) throws IOException {
        org.nd4j.linalg.factory.Nd4j.toFlattened('c', state.small_f);
    }

}
