/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Abstract base class for {@link Module}
 * that handles Dynamic ops and handles nesting.
 *
 * This is a logical unit for defining layers
 * very similar to pytorch's modules, or tensorflow's layers.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public abstract class BaseModule extends DynamicCustomOp implements Module {
    private List<Module> modules = new ArrayList<>();

    public BaseModule(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, List<Module> modules) {
        super(opName, inputs, outputs, tArguments, iArguments);
        this.modules = modules;
    }

    public BaseModule(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, List<Module> modules) {
        super(opName, sameDiff, args, inPlace);
        this.modules = modules;
    }

    @Override
    public Module[] subModules() {
        return modules.toArray(new Module[modules.size()]);
    }

    @Override
    public void addModule(Module module) {
        modules.add(module);
    }




}
