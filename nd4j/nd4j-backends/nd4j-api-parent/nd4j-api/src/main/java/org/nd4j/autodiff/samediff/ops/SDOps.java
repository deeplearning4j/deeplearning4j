/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff.ops;

import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Abstract class for defining categories of operations - such as {@link SDMath} that is available via {@code SameDiff.math()}
 *
 * @author Alex Black
 */
public abstract class SDOps {

    protected final SameDiff sd;

    public SDOps(SameDiff sameDiff) {
        this.sd = sameDiff;
    }

    protected DifferentialFunctionFactory f() {
        return sd.f();
    }

    protected SDVariable updateVariableNameAndReference(SDVariable varToUpdate, String newVarName) {
        return sd.updateVariableNameAndReference(varToUpdate, newVarName);
    }
}
