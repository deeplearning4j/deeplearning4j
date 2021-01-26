/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.util;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This is a wrapper for PrintVariable op that just prints out Variable to the stdout
 *
 * @author raver119@gmail.com
 */
public class PrintVariable extends DynamicCustomOp {

    public PrintVariable() {
        //
    }

    public PrintVariable(INDArray array, boolean printSpecial) {
        inputArguments.add(array);
        bArguments.add(printSpecial);
    }

    public PrintVariable(INDArray array) {
        this(array, false);
    }

    public PrintVariable(INDArray array, String message, boolean printSpecial) {
        this(array, Nd4j.create(message), printSpecial);
    }

    public PrintVariable(INDArray array, String message) {
        this(array, Nd4j.create(message), false);
    }

    public PrintVariable(INDArray array, INDArray message, boolean printSpecial) {
        this(array, printSpecial);
        Preconditions.checkArgument(message.isS(), "Message argument should have String data type, but got [" + message.dataType() +"] instead");
        inputArguments.add(message);
    }

    public PrintVariable(INDArray array, INDArray message) {
        this(array, message, false);
    }

    @Override
    public String opName() {
        return "print_variable";
    }
}
