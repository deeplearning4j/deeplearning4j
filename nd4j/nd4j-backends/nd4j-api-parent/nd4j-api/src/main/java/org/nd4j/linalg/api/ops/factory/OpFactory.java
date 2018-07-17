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

package org.nd4j.linalg.api.ops.factory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;

/**
 * Op factory
 *
 * @author Adam Gibson
 */
public interface OpFactory {

    GradientOp createGradientOp(String name, INDArray x, INDArray y, INDArray z);

    /**
     *
     *
     * @param name
     * @param x
     * @param z
     * @param extraArgs
     * @return
     */
    Op createShape(String name, INDArray x, INDArray z, Object[] extraArgs);

    /**
     * Create a loss function with the given inputs and outputs
     * @param name the opName of the function
     * @param x the input
     * @param y the output
     * @return a loss function representing the delta between the 2
     */
    LossFunction createLossFunction(String name, INDArray x, INDArray y);

    /**
     * Accumulation operation
     *
     * @param name the opName of the function to create
     * @param x    the input to the function
     * @return the operation
     */
    Accumulation createAccum(String name, INDArray x);

    /**
     * Accumulation operation
     *
     * @param name the opName of the function
     * @param x    the input
     * @param y    the pairwise transformation
     * @param z    the output
     * @return the operation
     */
    Accumulation createAccum(String name, INDArray x, INDArray y, INDArray z);

    /**
     *
     * @param name
     * @param x
     * @param y
     * @param z
     * @param extraArgs
     * @return
     */
    Accumulation createAccum(String name, INDArray x, INDArray y, INDArray z, Object[] extraArgs);

    /**
     * @param name
     * @param x
     * @param y
     * @return
     */
    Accumulation createAccum(String name, INDArray x, INDArray y);

    /**
     *
     * @param opName
     * @param x
     * @param y
     *@param z
     * @param extraArgs   @return
     */
    IndexAccumulation createIndexAccum(String opName, INDArray x, INDArray y, INDArray z, Object[] extraArgs);



    /**
     *
     * @param name
     * @param x
     * @return
     */
    IndexAccumulation createIndexAccum(String name, INDArray x);

    /**Index accumulation operation
     * @param name
     * @param x
     * @param y
     * @return
     */
    IndexAccumulation createIndexAccum(String name, INDArray x, INDArray y);

    /**
     * @param name
     * @param x
     * @param y
     * @return
     */
    TransformOp createTransform(String name, INDArray x, INDArray y);


    /**
     * @param name
     * @param x
     * @return
     */
    TransformOp createTransform(String name, INDArray x);

    /**
     * @param name
     * @param x
     * @param extraArgs
     * @return
     */
    TransformOp createTransform(String name, INDArray x, Object[] extraArgs);

    /**
     * @param name
     * @param x
     * @param y
     * @param z
     * @return
     */
    TransformOp createTransform(String name, INDArray x, INDArray y, INDArray z);


    /**
     * @param name
     * @param x
     * @param y
     * @param z
     * @return-m 'add more scalar ops' -a
     * git push or
     */
    TransformOp createTransform(String name, INDArray x, INDArray y, INDArray z, Object[] extraArgs);



    /**
     * @param name
     * @param x
     * @param y
     * @param scalar
     * @return
     */
    ScalarOp createScalarTransform(String name, INDArray x, INDArray y, double scalar);


    /**
     * @param name
     * @param x
     * @param scalar
     * @return
     */
    ScalarOp createScalarTransform(String name, INDArray x, double scalar);

    /**
     * @param name
     * @param x
     * @param extraArgs
     * @param scalar
     * @return
     */
    ScalarOp createScalarTransform(String name, INDArray x, Object[] extraArgs, double scalar);

    /**
     * @param name
     * @param x
     * @param y
     * @param z
     * @param scalar
     * @return
     */
    ScalarOp createScalarTransform(String name, INDArray x, INDArray y, INDArray z, double scalar);


    /**
     * @param name
     * @param x
     * @param y
     * @param z
     * @param scalar
     * @return
     */
    ScalarOp createScalarTransform(String name, INDArray x, INDArray y, INDArray z, Object[] extraArgs, double scalar);



    /** Create a vector operation
     *
     * @param name Name of the vector op
     * @param x NDArray to operate on
     * @param y Vector
     * @param z Result NDArray
     * @param dimension Dimension to do op along. 0 for row, 1 for column, etc
     * @return VectorOp
     */
    BroadcastOp createBroadcastOp(String name, INDArray x, INDArray y, INDArray z, int... dimension);

    /**
     *
     * @param name
     * @param x
     * @param y
     * @param z
     * @param extraArgs
     * @param dimension
     * @return
     */
    BroadcastOp createBroadcastOp(String name, INDArray x, INDArray y, INDArray z, Object[] extraArgs,
                    int... dimension);

    /** Create a vector operation
     *
     * @param name Name of the vector op
     * @param x NDArray to operate on
     * @param z Result NDArray
     * @param dimension Dimension to do op along. 0 for row, 1 for column, etc
     * @return VectorOp
     */
    BroadcastOp createBroadcastOp(String name, INDArray x, INDArray z, int... dimension);

    /**
     * This method returns op id number for given opName
     *
     * @return
     */
    int getOpNumByName(String opName);

    /**
     * This method returns op id number if opName exists, or -1 otherwise
     *
     * @param opName
     * @return
     */
    int getOpNumIfExists(String opName);


    /**
     * This method returns Op instance if opName exists, null otherwise
     * @param opName
     * @return
     */
    Op getOpByName(String opName);
}
