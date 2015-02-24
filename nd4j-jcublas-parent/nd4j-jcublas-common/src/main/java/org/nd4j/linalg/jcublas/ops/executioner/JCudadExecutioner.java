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

package org.nd4j.linalg.jcublas.ops.executioner;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;

/**
 * JCuda executioner.
 *
 * Runs ops directly on the gpu
 *
 * @author Adam Gibson
 */
public class JCudadExecutioner implements OpExecutioner {
    @Override
    public Op exec(Op op) {
        return null;
    }

    @Override
    public Op exec(Op op, Object[] extraArgs) {
        return null;
    }

    @Override
    public INDArray execAndReturn(TransformOp op) {
        return null;
    }

    @Override
    public INDArray execAndReturn(TransformOp op, Object[] extraArgs) {
        return null;
    }

    @Override
    public Accumulation execAndReturn(Accumulation op, Object[] extraArgs) {
        return null;
    }

    @Override
    public Accumulation execAndReturn(Accumulation op) {
        return null;
    }

    @Override
    public Op exec(Op op, int dimension) {
        return null;
    }

    @Override
    public Op exec(Op op, Object[] extraArgs, int dimension) {
        return null;
    }

    @Override
    public INDArray execAndReturn(TransformOp op, int dimension) {
        return null;
    }

    @Override
    public INDArray execAndReturn(TransformOp op, int dimension, Object[] extraArgs) {
        return null;
    }

    @Override
    public Accumulation execAndReturn(Accumulation op, int dimension, Object[] extraArgs) {
        return null;
    }

    @Override
    public Accumulation execAndReturn(Accumulation op, int dimension) {
        return null;
    }
}
