/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Ones;
import org.nd4j.linalg.util.LinAlgExceptions;

/**
 * A base op for basic getters and setters
 *
 * @author Adam Gibson
 */
public abstract class BaseTransformOp extends BaseOp implements TransformOp {
    public BaseTransformOp(INDArray x, INDArray z) {
        super(x, z);
        LinAlgExceptions.assertSameLength(x,z);
    }

    public BaseTransformOp() {
    }

    public BaseTransformOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public BaseTransformOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        if(y != null)
            LinAlgExceptions.assertSameLength(x,y);
        LinAlgExceptions.assertSameLength(x,z);

    }

    public BaseTransformOp(INDArray x) {
        super(x);
    }

    @Override
    public TransformOp derivative() {
        return new Ones(x, y, z, n);
    }
}
