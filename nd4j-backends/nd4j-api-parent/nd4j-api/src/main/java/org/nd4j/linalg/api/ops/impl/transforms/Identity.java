/*-
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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

/**
 * Identity function
 *
 * @author Adam Gibson
 */
public class Identity extends BaseDynamicTransformOp {


    @Override
    public String opName() {
        return "identity";
    }

    @Override
    public String onnxName() {
        return "Constant";
    }

    @Override
    public String tensorflowName() {
        return "Identity";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"Identity", "NoOp"};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {

        return Arrays.asList(sameDiff.one("grad-" + UUID.randomUUID().toString(), i_v.get(0).getShape()));
    }

}
