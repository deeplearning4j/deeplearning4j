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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndReplace;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A legacy op node is stored as its op type and number plus its op name. replaceWhere with an array
 * (CompareAndReplace) and replaceWhere with a value (the one-input CompareAndSet) are both TRANSFORM_SAME 13,
 * and a copy once rebuilt every CompareAndReplace as a CompareAndSet. The gradient graph works on a copy, so
 * the replacement array got the gradient of the array it replaces into.
 */
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class LegacyOpSerdeIdentityTest extends BaseNd4jTestWithBackends {

    private static INDArray values() {
        return Nd4j.createFromArray(0.1, 0.7, 0.3, 0.9, 0.5, 0.2).reshape(2, 3);
    }

    private static INDArray replacements() {
        return Nd4j.createFromArray(10.0, 20.0, 30.0, 40.0, 50.0, 60.0).reshape(2, 3);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void replaceWhereOpsKeepTheirClassThroughACopy(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", values());
        SDVariable replacement = sd.var("replacement", replacements());
        sd.replaceWhere("byArray", in, replacement, Conditions.lessThan(0.5));
        sd.replaceWhere("byValue", in, -1.0, Conditions.greaterThan(0.5));

        SameDiff copy = sd.dup();
        assertEquals(CompareAndReplace.class, copy.getVariableOutputOp("byArray").getClass());
        assertEquals(CompareAndSet.class, copy.getVariableOutputOp("byValue").getClass());

        // 0.5 matches neither condition
        Map<String, INDArray> out = copy.output(Collections.emptyMap(), "byArray", "byValue");
        assertEquals(Nd4j.createFromArray(10.0, 0.7, 30.0, 0.9, 0.5, 60.0).reshape(2, 3), out.get("byArray"));
        assertEquals(Nd4j.createFromArray(0.1, -1.0, 0.3, -1.0, 0.5, 0.2).reshape(2, 3), out.get("byValue"));
    }

    /**
     * byArray = in < 0.5 ? replacement : in, and byValue = in > 0.5 ? -1 : in. The loss weighs every element,
     * so each input's gradient is its weight wherever the output is taken from it, and 0 elsewhere.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void replaceWhereGradientsGoToTheInputEachElementCameFrom(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", values());
        SDVariable replacement = sd.var("replacement", replacements());
        SDVariable byArray = sd.replaceWhere("byArray", in, replacement, Conditions.lessThan(0.5));
        SDVariable byValue = sd.replaceWhere("byValue", in, -1.0, Conditions.greaterThan(0.5));
        INDArray arrayWeights = Nd4j.createFromArray(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).reshape(2, 3);
        INDArray valueWeights = Nd4j.createFromArray(0.5, 0.25, 0.125, 2.0, 4.0, 8.0).reshape(2, 3);
        byArray.mul(sd.constant("arrayWeights", arrayWeights)).sum()
                .add(byValue.mul(sd.constant("valueWeights", valueWeights)).sum())
                .markAsLoss();

        Map<String, INDArray> grads = sd.calculateGradients(Collections.emptyMap(), "in", "replacement");

        INDArray lessThanHalf = Nd4j.createFromArray(1.0, 0.0, 1.0, 0.0, 0.0, 1.0).reshape(2, 3);
        INDArray greaterThanHalf = Nd4j.createFromArray(0.0, 1.0, 0.0, 1.0, 0.0, 0.0).reshape(2, 3);
        INDArray expectedIn = arrayWeights.mul(lessThanHalf.rsub(1.0)).add(valueWeights.mul(greaterThanHalf.rsub(1.0)));
        assertEquals(expectedIn, grads.get("in"));
        assertEquals(arrayWeights.mul(lessThanHalf), grads.get("replacement"));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
