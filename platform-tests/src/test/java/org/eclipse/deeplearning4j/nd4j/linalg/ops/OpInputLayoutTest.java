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
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.UnaryOperator;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Ops that reshape, flatten or contract their operands give the same results whatever the operands' layouts: C order,
 * F order, an offset view, a stepped view and a permuted view, each for every operand at once and for the first
 * operand alone (so operands of different orders meet). A logical reshape keeps an array's elements in C order
 * whatever its memory order; reshaping in the memory order instead pairs F-ordered elements differently, and an
 * index guard that compares a stride-derived offset with the length skips a view's elements.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class OpInputLayoutTest extends BaseNd4jTestWithBackends {

    private static final class Layout {
        final String name;
        final UnaryOperator<INDArray> of;

        Layout(String name, UnaryOperator<INDArray> of) {
            this.name = name;
            this.of = of;
        }
    }

    private static final Layout[] LAYOUTS = {
            new Layout("F order", x -> x.dup('f')),
            new Layout("offset view", x -> {
                long[] s = x.shape().clone();
                s[0] += 1;
                INDArray parent = Nd4j.valueArrayOf(s, Double.NaN, x.dataType());
                INDArrayIndex[] idx = new INDArrayIndex[s.length];
                idx[0] = NDArrayIndex.interval(1, s[0]);
                for (int i = 1; i < s.length; i++)
                    idx[i] = NDArrayIndex.all();
                return parent.get(idx).assign(x);
            }),
            new Layout("stepped view", x -> {
                long[] s = x.shape().clone();
                int last = s.length - 1;
                long n = s[last];
                s[last] = 2 * n + 1;
                INDArray parent = Nd4j.valueArrayOf(s, Double.NaN, x.dataType());
                INDArrayIndex[] idx = new INDArrayIndex[s.length];
                for (int i = 0; i < last; i++)
                    idx[i] = NDArrayIndex.all();
                idx[last] = NDArrayIndex.interval(1, 2, 2 * n + 1);
                return parent.get(idx).assign(x);
            }),
            new Layout("permuted view", x -> {
                long[] s = x.shape();
                int r = s.length;
                long[] reversed = new long[r];
                long[] perm = new long[r];
                for (int i = 0; i < r; i++) {
                    reversed[i] = s[r - 1 - i];
                    perm[i] = r - 1 - i;
                }
                return Nd4j.create(x.dataType(), reversed, 'c').permute(perm).assign(x);
            }),
    };

    private static final class Case {
        final String name;
        final INDArray[] inputs;
        final Function<INDArray[], INDArray[]> op;

        Case(String name, Function<INDArray[], INDArray[]> op, INDArray... inputs) {
            this.name = name;
            this.op = op;
            this.inputs = inputs;
        }
    }

    private static INDArray rand(long... shape) {
        return Nd4j.rand(DataType.DOUBLE, shape).subi(0.5);
    }

    private static INDArray[] exec(String opName, INDArray[] in, long... iArgs) {
        DynamicCustomOp op = DynamicCustomOp.builder(opName).addInputs(in).addIntegerArguments(iArgs).build();
        return Nd4j.exec(op);
    }

    private static INDArray[] one(INDArray x) {
        return new INDArray[]{x};
    }

    private static List<Case> cases() {
        Nd4j.getRandom().setSeed(42);
        List<Case> cases = new ArrayList<>();
        cases.add(new Case("matmul rank 3", in -> one(Nd4j.linalg().matmul(in[0], in[1])), rand(2, 3, 4), rand(2, 4, 5)));
        cases.add(new Case("matmul rank 4", in -> one(Nd4j.linalg().matmul(in[0], in[1])), rand(2, 2, 3, 4), rand(2, 2, 4, 5)));
        cases.add(new Case("matmul rank 3 transposed", in -> one(Nd4j.linalg().matmul(in[0], in[1], 1.0, 0.0, true, true)),
                rand(2, 4, 3), rand(2, 5, 4)));
        cases.add(new Case("tensormmul", in -> one(Nd4j.base().tensorMmul(in[0], in[1], new int[]{1, 2}, new int[]{1, 0})),
                rand(2, 3, 4), rand(4, 3, 5)));
        cases.add(new Case("cross", in -> one(Nd4j.linalg().cross(in[0], in[1])), rand(4, 3), rand(4, 3)));
        cases.add(new Case("cross rank 3", in -> one(Nd4j.linalg().cross(in[0], in[1])), rand(2, 2, 3), rand(2, 2, 3)));
        cases.add(new Case("space_to_batch", in -> one(Nd4j.cnn().spaceToBatch(in[0], new int[]{2, 2}, new int[]{0, 0}, 0, 0)),
                rand(2, 4, 4, 3)));
        cases.add(new Case("batch_to_space", in -> one(Nd4j.cnn().batchToSpace(in[0], new int[]{2, 2}, new int[]{0, 0}, 0, 0)),
                rand(8, 2, 2, 3)));
        cases.add(new Case("flatten_2d", in -> exec("flatten_2d", in, 1), rand(2, 3, 4)));
        cases.add(new Case("reshape splitting an axis", in -> exec("reshape", in, -'c', 3, 3, 2, 4, 5), rand(3, 6, 4, 5)));
        cases.add(new Case("reshape merging axes", in -> exec("reshape", in, -'c', 3, 24, 5), rand(3, 6, 4, 5)));
        cases.add(new Case("deconv2d", in -> one(Nd4j.cnn().deconv2d(in[0], in[1], in[2], DeConv2DConfig.builder()
                .kH(3).kW(3).sH(2).sW(2).build())), rand(2, 3, 5, 5), rand(3, 3, 4, 3), rand(4)));
        cases.add(new Case("conv1d", in -> one(Nd4j.cnn().conv1d(in[0], in[1], in[2], Conv1DConfig.builder()
                .k(3).s(1).paddingMode(PaddingMode.SAME).dataFormat(Conv1DConfig.NCW).build())),
                rand(2, 3, 7), rand(3, 3, 4), rand(4)));
        cases.add(new Case("conv3d", in -> one(Nd4j.cnn().conv3d(in[0], in[1], in[2], Conv3DConfig.builder()
                .kD(2).kH(2).kW(2).biasUsed(true).paddingMode(PaddingMode.VALID).dataFormat(Conv3DConfig.NCDHW).build())),
                rand(2, 2, 4, 4, 4), rand(2, 2, 2, 2, 3), rand(3)));
        cases.add(new Case("avgpool2d", in -> one(Nd4j.cnn().avgPooling2d(in[0], Pooling2DConfig.builder()
                .kH(2).kW(2).sH(2).sW(2).build())), rand(2, 3, 6, 6)));
        cases.add(new Case("maxpool2d", in -> one(Nd4j.cnn().maxPooling2d(in[0], Pooling2DConfig.builder()
                .kH(3).kW(3).sH(1).sW(1).paddingMode(PaddingMode.SAME).build())), rand(2, 3, 6, 6)));
        cases.add(new Case("upsampling2d", in -> one(Nd4j.cnn().upsampling2d(in[0], 2)), rand(2, 3, 3, 3)));
        cases.add(new Case("softmax axis 1", in -> exec("softmax", in, 1), rand(2, 5, 3)));
        cases.add(new Case("reduce_sum axes 1, 2", in -> one(in[0].sum(true, 1, 2)), rand(2, 3, 4)));
        cases.add(new Case("concat axis 1", in -> one(Nd4j.concat(1, in[0], in[1])), rand(2, 3, 4), rand(2, 2, 4)));
        cases.add(new Case("betainc", in -> exec("betainc", in),
                rand(3, 4).addi(1.0), rand(3, 4).addi(1.0), rand(3, 4).addi(0.5)));
        cases.add(new Case("mirror pad rank 1", in -> exec("pad", in, 1), rand(6),
                Nd4j.createFromArray(new long[][]{{2, 2}})));
        cases.add(new Case("mirror pad rank 2", in -> exec("pad", in, 2), rand(3, 4),
                Nd4j.createFromArray(new long[][]{{1, 1}, {2, 2}})));
        return cases;
    }

    private static void compare(List<String> failures, String what, INDArray[] expected, INDArray[] actual) {
        for (int i = 0; i < expected.length; i++) {
            INDArray e = expected[i].castTo(DataType.DOUBLE), a = actual[i].castTo(DataType.DOUBLE);
            if (!Arrays.equals(e.shape(), a.shape())) {
                failures.add(what + ", output " + i + ": shape " + Arrays.toString(a.shape()) + " vs "
                        + Arrays.toString(e.shape()));
                continue;
            }
            double maxDiff = e.sub(a).amaxNumber().doubleValue();
            if (!(maxDiff <= 1e-12 * (1 + e.amaxNumber().doubleValue())))
                failures.add(what + ", output " + i + ": max |diff| " + maxDiff);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void resultsDoNotDependOnTheInputLayout(Nd4jBackend backend) {
        List<String> failures = new ArrayList<>();
        for (Case c : cases()) {
            INDArray[] cOrder = Arrays.stream(c.inputs).map(x -> x.dup('c')).toArray(INDArray[]::new);
            INDArray[] expected;
            try {
                expected = c.op.apply(cOrder);
                Nd4j.getExecutioner().commit();
            } catch (RuntimeException e) {
                failures.add(c.name + ", C-order operands: " + e);
                continue;
            }
            for (Layout layout : LAYOUTS) {
                INDArray[] allInLayout = Arrays.stream(c.inputs).map(layout.of).toArray(INDArray[]::new);
                try {
                    INDArray[] actual = c.op.apply(allInLayout);
                    Nd4j.getExecutioner().commit();
                    compare(failures, c.name + ", every operand " + layout.name, expected, actual);
                } catch (RuntimeException e) {
                    failures.add(c.name + ", every operand " + layout.name + ": " + e);
                }
                if (c.inputs.length > 1) {
                    INDArray[] firstInLayout = Arrays.stream(c.inputs).map(x -> x.dup('c')).toArray(INDArray[]::new);
                    firstInLayout[0] = layout.of.apply(c.inputs[0]);
                    try {
                        INDArray[] actual = c.op.apply(firstInLayout);
                        Nd4j.getExecutioner().commit();
                        compare(failures, c.name + ", first operand " + layout.name, expected, actual);
                    } catch (RuntimeException e) {
                        failures.add(c.name + ", first operand " + layout.name + ": " + e);
                    }
                }
            }
        }
        assertTrue(failures.isEmpty(), failures.size() + " failures:\n" + String.join("\n", failures));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void flattenIntoAnFOrderOutputKeepsCOrder(Nd4jBackend backend) {
        INDArray x = rand(2, 3, 4);
        INDArray expected = x.dup('c').reshape('c', 2, 12);
        INDArray z = Nd4j.create(DataType.DOUBLE, new long[]{2, 12}, 'f');
        Nd4j.exec(DynamicCustomOp.builder("flatten_2d").addInputs(x).addOutputs(z).addIntegerArguments(1).build());
        assertTrue(expected.equalsWithEps(z, 1e-12), "flatten_2d into an F-order output: " + z + " vs " + expected);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reshapeBetweenFOrderArraysKeepsCOrder(Nd4jBackend backend) {
        // An F-ordered array copied into an F-ordered array of another shape: a memory copy would pair the elements
        // in F order
        INDArray x = rand(3, 6, 4, 5).dup('f');
        INDArray expected = x.dup('c').reshape('c', 3, 3, 2, 4, 5);
        INDArray z = Nd4j.create(DataType.DOUBLE, new long[]{3, 3, 2, 4, 5}, 'f');
        Nd4j.exec(DynamicCustomOp.builder("reshape").addInputs(x).addOutputs(z)
                .addIntegerArguments(-'c', 3, 3, 2, 4, 5).build());
        assertTrue(expected.equalsWithEps(z, 1e-12), "reshape between F-ordered arrays: " + z + " vs " + expected);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
