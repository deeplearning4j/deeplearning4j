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
package org.eclipse.deeplearning4j.dl4jcore.nn.layers.convolution;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.layers.SpaceToDepthLayer;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.DepthToSpace;
import org.nd4j.linalg.api.ops.impl.layers.convolution.SpaceToDepth;
import org.nd4j.enums.DataFormat;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.DisplayName;

@DisplayName("Space To Depth Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
class SpaceToDepthTest extends BaseDL4JTest {

    private int mb = 1;

    private int inDepth = 2;

    private int inputWidth = 2;

    private int inputHeight = 2;

    private int blockSize = 2;

    private SpaceToDepthLayer.DataFormat dataFormat = SpaceToDepthLayer.DataFormat.NCHW;

    private int outDepth = inDepth * blockSize * blockSize;

    private int outputHeight = inputHeight / blockSize;

    private int outputWidth = inputWidth / blockSize;

    private INDArray getContainedData() {
        return Nd4j.create(new double[] { 1., 2., 3., 4., 5., 6., 7., 8. }, new int[] { mb, inDepth, inputHeight, inputWidth }, 'c');
    }

    private INDArray getContainedOutput() {
        return Nd4j.create(new double[] { 1., 5., 2., 6., 3., 7., 4., 8. }, new int[] { mb, outDepth, outputHeight, outputWidth }, 'c');
    }

    private Layer getSpaceToDepthLayer() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).seed(123).layer(new SpaceToDepthLayer.Builder(blockSize, dataFormat).build()).build();
        return conf.getLayer().instantiate(conf, null, 0, null, true, Nd4j.defaultFloatingPointType());
    }

    @Test
    @DisplayName("Test Space To Depth Forward")
    void testSpaceToDepthForward() throws Exception {
        INDArray containedInput = getContainedData();
        INDArray containedExpectedOut = getContainedOutput();
        Layer std = getSpaceToDepthLayer();
        INDArray containedOutput = std.activate(containedInput, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(containedExpectedOut.shape(), containedOutput.shape()));
        assertEquals(containedExpectedOut, containedOutput);
    }

    @Test
    @DisplayName("Test Space To Depth Backward")
    void testSpaceToDepthBackward() throws Exception {
        INDArray containedInputEpsilon = getContainedOutput();
        INDArray containedExpectedOut = getContainedData();
        Layer std = getSpaceToDepthLayer();
        std.setInput(getContainedData(), LayerWorkspaceMgr.noWorkspaces());
        INDArray containedOutput = std.backpropGradient(containedInputEpsilon, LayerWorkspaceMgr.noWorkspaces()).getRight();
        assertTrue(Arrays.equals(containedExpectedOut.shape(), containedOutput.shape()));
        assertEquals(containedExpectedOut, containedOutput);
    }

    @Test
    @DisplayName("Space To Depth and inverse respect F-order strides")
    void testSpaceToDepthAndInverseWithFOrderArrays() {
        for (DataFormat format : new DataFormat[] {DataFormat.NCHW, DataFormat.NHWC}) {
            long[] inputShape = format == DataFormat.NCHW ? new long[] {2, 2, 4, 4} : new long[] {2, 4, 4, 2};
            long[] outputShape = format == DataFormat.NCHW ? new long[] {2, 8, 2, 2} : new long[] {2, 2, 2, 8};
            INDArray cInput = Nd4j.linspace(1, 64, 64, DataType.DOUBLE).reshape('c', inputShape);
            INDArray expected = Nd4j.createUninitialized(DataType.DOUBLE, outputShape, 'c');
            // Independent coordinate oracle: do not compare two invocations of the same helper.
            for (int n = 0; n < 2; n++)
                for (int h = 0; h < 4; h++)
                    for (int w = 0; w < 4; w++)
                        for (int d = 0; d < 2; d++) {
                            int od = ((h % 2) * 2 + w % 2) * 2 + d;
                            long[] source = format == DataFormat.NCHW ? new long[] {n, d, h, w} : new long[] {n, h, w, d};
                            long[] target = format == DataFormat.NCHW ? new long[] {n, od, h / 2, w / 2}
                                    : new long[] {n, h / 2, w / 2, od};
                            expected.putScalar(target, cInput.getDouble(source));
                        }

            INDArray fOutput = Nd4j.createUninitialized(DataType.DOUBLE, outputShape, 'f');
            Nd4j.getExecutioner().exec(new SpaceToDepth(cInput.dup('f'), fOutput, 2, format));
            assertEquals(expected, fOutput, format + " forward with F-order input and output");

            INDArray fRestored = Nd4j.createUninitialized(DataType.DOUBLE, inputShape, 'f');
            Nd4j.getExecutioner().exec(new DepthToSpace(fOutput, fRestored, 2, format));
            assertEquals(cInput, fRestored, format + " inverse with F-order input and output");
        }
    }

    private static long[] shapeOf(SpaceToDepth op) {
        var descriptor = op.calculateOutputShape().get(0);
        return new long[] {descriptor.getLong(1), descriptor.getLong(2), descriptor.getLong(3), descriptor.getLong(4)};
    }

    private static long[] shapeOf(DepthToSpace op) {
        var descriptor = op.calculateOutputShape().get(0);
        return new long[] {descriptor.getLong(1), descriptor.getLong(2), descriptor.getLong(3), descriptor.getLong(4)};
    }

    @Test
    void testShapeArithmeticBeyondIntWithoutAllocatingTensorData() {
        // A zero batch preserves all spatial/channel dimensions but allocates no huge tensor.
        for (DataFormat format : new DataFormat[] {DataFormat.NCHW, DataFormat.NHWC}) {
            boolean nchw = format == DataFormat.NCHW;
            long[] spatial = nchw ? new long[] {0, 1, 65536, 65536} : new long[] {0, 65536, 65536, 1};
            long[] packed = nchw ? new long[] {0, 4294967296L, 1, 1} : new long[] {0, 1, 1, 4294967296L};
            assertArrayEquals(packed, shapeOf(new SpaceToDepth(Nd4j.create(DataType.DOUBLE, spatial), 65536, format)));
            assertArrayEquals(spatial, shapeOf(new DepthToSpace(Nd4j.create(DataType.DOUBLE, packed), 65536, format)));
            long[] wide = nchw ? new long[] {0, 1, 2147483650L, 2} : new long[] {0, 2147483650L, 2, 1};
            long[] reduced = nchw ? new long[] {0, 4, 1073741825L, 1} : new long[] {0, 1073741825L, 1, 4};
            assertArrayEquals(reduced, shapeOf(new SpaceToDepth(Nd4j.create(DataType.DOUBLE, wide), 2, format)));
            assertArrayEquals(wide, shapeOf(new DepthToSpace(Nd4j.create(DataType.DOUBLE, reduced), 2, format)));
            long[] channelOverflow = nchw ? new long[] {0, Long.MAX_VALUE / 4 + 1, 2, 0}
                    : new long[] {0, 2, 0, Long.MAX_VALUE / 4 + 1};
            assertThrows(RuntimeException.class, () -> new SpaceToDepth(
                    Nd4j.create(DataType.DOUBLE, channelOverflow), 2, format).calculateOutputShape());
            long[] heightOverflow = nchw ? new long[] {0, 4, Long.MAX_VALUE / 2 + 1, 0}
                    : new long[] {0, Long.MAX_VALUE / 2 + 1, 0, 4};
            assertThrows(RuntimeException.class, () -> new DepthToSpace(
                    Nd4j.create(DataType.DOUBLE, heightOverflow), 2, format).calculateOutputShape());
        }
    }

    @Test
    void testInvalidRankBlockAndDivisibilityAreRejectedDuringShapeInference() {
        for (DataFormat format : new DataFormat[] {DataFormat.NCHW, DataFormat.NHWC}) {
            INDArray scalar = Nd4j.scalar(1.0);
            assertThrows(RuntimeException.class, () -> new SpaceToDepth(scalar, 2, format).calculateOutputShape());
            assertThrows(RuntimeException.class, () -> new DepthToSpace(scalar, 2, format).calculateOutputShape());
            INDArray small = Nd4j.ones(DataType.DOUBLE, 1, 1, 1, 1);
            for (int block : new int[] {0, -1, 2, 65536}) {
                assertThrows(RuntimeException.class, () -> new SpaceToDepth(small, block, format).calculateOutputShape());
                assertThrows(RuntimeException.class, () -> new DepthToSpace(small, block, format).calculateOutputShape());
            }
            // The old narrowing to int turned this invalid argument into block 1.
            SpaceToDepth forward = new SpaceToDepth();
            forward.addInputArgument(small);
            forward.addIArgument(4294967297L, format == DataFormat.NHWC ? 1 : 0);
            assertThrows(RuntimeException.class, forward::calculateOutputShape);
            DepthToSpace inverse = new DepthToSpace();
            inverse.addInputArgument(small);
            inverse.addIArgument(4294967297L, format == DataFormat.NHWC ? 1 : 0);
            assertThrows(RuntimeException.class, inverse::calculateOutputShape);
        }
    }
}
