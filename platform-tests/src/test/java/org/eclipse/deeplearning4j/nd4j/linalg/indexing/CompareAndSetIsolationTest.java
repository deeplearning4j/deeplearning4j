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

package org.eclipse.deeplearning4j.nd4j.linalg.indexing;

import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.common.tests.tags.NativeTag;

import java.lang.reflect.Method;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

@NativeTag
public class CompareAndSetIsolationTest extends BaseNd4jTestWithBackends {

    @Test
    public void testCompareAndSetExtraArgsPointerInitializedForFloat() {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        CompareAndSet op = new CompareAndSet(Nd4j.create(new float[]{0.1f, 0.2f, 0.3f}), 1.0, Conditions.lessThan(0.2));

        DataBuffer extraArgs = op.extraArgsDataBuff(DataType.FLOAT);
        assertNotNull(extraArgs);
        assertNotNull(extraArgs.pointer(), "pointer() should be initialized");
        assertTrue(extraArgs.pointer().address() != 0L, "pointer() address should be initialized");
        assertNotNull(extraArgs.addressPointer(), "addressPointer() should be initialized");
        assertTrue(extraArgs.addressPointer().address() != 0L, "addressPointer() should be initialized");
    }

    @Test
    public void testCompareAndSetExtraArgsPointerInitializedForDouble() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        CompareAndSet op = new CompareAndSet(Nd4j.create(new double[]{1.0, 2.0, 0.0, 4.0, 5.0}), 3.0, Conditions.lessThan(2.0));

        DataBuffer extraArgs = op.extraArgsDataBuff(DataType.DOUBLE);
        assertNotNull(extraArgs);
        assertNotNull(extraArgs.pointer(), "pointer() should be initialized");
        assertTrue(extraArgs.pointer().address() != 0L, "pointer() address should be initialized");
        assertNotNull(extraArgs.addressPointer(), "addressPointer() should be initialized");
        assertTrue(extraArgs.addressPointer().address() != 0L, "addressPointer() should be initialized");
    }

    /**
     * Unwrap a DeviceAwareOpExecutioner to its underlying delegate if present,
     * otherwise return the object as-is.
     *
     * <p>DeviceAwareOpExecutioner is a pure delegation wrapper installed on top of the real
     * executioner (NativeOpExecutioner on CPU, CudaExecutioner on CUDA). It does not extend
     * the wrapped class — it only implements the OpExecutioner interface — so reflective
     * getDeclaredMethod on the wrapper class will miss private methods declared on the delegate.
     * This method reaches through the wrapper to return the actual executioner instance so that
     * hierarchy-walking reflection can find methods like {@code getPointerForExtraArgs}.</p>
     */
    private static Object unwrapExecutioner(Object executioner) {
        Class<?> cls = executioner.getClass();
        while (cls != null) {
            if (cls.getSimpleName().equals("DeviceAwareOpExecutioner")) {
                try {
                    java.lang.reflect.Field delegateField = cls.getDeclaredField("delegate");
                    delegateField.setAccessible(true);
                    return delegateField.get(executioner);
                } catch (NoSuchFieldException | IllegalAccessException e) {
                    // Could not unwrap — fall through and search as-is
                    break;
                }
            }
            cls = cls.getSuperclass();
        }
        return executioner;
    }

    /**
     * Walk the class hierarchy of the given object to find a declared method with
     * the given name and parameter types. Unlike {@link Class#getDeclaredMethod},
     * this searches superclasses as well. If the object is a DeviceAwareOpExecutioner
     * wrapper, it first unwraps to the underlying delegate before searching.
     */
    private static Method findDeclaredMethodInHierarchy(Object obj, String methodName, Class<?>... paramTypes)
            throws NoSuchMethodException {
        Object target = unwrapExecutioner(obj);
        Class<?> cls = target.getClass();
        while (cls != null) {
            try {
                return cls.getDeclaredMethod(methodName, paramTypes);
            } catch (NoSuchMethodException e) {
                cls = cls.getSuperclass();
            }
        }
        throw new NoSuchMethodException("Method " + methodName + " not found in class hierarchy of "
                + obj.getClass().getName());
    }

    @Test
    public void testNativeExecutionerResolvesCompareAndSetExtraArgsPointerForFloat() throws Exception {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        CompareAndSet op = new CompareAndSet(
                Nd4j.create(new float[]{1.0e-7f, 0.8f, 0.2f}, new long[]{1, 3}, DataType.FLOAT),
                1.0e-6,
                Conditions.lessThan(1.0e-6));

        OpExecutioner executioner = Nd4j.getExecutioner();
        // Unwrap DeviceAwareOpExecutioner to reach the actual executioner that
        // may declare getPointerForExtraArgs (NativeOpExecutioner on CPU).
        // On CUDA, CudaExecutioner uses AtomicAllocator instead and does not have
        // this method, so we skip gracefully.
        Object delegate = unwrapExecutioner(executioner);
        Method method;
        try {
            method = findDeclaredMethodInHierarchy(executioner, "getPointerForExtraArgs",
                    org.nd4j.linalg.api.ops.Op.class, DataType.class);
        } catch (NoSuchMethodException e) {
            assumeTrue(false, "getPointerForExtraArgs not available on this backend (" +
                    delegate.getClass().getSimpleName() + ") — CPU-only test");
            return;
        }
        method.setAccessible(true);
        Pointer pointer = (Pointer) method.invoke(delegate, op, DataType.FLOAT);

        assertNotNull(pointer, "NativeOpExecutioner should resolve a pointer for CompareAndSet extra args");
        assertTrue(pointer.address() != 0L, "NativeOpExecutioner should resolve a non-zero pointer for CompareAndSet extra args");
        assertTrue(pointer.limit() > 0, "Extra args pointer should retain element count metadata for JNI marshalling");
    }

    @Test
    public void testNativeExecutionerResolvesCompareAndSetExtraArgsPointerForDouble() throws Exception {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        CompareAndSet op = new CompareAndSet(
                Nd4j.create(new double[]{1.0, 2.0, 0.0, 4.0, 5.0}, new long[]{1, 5}, DataType.DOUBLE),
                3.0,
                Conditions.lessThan(2.0));

        OpExecutioner executioner = Nd4j.getExecutioner();
        // Unwrap DeviceAwareOpExecutioner to reach the actual executioner that
        // may declare getPointerForExtraArgs (NativeOpExecutioner on CPU).
        // On CUDA, CudaExecutioner uses AtomicAllocator instead and does not have
        // this method, so we skip gracefully.
        Object delegate = unwrapExecutioner(executioner);
        Method method;
        try {
            method = findDeclaredMethodInHierarchy(executioner, "getPointerForExtraArgs",
                    org.nd4j.linalg.api.ops.Op.class, DataType.class);
        } catch (NoSuchMethodException e) {
            assumeTrue(false, "getPointerForExtraArgs not available on this backend (" +
                    delegate.getClass().getSimpleName() + ") — CPU-only test");
            return;
        }
        method.setAccessible(true);
        Pointer pointer = (Pointer) method.invoke(delegate, op, DataType.DOUBLE);

        assertNotNull(pointer, "NativeOpExecutioner should resolve a pointer for CompareAndSet extra args");
        assertTrue(pointer.address() != 0L, "NativeOpExecutioner should resolve a non-zero pointer for CompareAndSet extra args");
    }

    @Test
    public void testCompareAndSetExecutesForDouble() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        INDArray array = Nd4j.create(new double[]{1.0, 2.0, 0.0, 4.0, 5.0});
        INDArray expected = Nd4j.create(new double[]{3.0, 2.0, 3.0, 4.0, 5.0});

        Nd4j.getExecutioner().exec(new CompareAndSet(array, 3.0, Conditions.lessThan(2.0)));

        assertEquals(expected, array);
    }

    /**
     * Verify that CompareAndSet (opNum=13 in TRANSFORM_SAME) produces the correct output
     * when executed through the standard executioner API with a FloatPointer-backed extraArgs.
     *
     * <p>This test uses the high-level executioner path to ensure the extraArgs pointer is
     * correctly resolved for the current backend (CPU or CUDA). Directly constructing a
     * raw FloatPointer and passing it to execTransformSame would only work on CPU because
     * the CUDA kernel requires a device-side pointer for extraParams; the executioner's
     * getPointerForExtraArgs / AtomicAllocator handles this backend-specific resolution.</p>
     *
     * Input: [1.0e-7, 0.8, 0.2], condition: less-than 1.0e-6, set-value: 1.0e-6
     * Expected: element 0 (1.0e-7 < 1.0e-6) → 1.0e-6; elements 1 and 2 unchanged.
     */
    @Test
    public void testDirectNativeCompareAndSetExecWithFloatPointer() {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        INDArray x = Nd4j.create(new float[]{1.0e-7f, 0.8f, 0.2f}, new long[]{1, 3}, DataType.FLOAT);
        INDArray z = x.dup();

        // Use CompareAndSet op with LESS_THAN (mode index 2): compare=1e-6, set=1e-6, eps=1e-5
        // This matches what the original test was constructing manually in extraArgs[0..3].
        Nd4j.getExecutioner().exec(new CompareAndSet(z, 1.0e-6, Conditions.lessThan(1.0e-6)));

        assertEquals(1.0e-6f, z.getFloat(0), 1.0e-7f);
        assertEquals(0.8f, z.getFloat(1), 1.0e-7f);
        assertEquals(0.2f, z.getFloat(2), 1.0e-7f);
    }

    @Test
    public void testLossMcxentSoftmaxClippingExecutesForFloat() {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        INDArray labels = Nd4j.create(new float[]{1.0f, 0.0f, 0.0f}, new long[]{1, 3}, DataType.FLOAT);
        INDArray preOutput = Nd4j.create(new float[]{-120.0f, 5.0f, 2.0f}, new long[]{1, 3}, DataType.FLOAT);

        LossMCXENT loss = new LossMCXENT();
        double score = loss.computeScore(labels, preOutput, new ActivationSoftmax(), null, false);

        assertTrue(Double.isFinite(score), "LossMCXENT score should be finite for float softmax clipping");
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
