package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Diagnostic test for all-zeros stdev bug.
 */
public class TestStdevDebug {

    @Test
    public void testPlaceholderSerDeser() throws Exception {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);

        int nOut = 4;
        int minibatch = 10;
        SDVariable input = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100).castTo(DataType.DOUBLE);

        SDVariable loss = sd.mean("loss", input);
        sd.associateArrayWithVariable(inputArr, input);

        // Check input array BEFORE serialization
        System.out.println("inputArr getDouble(0): " + inputArr.getDouble(0));
        System.out.println("inputArr isView: " + inputArr.isView());
        System.out.println("inputArr data isConstant: " + inputArr.data().isConstant());

        // Get from placeholder store
        INDArray phArr = sd.getVariable("in").getArr();
        System.out.println("phArr same as inputArr: " + (phArr == inputArr));
        System.out.println("phArr getDouble(0): " + phArr.getDouble(0));

        // Forward pass
        INDArray expOut = inputArr.mean();
        Map<String, INDArray> fwd = sd.output((Map<String, INDArray>) null, "loss");
        System.out.println("Forward pass loss: " + fwd.get("loss") + " expected: " + expOut);

        // Force sync input before serialization
        org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance().ensureHostAccess(inputArr);
        System.out.println("After ensureHostAccess inputArr getDouble(0): " + inputArr.getDouble(0));

        // Serialize
        java.nio.ByteBuffer serialized = sd.asFlatBuffers(true);
        SameDiff deserialized = SameDiff.fromFlatBuffers(serialized);
        deserialized.setDspAutoCompileEnabled(false);
        deserialized.setDspNativeAutoCompileEnabled(false);

        // Check deserialized variable
        SDVariable deserIn = deserialized.getVariable("in");
        System.out.println("deserIn type: " + deserIn.getVariableType());
        INDArray deserArr = deserIn.getArr();
        System.out.println("deserArr: " + (deserArr == null ? "NULL" : "not null"));
        if (deserArr != null) {
            System.out.println("deserArr isConstant: " + deserArr.data().isConstant());
            org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance().ensureHostAccess(deserArr);
            System.out.println("deserArr getDouble(0): " + deserArr.getDouble(0));
        }

        // Run deserialized graph
        Map<String, INDArray> deserOut = deserialized.outputAll((Map<String, INDArray>) null);
        System.out.println("Deserialized loss: " + deserOut.get("loss") + " expected: " + expOut);
        assertFalse(deserOut.get("loss").getDouble(0) == 0.0 && expOut.getDouble(0) != 0.0,
                "Deserialized loss should not be zero! Got: " + deserOut.get("loss") + " expected: " + expOut);
    }

    @Test
    public void testStdevSlotBySlot() {
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);

        try {
            INDArray input = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new int[]{3, 4});
            System.out.println("Input array: " + input);

            // Direct op execution (no SameDiff) - baseline
            INDArray directStd = input.std(true);
            System.out.println("Direct std (no SameDiff): " + directStd);

            // SameDiff execution with DSP disabled
            SameDiff sd = SameDiff.create();
            sd.setDspAutoCompileEnabled(false);
            sd.setDspNativeAutoCompileEnabled(false);

            SDVariable var = sd.var("in", input);
            SDVariable stdev = var.std(true);

            Map<String, INDArray> out = sd.output((Map<String, INDArray>) null, stdev.name());
            INDArray sdResult = out.get(stdev.name());
            System.out.println("SameDiff std result: " + sdResult);
            System.out.println("Expected: " + directStd);

            assertFalse(sdResult.getDouble(0) == 0.0, "SameDiff stdev should not be zero! Got: " + sdResult);
        } finally {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
        }
    }

    @Test
    public void testCompactCopyIdentity() {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        System.out.println("numDevices: " + numDevices);
        System.out.println("currentDevice: " + Nd4j.getAffinityManager().getDeviceForCurrentThread());

        // Step 1: createUninitialized + assign + commit (the compact copy pattern)
        INDArray original = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 4);
        assertTrue(original.isView(), "Reshaped linspace should be a view");
        System.out.println("original getDouble(0): " + original.getDouble(0));

        INDArray compactCopy = Nd4j.createUninitialized(original.dataType(), original.shape(), original.ordering());
        compactCopy.assign(original);
        Nd4j.getExecutioner().commit();

        // Step 2: Verify compact copy has data BEFORE DeviceLocalNDArray
        System.out.println("=== BEFORE DeviceLocalNDArray ===");
        org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance().ensureHostAccess(compactCopy);
        System.out.println("compactCopy getDouble(0) after force sync: " + compactCopy.getDouble(0));
        System.out.println("compactCopy data addr: " + compactCopy.data().addressPointer().address());
        assertEquals(1.0, compactCopy.getDouble(0), 1e-5, "Compact copy should have correct data before DLA");

        // Step 3: Store in DeviceLocalNDArray with delayed mode (like ThreadSafeArrayHolder)
        System.out.println("=== Storing in DeviceLocalNDArray(delayed=true) ===");
        org.nd4j.linalg.util.DeviceLocalNDArray dla = new org.nd4j.linalg.util.DeviceLocalNDArray(compactCopy, true);

        // Step 4: Retrieve from DeviceLocalNDArray
        INDArray retrieved = dla.get();
        System.out.println("retrieved identity: " + System.identityHashCode(retrieved));
        System.out.println("compactCopy identity: " + System.identityHashCode(compactCopy));
        System.out.println("same object? " + (compactCopy == retrieved));
        System.out.println("retrieved data addr: " + retrieved.data().addressPointer().address());

        // Step 5: Force sync and check
        org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance().ensureHostAccess(retrieved);
        System.out.println("retrieved getDouble(0) after force sync: " + retrieved.getDouble(0));
        assertEquals(1.0, retrieved.getDouble(0), 1e-5, "DLA.get() should return correct data");

        // Step 6: Manually replicate what sd.var does step by step
        System.out.println("=== Manual sd.var replication ===");
        INDArray view = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 4);
        System.out.println("view getDouble(0): " + view.getDouble(0));
        System.out.println("view isView: " + view.isView());

        // Compact copy (same as sd.var)
        INDArray cc2 = Nd4j.createUninitialized(view.dataType(), view.shape(), view.ordering());
        cc2.assign(view);
        Nd4j.getExecutioner().commit();
        org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance().ensureHostAccess(cc2);
        System.out.println("cc2 getDouble(0) after force sync: " + cc2.getDouble(0));
        System.out.println("cc2 data addr: " + cc2.data().addressPointer().address());

        // setCloseable (same as sd.var)
        cc2.setCloseable(false);

        // ThreadSafeArrayHolder.setArray path: detach + DLA
        INDArray toBroadcast = cc2.detach();
        System.out.println("toBroadcast same as cc2? " + (toBroadcast == cc2));
        System.out.println("toBroadcast getDouble(0): " + toBroadcast.getDouble(0));

        org.nd4j.linalg.util.DeviceLocalNDArray dla2 = new org.nd4j.linalg.util.DeviceLocalNDArray(toBroadcast, true);
        INDArray fromDla = dla2.get();
        System.out.println("fromDla same as toBroadcast? " + (fromDla == toBroadcast));
        org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance().ensureHostAccess(fromDla);
        System.out.println("fromDla getDouble(0) after force sync: " + fromDla.getDouble(0));

        assertEquals(1.0, fromDla.getDouble(0), 1e-5, "Manual replication should work");

        // Step 7: NOW go through actual sd.var
        System.out.println("=== Actual sd.var with view ===");
        INDArray view2 = Nd4j.linspace(1, 12, 12, DataType.DOUBLE).reshape(3, 4);
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.var("test", view2);
        INDArray sdArr = sd.getVariable("test").getArr();
        org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance().ensureHostAccess(sdArr);
        System.out.println("sd.var getDouble(0) after force sync: " + sdArr.getDouble(0));
        assertEquals(1.0, sdArr.getDouble(0), 1e-5, "sd.var should have correct data");
    }

    @Test
    public void testStdevViaOpValidation() throws Exception {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);

        List<Pair<INDArray, String>> testMatrices = NDArrayCreationUtil.getAllTestMatricesWithShape(3, 4, 12345, DataType.DOUBLE);
        List<String> failures = new java.util.ArrayList<>();

        for (int idx = 0; idx < testMatrices.size(); idx++) {
            Pair<INDArray, String> p = testMatrices.get(idx);
            INDArray arr = p.getFirst();
            String matrixType = p.getSecond();
            System.out.println("=== Matrix " + idx + ": " + matrixType + " ===");
            System.out.println("  isView=" + arr.isView() + " ordering=" + arr.ordering()
                    + " stride=" + java.util.Arrays.toString(arr.stride())
                    + " shape=" + java.util.Arrays.toString(arr.shape())
                    + " dataLen=" + arr.data().length());

            INDArray expStd = arr.std(false);
            System.out.println("  Expected std(false): " + expStd);

            // Serialize through SameDiff
            SameDiff sd = SameDiff.create();
            sd.setDspAutoCompileEnabled(false);
            sd.setDspNativeAutoCompileEnabled(false);
            SDVariable var = sd.var("in", arr);
            SDVariable stdev = var.std(false);

            // Check the stored variable data BEFORE serialization
            INDArray storedArr = sd.getVariable("in").getArr();
            org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance().ensureHostAccess(storedArr);
            System.out.println("  Stored var getDouble(0)=" + storedArr.getDouble(0) + " isView=" + storedArr.isView());

            // Serialize
            java.nio.ByteBuffer serialized = sd.asFlatBuffers(true);
            SameDiff deserialized = SameDiff.fromFlatBuffers(serialized);
            deserialized.setDspAutoCompileEnabled(false);
            deserialized.setDspNativeAutoCompileEnabled(false);

            // Check deserialized variable data
            INDArray deserArr = deserialized.getVariable("in").getArr();
            org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance().ensureHostAccess(deserArr);
            System.out.println("  Deserialized var getDouble(0)=" + deserArr.getDouble(0)
                    + " data=" + java.util.Arrays.toString(deserArr.data().asDouble()));

            // Run deserialized graph
            Map<String, INDArray> deserOut = deserialized.outputAll((Map<String, INDArray>) null);
            INDArray deserStd = deserOut.get(stdev.name());
            System.out.println("  Deserialized std: " + deserStd + " expected: " + expStd);

            if (deserStd.getDouble(0) == 0.0 && expStd.getDouble(0) != 0.0) {
                failures.add("Matrix " + idx + " (" + matrixType + "): expected " + expStd + " got " + deserStd);
            }
        }

        if (!failures.isEmpty()) {
            fail("Failures:\n" + String.join("\n", failures));
        }
    }

    @Test
    public void testEntropySerDeser() throws Exception {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        INDArray inputArr = Nd4j.linspace(0.01, 0.99, 12, DataType.DOUBLE).reshape('c', 3, 4);

        // Show intermediate values
        INDArray logArr = org.nd4j.linalg.ops.transforms.Transforms.log(inputArr, true);
        System.out.println("inputArr (should be unchanged): " + inputArr);
        System.out.println("logArr: " + logArr);
        INDArray xLogX = inputArr.mul(logArr);
        System.out.println("x * log(x): " + xLogX);
        INDArray summed = xLogX.sum();
        System.out.println("sum(x*log(x)): " + summed);

        // Direct computation for reference
        INDArray expected = summed.negi();
        org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance().ensureHostAccess(expected);
        System.out.println("Expected entropy (-sum(x*log(x))): " + expected);

        // Also check direct INDArray entropy
        Number directEntropy = inputArr.entropyNumber();
        System.out.println("Direct inputArr.entropyNumber(): " + directEntropy);

        // Build SameDiff graph
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        SDVariable input = sd.var("in", inputArr);
        SDVariable loss = sd.math().entropy("loss", input, 0, 1);

        // Run original
        Map<String, INDArray> origOut = sd.output((Map<String, INDArray>) null, "loss");
        INDArray origResult = origOut.get("loss");
        System.out.println("Original entropy result: " + origResult);

        // Serialize
        java.nio.ByteBuffer serialized = sd.asFlatBuffers(true);

        // Deserialize
        SameDiff deserialized = SameDiff.fromFlatBuffers(serialized);
        deserialized.setDspAutoCompileEnabled(false);
        deserialized.setDspNativeAutoCompileEnabled(false);

        // Check deserialized variable
        INDArray deserArr = deserialized.getVariable("in").getArr();
        org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance().ensureHostAccess(deserArr);
        System.out.println("Deserialized input getDouble(0): " + deserArr.getDouble(0) + " expected: " + inputArr.getDouble(0));

        // Check deserialized op type
        for (org.nd4j.autodiff.functions.DifferentialFunction f : deserialized.ops()) {
            System.out.println("Op: " + f.opName() + " class=" + f.getClass().getSimpleName()
                    + " opNum=" + (f instanceof org.nd4j.linalg.api.ops.Op ? ((org.nd4j.linalg.api.ops.Op)f).opNum() : "N/A"));
        }

        // Run deserialized
        Map<String, INDArray> deserOut = deserialized.output((Map<String, INDArray>) null, "loss");
        INDArray deserResult = deserOut.get("loss");
        System.out.println("Deserialized entropy result: " + deserResult);
        System.out.println("Expected: " + expected);

        assertEquals(expected.getDouble(0), deserResult.getDouble(0), 1e-4,
                "Deserialized entropy should match expected. Got: " + deserResult + " expected: " + expected);
    }

}
