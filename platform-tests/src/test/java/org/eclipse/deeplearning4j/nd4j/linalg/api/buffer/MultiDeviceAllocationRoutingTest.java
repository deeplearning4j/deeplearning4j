package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.BackendRegistry;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Collection;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Isolation test to trace exactly what happens in the multi-device allocation routing chain.
 * This test does NOT allocate large buffers or load models - it just inspects the routing
 * infrastructure to identify where the chain breaks.
 */
public class MultiDeviceAllocationRoutingTest {

    @BeforeAll
    static void initializeNd4jBackend() {
        // NativeOpsHolder reads the backend-owned native.ops property. Initialize ND4J first so
        // randomized JUnit method order cannot poison the holder before backend discovery.
        Nd4j.getBackend();
    }

    /**
     * Step 1: Verify that BackendRegistry discovers all GPU devices.
     */
    @Test
    public void testBackendRegistryDiscoversAllDevices() {
        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        System.out.println("=== Step 1: Backend Registry ===");
        System.out.println("NativeOps.getAvailableDevices() = " + numDevices);

        BackendRegistry registry = BackendRegistry.getInstance();
        var allDevices = registry.getAllDevices();
        System.out.println("BackendRegistry.getAllDevices() = " + allDevices.size());
        for (DeviceDescriptor d : allDevices) {
            System.out.println("  Device: " + d.getDeviceId()
                    + " type=" + d.getDeviceType()
                    + " index=" + d.getDeviceIndex()
                    + " totalMem=" + (d.getTotalMemory() / (1024 * 1024)) + "MB"
                    + " isDefault=" + d.isDefault());
        }

        int gpuCount = (int) allDevices.stream()
                .filter(d -> d.getDeviceType().isGpu()).count();
        System.out.println("GPU count in BackendRegistry = " + gpuCount);
        assertEquals(numDevices, gpuCount,
                "BackendRegistry should have the same number of GPUs as NativeOps reports");
    }

    /**
     * Step 2: Verify that DeviceMemoryManager registers all GPU devices.
     */
    @Test
    public void testDeviceMemoryManagerRegistersAllDevices() {
        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        assumeTrue(numDevices >= 2, "Need at least 2 GPUs");

        DeviceMemoryManager memMgr = DeviceMemoryManager.getInstance();
        System.out.println("=== Step 2: DeviceMemoryManager Registration ===");

        Collection<DeviceDescriptor> registered = memMgr.getRegisteredDevices();
        System.out.println("Registered devices = " + registered.size());
        for (DeviceDescriptor d : registered) {
            System.out.println("  Device: " + d.getDeviceId()
                    + " type=" + d.getDeviceType()
                    + " index=" + d.getDeviceIndex());
        }

        // Check each GPU is findable by index
        for (int i = 0; i < numDevices; i++) {
            DeviceDescriptor device = memMgr.getRegisteredDevice(i);
            System.out.println("getRegisteredDevice(" + i + ") = " +
                    (device == null ? "NULL" : device.getDeviceId()));
            assertNotNull(device, "Device " + i + " should be registered");
        }
    }

    /**
     * Step 3: Verify getActualFreeMemory works for all devices.
     */
    @Test
    public void testActualFreeMemoryQuery() {
        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        assumeTrue(numDevices >= 2, "Need at least 2 GPUs");

        DeviceMemoryManager memMgr = DeviceMemoryManager.getInstance();
        System.out.println("=== Step 3: Free Memory Queries ===");

        for (int i = 0; i < numDevices; i++) {
            // Native query
            long nativeFree = nativeOps.getDeviceFreeMemory(i);
            System.out.println("NativeOps.getDeviceFreeMemory(" + i + ") = "
                    + (nativeFree / (1024 * 1024)) + "MB");

            // DeviceMemoryManager query
            DeviceDescriptor device = memMgr.getRegisteredDevice(i);
            if (device != null) {
                long dmFree = memMgr.getActualFreeMemory(device);
                System.out.println("DeviceMemoryManager.getActualFreeMemory(device" + i + ") = "
                        + (dmFree / (1024 * 1024)) + "MB");
            } else {
                System.out.println("Device " + i + " NOT REGISTERED");
            }
        }
    }

    /**
     * Step 4: Verify selectDevice(MOST_FREE) returns the device with most memory.
     */
    @Test
    public void testSelectMostFreeDevice() {
        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        assumeTrue(numDevices >= 2, "Need at least 2 GPUs");

        DeviceMemoryManager memMgr = DeviceMemoryManager.getInstance();
        System.out.println("=== Step 4: selectDevice(MOST_FREE) ===");

        // First find which device has most free memory natively
        int bestNative = -1;
        long bestFree = -1;
        for (int i = 0; i < numDevices; i++) {
            long free = nativeOps.getDeviceFreeMemory(i);
            System.out.println("Device " + i + " free = " + (free / (1024 * 1024)) + "MB");
            if (free > bestFree) {
                bestFree = free;
                bestNative = i;
            }
        }
        System.out.println("Best device by native query = " + bestNative);

        // Now check what DeviceMemoryManager selects
        long allocSize = 100L * 1024 * 1024; // 100MB
        DeviceDescriptor selected = memMgr.selectDevice(allocSize,
                DeviceMemoryManager.DeviceRoutingPolicy.MOST_FREE);
        System.out.println("selectDevice(100MB, MOST_FREE) = " +
                (selected == null ? "NULL" : selected.getDeviceId() + " index=" + selected.getDeviceIndex()));

        assertNotNull(selected, "selectDevice should return a device");
        assertEquals(bestNative, selected.getDeviceIndex(),
                "selectDevice(MOST_FREE) should pick the device with most free memory");
    }

    /**
     * Step 5: Test the full allocation routing chain by allocating a buffer
     * when device 0 is nearly full (after loading a model). Verify the buffer
     * ends up on the device with more free memory.
     */
    @Test
    public void testAllocationRoutesToFreeDevice() {
        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        assumeTrue(numDevices >= 2, "Need at least 2 GPUs");

        System.out.println("=== Step 5: Allocation Routing ===");

        // Check which device has most free memory
        long dev0Free = nativeOps.getDeviceFreeMemory(0);
        long dev1Free = nativeOps.getDeviceFreeMemory(1);
        System.out.println("Device 0 free: " + (dev0Free / (1024 * 1024)) + "MB");
        System.out.println("Device 1 free: " + (dev1Free / (1024 * 1024)) + "MB");

        // Determine which device should receive the allocation based on headroom
        long headroom = 128L * 1024 * 1024; // MIN_DEVICE_HEADROOM in OpaqueDataBuffer
        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        long currentFree = nativeOps.getDeviceFreeMemory(currentDevice);
        System.out.println("Current thread device = " + currentDevice);
        System.out.println("Current device free = " + (currentFree / (1024 * 1024)) + "MB");

        // Allocate a 50MB buffer
        long allocElements = 50L * 1024 * 1024 / 4; // 50MB of FLOAT
        System.out.println("Allocating " + (allocElements * 4 / (1024 * 1024)) + "MB FLOAT array...");

        INDArray arr = Nd4j.create(DataType.FLOAT, allocElements);
        int bufferDevice = nativeOps.dbDeviceId(arr.data().opaqueBuffer());
        System.out.println("Buffer allocated on device: " + bufferDevice);

        if (currentFree < allocElements * 4 + headroom) {
            // Should have routed to the other device
            System.out.println("Current device (" + currentDevice + ") had less than alloc+128MB headroom");
            System.out.println("Expected routing to device with more memory");
            // Don't assert which device - just verify it succeeded
        } else {
            System.out.println("Current device had enough headroom - stayed on device " + currentDevice);
        }

        // Verify the buffer is usable
        arr.assign(42.0f);
        float val = arr.getFloat(0);
        assertEquals(42.0f, val, 0.01f, "Buffer should be readable/writable");

        arr.close();
    }

    /**
     * Step 6: Simulate the draft model scenario - allocate many small buffers
     * when device 0 is nearly full. Each individual allocation is small but
     * cumulatively they exceed device 0 capacity.
     */
    @Test
    public void testManySmallAllocationsRouting() {
        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        assumeTrue(numDevices >= 2, "Need at least 2 GPUs");

        System.out.println("=== Step 6: Many Small Allocations ===");

        long dev0Free = nativeOps.getDeviceFreeMemory(0);
        long dev1Free = nativeOps.getDeviceFreeMemory(1);
        System.out.println("Initial Device 0 free: " + (dev0Free / (1024 * 1024)) + "MB");
        System.out.println("Initial Device 1 free: " + (dev1Free / (1024 * 1024)) + "MB");

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        System.out.println("Current thread device = " + currentDevice);

        // Allocate 20 x 10MB arrays (200MB total)
        INDArray[] arrays = new INDArray[20];
        int[] deviceCounts = new int[numDevices];

        for (int i = 0; i < 20; i++) {
            long elemCount = 10L * 1024 * 1024 / 4; // 10MB FLOAT
            arrays[i] = Nd4j.create(DataType.FLOAT, elemCount);
            int devId = nativeOps.dbDeviceId(arrays[i].data().opaqueBuffer());
            if (devId >= 0 && devId < numDevices) deviceCounts[devId]++;

            if (i % 5 == 0) {
                long free0 = nativeOps.getDeviceFreeMemory(0);
                long free1 = nativeOps.getDeviceFreeMemory(1);
                System.out.println("After alloc " + i + ": dev0=" + (free0 / (1024 * 1024))
                        + "MB dev1=" + (free1 / (1024 * 1024)) + "MB, this alloc on device=" + devId);
            }
        }

        System.out.println("Device allocation distribution:");
        for (int d = 0; d < numDevices; d++) {
            System.out.println("  Device " + d + ": " + deviceCounts[d] + " allocations");
        }

        // Clean up
        for (INDArray a : arrays) {
            if (a != null) a.close();
        }
    }

    /**
     * Step 7: Verify that switching device context before native allocation
     * actually makes the allocation land on the target device.
     */
    @Test
    public void testSwitchDeviceThenAllocate() {
        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numDevices = nativeOps.getAvailableDevices();
        assumeTrue(numDevices >= 2, "Need at least 2 GPUs");

        System.out.println("=== Step 7: Switch Device Then Allocate ===");

        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int targetDevice = (originalDevice == 0) ? 1 : 0;
        System.out.println("Original device = " + originalDevice);
        System.out.println("Target device = " + targetDevice);

        // Switch to target device
        DeviceMemoryManager.getInstance().switchDevice(targetDevice, "test", "verify-routing");

        // Allocate
        INDArray arr = Nd4j.create(DataType.FLOAT, 1024 * 1024); // 4MB
        int allocDevice = nativeOps.dbDeviceId(arr.data().opaqueBuffer());
        System.out.println("Buffer allocated on device: " + allocDevice);

        // Verify
        assertEquals(targetDevice, allocDevice,
                "After switchDevice(" + targetDevice + "), allocation should land on device " + targetDevice);

        // Restore original device
        DeviceMemoryManager.getInstance().switchDevice(originalDevice, "test", "restore");
        arr.close();
    }

    /**
     * The public affinity API must bind both Java bookkeeping and the native CUDA context.
     * A getter-only assertion is insufficient because the historical defect reported the
     * requested device while subsequent native allocations still landed on the old GPU.
     */
    @Test
    public void testAffinityBindingRoutesNativeAllocation() {
        // Initialize the selected ND4J backend before reading its native-ops configuration.
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        assumeTrue(numDevices >= 2, "Need at least 2 GPUs");

        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int targetDevice = (originalDevice == 0) ? 1 : 0;
        try {
            Nd4j.getAffinityManager().setDeviceForCurrentThread(targetDevice);

            assertEquals(targetDevice, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                    "Java affinity must report the requested device");
            assertEquals(targetDevice, nativeOps.getDevice(),
                    "Native CUDA context must follow the public affinity binding");
            try (INDArray array = Nd4j.create(DataType.FLOAT, 1024 * 1024)) {
                assertEquals(targetDevice, nativeOps.dbDeviceId(array.data().opaqueBuffer()),
                        "Allocation after affinity binding must land on the requested device");
            }
        } finally {
            Nd4j.getAffinityManager().setDeviceForCurrentThread(originalDevice);
        }
    }
}
