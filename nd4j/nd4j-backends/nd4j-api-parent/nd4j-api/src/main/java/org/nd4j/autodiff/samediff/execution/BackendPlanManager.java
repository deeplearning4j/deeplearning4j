package org.nd4j.autodiff.samediff.execution;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import java.util.ArrayList;
import java.util.List;

/**
 * Unified device-specific backend plan management.
 *
 * Provides per-segment backend query, cache inspection/invalidation,
 * backend availability/priority listing, and backend selection override.
 */
public class BackendPlanManager {

    // ── Backend enumeration ──

    /**
     * List all available backends on current platform with priority order.
     */
    public static List<BackendInfo> getAvailableBackends(Pointer nativePlanHandle) {
        NativeOps ops = Nd4j.getNativeOps();
        String json = ops.getPlanAvailableBackends(nativePlanHandle);
        return BackendInfo.parseJsonArray(json);
    }

    /**
     * Get which backend compiled a specific segment.
     */
    public static String getSegmentBackendName(Pointer nativePlanHandle, int segIdx) {
        return Nd4j.getNativeOps().getPlanSegmentCompiledBackend(nativePlanHandle, segIdx);
    }

    /**
     * Get compilation audit for a segment (per-op compilation status from backend).
     */
    public static String getSegmentCompilationAudit(Pointer nativePlanHandle, int segIdx) {
        return Nd4j.getNativeOps().getPlanSegmentCompilationAudit(nativePlanHandle, segIdx);
    }

    // ── Cache management per backend ──

    /**
     * Invalidate compiled cache for a specific segment (forces recompilation).
     */
    public static void invalidateSegmentCache(Pointer nativePlanHandle, int segIdx) {
        Nd4j.getNativeOps().invalidatePlanSegmentCache(nativePlanHandle, segIdx);
    }

    /**
     * Invalidate ALL segment caches for a specific backend type.
     */
    public static void invalidateBackendCaches(Pointer nativePlanHandle, String backendName) {
        Nd4j.getNativeOps().invalidatePlanBackendCaches(nativePlanHandle, backendName);
    }

    /**
     * Get cache statistics for all backends.
     */
    public static String getBackendCacheStats(Pointer nativePlanHandle) {
        return Nd4j.getNativeOps().getPlanBackendCacheStats(nativePlanHandle);
    }

    // ── Per-device replay cache management ──

    /**
     * Load cached replay metadata for a specific device (bulk warm-start).
     */
    public static int loadReplayCacheForDevice(Pointer nativePlanHandle, DeviceKey device) {
        return Nd4j.getNativeOps().loadReplayCacheForDevice(nativePlanHandle,
            device.typeOrdinal(), device.index);
    }

    /**
     * Get per-device cache stats.
     */
    public static String getDeviceCacheStats() {
        return Nd4j.getNativeOps().getReplayCacheDeviceStatsJson();
    }

    /**
     * Clear replay cache for a specific device.
     */
    public static void clearDeviceCache(DeviceKey device) {
        Nd4j.getNativeOps().clearReplayCacheForDevice(device.typeOrdinal(), device.index);
    }

    /**
     * Migrate replay cache between compatible devices.
     */
    public static boolean migrateDeviceCache(DeviceKey from, DeviceKey to) {
        return Nd4j.getNativeOps().migrateReplayCache(
            from.typeOrdinal(), from.index, to.typeOrdinal(), to.index);
    }

    /**
     * Prune cache entries for devices that no longer exist.
     */
    public static int pruneStaleDeviceCaches() {
        return Nd4j.getNativeOps().pruneStaleReplayCacheDevices();
    }

    /**
     * Get all device keys that have cached replay data.
     */
    public static String getCachedDevicesJson() {
        return Nd4j.getNativeOps().getReplayCachedDevicesJson();
    }

    // ── Backend selection control ──

    /**
     * Override backend selection for a segment.
     * Pass null to restore automatic selection.
     */
    public static void setSegmentBackendOverride(Pointer nativePlanHandle, int segIdx,
                                                  String backendName) {
        Nd4j.getNativeOps().setPlanSegmentBackendOverride(nativePlanHandle, segIdx,
            backendName != null ? backendName : "");
    }

    /**
     * Set global backend priority override (reorder the priority chain).
     */
    public static void setBackendPriority(Pointer nativePlanHandle, String[] backendNames) {
        Nd4j.getNativeOps().setPlanBackendPriority(nativePlanHandle,
            String.join(",", backendNames));
    }

    // ── Backend info ──

    public static class BackendInfo {
        public String name;
        public String type;
        public boolean available;
        public int priority;
        public int compiledSegments;
        public String deviceInfo;

        /**
         * Parse a JSON array of backend info objects.
         * Simplified parser for the format: [{"name":"...","type":"...","available":true,"priority":0}, ...]
         */
        public static List<BackendInfo> parseJsonArray(String json) {
            List<BackendInfo> result = new ArrayList<>();
            if (json == null || json.isEmpty() || json.equals("[]")) return result;

            // Simple parsing - split by },{ and extract fields
            String inner = json.trim();
            if (inner.startsWith("[")) inner = inner.substring(1);
            if (inner.endsWith("]")) inner = inner.substring(0, inner.length() - 1);

            String[] entries = inner.split("\\},\\{");
            for (String entry : entries) {
                entry = entry.replace("{", "").replace("}", "");
                BackendInfo info = new BackendInfo();
                for (String field : entry.split(",")) {
                    String[] kv = field.split(":", 2);
                    if (kv.length != 2) continue;
                    String key = kv[0].trim().replace("\"", "");
                    String val = kv[1].trim().replace("\"", "");
                    switch (key) {
                        case "name": info.name = val; break;
                        case "type": info.type = val; break;
                        case "available": info.available = Boolean.parseBoolean(val); break;
                        case "priority": info.priority = Integer.parseInt(val); break;
                        case "compiledSegments": info.compiledSegments = Integer.parseInt(val); break;
                        case "deviceInfo": info.deviceInfo = val; break;
                    }
                }
                if (info.name != null) result.add(info);
            }
            return result;
        }
    }
}
