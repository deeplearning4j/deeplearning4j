package org.nd4j.autodiff.samediff.execution;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Manages replay profiles across training runs.
 *
 * A replay profile is the metadata for a captured graph at a specific shape configuration.
 * Profiles are keyed by shape hash so different batch sizes each have their own profile.
 *
 * Integration with TrainingSession:
 * - After successful shape-frozen execution, saves the profile
 * - On training resume, loads profiles to pre-warm cache for faster re-capture
 * - Shape change → new profile (different batch size)
 */
public class ReplayProfileManager {

    private static final Logger log = LoggerFactory.getLogger(ReplayProfileManager.class);

    // ── Profile lifecycle ──

    /**
     * Save current replay state for all segments to a profile keyed by shape hash.
     */
    public static ReplayProfile captureProfile(Pointer nativePlanHandle, Map<String, long[]> shapes) {
        NativeOps ops = Nd4j.getNativeOps();
        int numSegments = ops.getPlanNumSegments(nativePlanHandle);

        ReplayProfile profile = new ReplayProfile();
        profile.shapeHash = computeShapeHash(shapes);
        profile.shapes = new HashMap<>(shapes);
        profile.numSegments = numSegments;
        profile.captureTimestamp = System.currentTimeMillis();
        profile.segments = new ArrayList<>();

        for (int i = 0; i < numSegments; i++) {
            ReplayProfile.SegmentReplayInfo info = new ReplayProfile.SegmentReplayInfo();
            info.segIdx = i;
            info.capturable = ops.isPlanSegmentCapturable(nativePlanHandle, i);
            info.replayState = ops.getPlanSegmentReplayState(nativePlanHandle, i);
            info.replayCount = ops.getPlanSegmentReplayCount(nativePlanHandle, i);
            info.numCaptureBuffers = ops.getPlanSegmentNumCaptureBuffers(nativePlanHandle, i);

            String backendName = ops.getPlanSegmentBackendName(nativePlanHandle, i);
            if (backendName != null && !backendName.isEmpty()) {
                profile.backendName = backendName;
            }

            String statsJson = ops.getPlanSegmentStatisticsJson(nativePlanHandle, i);
            info.statisticsJson = statsJson;

            profile.segments.add(info);
        }

        return profile;
    }

    /**
     * Load a previously saved profile into the native plan's replay cache.
     * Returns true if the profile was loaded and provides warm-start hints.
     */
    public static boolean loadProfile(Pointer nativePlanHandle, ReplayProfile profile) {
        if (nativePlanHandle == null || profile == null) return false;

        // The C++ side uses ReplayCacheManager for warm-start hints.
        // We signal the profile's shape hash to help with cache lookup.
        NativeOps ops = Nd4j.getNativeOps();
        if (ops.isReplayCacheEnabled()) {
            // Loading the profile triggers cache lookup in C++ side
            DeviceKey device = DeviceKey.currentDevice();
            int loaded = ops.loadReplayCacheForDevice(nativePlanHandle,
                device.typeOrdinal(), device.index);
            if (loaded > 0) {
                log.info("Loaded {} cached replay entries for device {}", loaded, device);
                return true;
            }
        }
        return false;
    }

    /**
     * Save profile to disk (JSON). Returns file path.
     */
    public static String saveProfileToDisk(ReplayProfile profile, String directory) {
        try {
            Files.createDirectories(Paths.get(directory));
            String filename = "replay_profile_" + profile.shapeHash + ".json";
            String path = directory + File.separator + filename;
            try (Writer writer = new FileWriter(path)) {
                writer.write(profile.toJson());
            }
            return path;
        } catch (IOException e) {
            log.warn("Failed to save replay profile to disk: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Load profile from disk.
     */
    public static ReplayProfile loadProfileFromDisk(String filePath) {
        try {
            String json = new String(Files.readAllBytes(Paths.get(filePath)));
            return ReplayProfile.fromJson(json);
        } catch (IOException e) {
            log.warn("Failed to load replay profile from disk: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Capture a profile enriched with transfer analytics data.
     * Merges per-segment transfer stats from the analytics report into each SegmentReplayInfo.
     */
    public static ReplayProfile captureProfileWithAnalytics(
            Pointer nativePlanHandle, Map<String, long[]> shapes,
            DspReplayTransferAnalytics analytics) {

        ReplayProfile profile = captureProfile(nativePlanHandle, shapes);

        if (analytics != null) {
            DspReplayTransferAnalytics.ReplayTransferReport report = analytics.getReport();

            for (ReplayProfile.SegmentReplayInfo seg : profile.getSegments()) {
                DspReplayTransferAnalytics.SegmentTransferSummary segSummary =
                        report.getSegmentSummaries().get(seg.getSegIdx());
                if (segSummary != null) {
                    seg.setTransferBytes(segSummary.getTotalTransferBytes());
                    seg.setTransferCount(segSummary.getTotalTransferCount());
                    seg.setTransferDurationNanos(segSummary.getTotalTransferDurationNanos());

                    Map<String, Long> reasonMap = new HashMap<>();
                    for (Map.Entry<org.nd4j.linalg.framework.device.TransferReason,
                            DspReplayTransferAnalytics.TransferBreakdown> entry :
                            segSummary.getBreakdownByReason().entrySet()) {
                        reasonMap.put(entry.getKey().name(), entry.getValue().getBytes());
                    }
                    seg.setTransferBytesByReason(reasonMap);
                }
            }
        }

        return profile;
    }

    // ── Multi-profile management ──

    /**
     * Get or create a profile collection for a plan.
     */
    public static ReplayProfileCollection getProfiles(Pointer nativePlanHandle) {
        return new ReplayProfileCollection();
    }

    /**
     * Find best matching profile for given shapes.
     */
    public static ReplayProfile findMatchingProfile(ReplayProfileCollection profiles,
                                                     Map<String, long[]> shapes) {
        if (profiles == null) return null;
        long hash = computeShapeHash(shapes);
        return profiles.getExact(hash);
    }

    // ── Utility ──

    /**
     * Compute FNV-1a hash of placeholder shapes for profile keying.
     */
    public static long computeShapeHash(Map<String, long[]> shapes) {
        long hash = 0xcbf29ce484222325L;
        TreeMap<String, long[]> sorted = new TreeMap<>(shapes);
        for (Map.Entry<String, long[]> entry : sorted.entrySet()) {
            for (char c : entry.getKey().toCharArray()) {
                hash ^= c;
                hash *= 0x100000001b3L;
            }
            for (long dim : entry.getValue()) {
                hash ^= dim;
                hash *= 0x100000001b3L;
            }
        }
        return hash;
    }

    // ── Profile data classes ──

    /**
     * Replay profile for a specific shape configuration.
     */
    public static class ReplayProfile {
        private long shapeHash;
        private Map<String, long[]> shapes;
        private int numSegments;
        private List<SegmentReplayInfo> segments;
        private long captureTimestamp;
        private String backendName;

        // Analytics-enriched fields (backward-compatible: default to 0/null)
        private int primaryDeviceId;
        private Map<Integer, Long> deviceMemoryAtCapture;

        public long getShapeHash() { return shapeHash; }
        public Map<String, long[]> getShapes() { return shapes; }
        public int getNumSegments() { return numSegments; }
        public List<SegmentReplayInfo> getSegments() { return segments; }
        public long getCaptureTimestamp() { return captureTimestamp; }
        public String getBackendName() { return backendName; }
        public int getPrimaryDeviceId() { return primaryDeviceId; }
        public void setPrimaryDeviceId(int deviceId) { this.primaryDeviceId = deviceId; }
        public Map<Integer, Long> getDeviceMemoryAtCapture() { return deviceMemoryAtCapture; }
        public void setDeviceMemoryAtCapture(Map<Integer, Long> mem) { this.deviceMemoryAtCapture = mem; }

        public static class SegmentReplayInfo {
            int segIdx;
            boolean capturable;
            int replayState;
            int replayCount;
            int numCaptureBuffers;
            String statisticsJson;

            // Analytics-enriched fields
            int executionDeviceId;
            long transferBytes;
            long transferCount;
            long transferDurationNanos;
            Map<String, Long> transferBytesByReason;

            public int getSegIdx() { return segIdx; }
            public boolean isCapturable() { return capturable; }
            public int getReplayState() { return replayState; }
            public int getReplayCount() { return replayCount; }
            public int getNumCaptureBuffers() { return numCaptureBuffers; }
            public int getExecutionDeviceId() { return executionDeviceId; }
            public void setExecutionDeviceId(int id) { this.executionDeviceId = id; }
            public long getTransferBytes() { return transferBytes; }
            public void setTransferBytes(long b) { this.transferBytes = b; }
            public long getTransferCount() { return transferCount; }
            public void setTransferCount(long c) { this.transferCount = c; }
            public long getTransferDurationNanos() { return transferDurationNanos; }
            public void setTransferDurationNanos(long d) { this.transferDurationNanos = d; }
            public Map<String, Long> getTransferBytesByReason() { return transferBytesByReason; }
            public void setTransferBytesByReason(Map<String, Long> m) { this.transferBytesByReason = m; }
        }

        /**
         * Simple JSON serialization.
         */
        public String toJson() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\"shapeHash\":").append(shapeHash);
            sb.append(",\"numSegments\":").append(numSegments);
            sb.append(",\"captureTimestamp\":").append(captureTimestamp);
            sb.append(",\"backendName\":\"").append(backendName != null ? backendName : "").append("\"");
            sb.append(",\"primaryDeviceId\":").append(primaryDeviceId);

            // Device memory at capture
            sb.append(",\"deviceMemoryAtCapture\":{");
            if (deviceMemoryAtCapture != null) {
                boolean dmFirst = true;
                for (Map.Entry<Integer, Long> dm : deviceMemoryAtCapture.entrySet()) {
                    if (!dmFirst) sb.append(",");
                    dmFirst = false;
                    sb.append("\"").append(dm.getKey()).append("\":").append(dm.getValue());
                }
            }
            sb.append("}");

            // Shapes
            sb.append(",\"shapes\":{");
            boolean first = true;
            if (shapes != null) {
                for (Map.Entry<String, long[]> entry : shapes.entrySet()) {
                    if (!first) sb.append(",");
                    first = false;
                    sb.append("\"").append(entry.getKey()).append("\":[");
                    long[] dims = entry.getValue();
                    for (int i = 0; i < dims.length; i++) {
                        if (i > 0) sb.append(",");
                        sb.append(dims[i]);
                    }
                    sb.append("]");
                }
            }
            sb.append("}");

            // Segments
            sb.append(",\"segments\":[");
            if (segments != null) {
                for (int i = 0; i < segments.size(); i++) {
                    if (i > 0) sb.append(",");
                    SegmentReplayInfo info = segments.get(i);
                    sb.append("{\"segIdx\":").append(info.segIdx);
                    sb.append(",\"capturable\":").append(info.capturable);
                    sb.append(",\"replayState\":").append(info.replayState);
                    sb.append(",\"replayCount\":").append(info.replayCount);
                    sb.append(",\"numCaptureBuffers\":").append(info.numCaptureBuffers);
                    sb.append(",\"executionDeviceId\":").append(info.executionDeviceId);
                    sb.append(",\"transferBytes\":").append(info.transferBytes);
                    sb.append(",\"transferCount\":").append(info.transferCount);
                    sb.append(",\"transferDurationNanos\":").append(info.transferDurationNanos);
                    sb.append("}");
                }
            }
            sb.append("]}");
            return sb.toString();
        }

        /**
         * Simple JSON deserialization.
         */
        public static ReplayProfile fromJson(String json) {
            ReplayProfile profile = new ReplayProfile();
            // Extract shapeHash
            profile.shapeHash = extractLong(json, "shapeHash");
            profile.numSegments = (int) extractLong(json, "numSegments");
            profile.captureTimestamp = extractLong(json, "captureTimestamp");
            profile.backendName = extractString(json, "backendName");
            profile.primaryDeviceId = (int) extractLong(json, "primaryDeviceId");
            profile.shapes = new HashMap<>();
            profile.segments = new ArrayList<>();

            // Parse segments array
            int segStart = json.indexOf("\"segments\":[");
            if (segStart >= 0) {
                segStart = json.indexOf("[", segStart) + 1;
                int segEnd = json.lastIndexOf("]");
                if (segEnd > segStart) {
                    String segArray = json.substring(segStart, segEnd);
                    String[] segEntries = segArray.split("\\},\\{");
                    for (String entry : segEntries) {
                        entry = entry.replace("{", "").replace("}", "");
                        SegmentReplayInfo info = new SegmentReplayInfo();
                        info.segIdx = (int) extractLongFromEntry(entry, "segIdx");
                        info.capturable = extractLongFromEntry(entry, "capturable") != 0
                            || entry.contains("\"capturable\":true");
                        info.replayState = (int) extractLongFromEntry(entry, "replayState");
                        info.replayCount = (int) extractLongFromEntry(entry, "replayCount");
                        info.numCaptureBuffers = (int) extractLongFromEntry(entry, "numCaptureBuffers");
                        info.executionDeviceId = (int) extractLongFromEntry(entry, "executionDeviceId");
                        info.transferBytes = extractLongFromEntry(entry, "transferBytes");
                        info.transferCount = extractLongFromEntry(entry, "transferCount");
                        info.transferDurationNanos = extractLongFromEntry(entry, "transferDurationNanos");
                        profile.segments.add(info);
                    }
                }
            }

            return profile;
        }

        private static long extractLong(String json, String field) {
            String key = "\"" + field + "\":";
            int pos = json.indexOf(key);
            if (pos < 0) return 0;
            pos += key.length();
            int end = pos;
            while (end < json.length() && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '-')) end++;
            if (end == pos) return 0;
            return Long.parseLong(json.substring(pos, end));
        }

        private static long extractLongFromEntry(String entry, String field) {
            String key = "\"" + field + "\":";
            int pos = entry.indexOf(key);
            if (pos < 0) return 0;
            pos += key.length();
            int end = pos;
            while (end < entry.length() && (Character.isDigit(entry.charAt(end)) || entry.charAt(end) == '-')) end++;
            if (end == pos) return 0;
            return Long.parseLong(entry.substring(pos, end));
        }

        private static String extractString(String json, String field) {
            String key = "\"" + field + "\":\"";
            int pos = json.indexOf(key);
            if (pos < 0) return "";
            pos += key.length();
            int end = json.indexOf("\"", pos);
            if (end < 0) return "";
            return json.substring(pos, end);
        }
    }

    /**
     * Collection of replay profiles keyed by shape hash.
     */
    public static class ReplayProfileCollection {
        private final Map<Long, ReplayProfile> profilesByShapeHash = new HashMap<>();

        public void put(ReplayProfile profile) {
            profilesByShapeHash.put(profile.shapeHash, profile);
        }

        public ReplayProfile getExact(long shapeHash) {
            return profilesByShapeHash.get(shapeHash);
        }

        public int size() { return profilesByShapeHash.size(); }

        public String summary() {
            StringBuilder sb = new StringBuilder();
            sb.append("ReplayProfileCollection: ").append(size()).append(" profiles\n");
            for (Map.Entry<Long, ReplayProfile> entry : profilesByShapeHash.entrySet()) {
                ReplayProfile p = entry.getValue();
                sb.append("  hash=").append(entry.getKey())
                  .append(" segs=").append(p.numSegments)
                  .append(" backend=").append(p.backendName)
                  .append(" ts=").append(p.captureTimestamp)
                  .append("\n");
            }
            return sb.toString();
        }

        /**
         * Save all profiles to directory.
         */
        public void saveToDisk(String directory) {
            for (ReplayProfile profile : profilesByShapeHash.values()) {
                ReplayProfileManager.saveProfileToDisk(profile, directory);
            }
        }

        /**
         * Load all profiles from directory.
         */
        public static ReplayProfileCollection loadFromDisk(String directory) {
            ReplayProfileCollection collection = new ReplayProfileCollection();
            try {
                Path dir = Paths.get(directory);
                if (!Files.exists(dir)) return collection;
                try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "replay_profile_*.json")) {
                    for (Path path : stream) {
                        ReplayProfile profile = ReplayProfileManager.loadProfileFromDisk(path.toString());
                        if (profile != null) {
                            collection.put(profile);
                        }
                    }
                }
            } catch (IOException e) {
                log.warn("Failed to load replay profiles from {}: {}", directory, e.getMessage());
            }
            return collection;
        }
    }
}
