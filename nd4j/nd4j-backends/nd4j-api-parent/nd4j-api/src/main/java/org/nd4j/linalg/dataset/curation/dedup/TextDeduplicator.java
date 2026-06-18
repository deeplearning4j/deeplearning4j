package org.nd4j.linalg.dataset.curation.dedup;

import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Two-level text deduplication: exact (SHA-256) and near-duplicate (MinHash LSH).
 * Supports streaming mode for large datasets.
 */
public class TextDeduplicator implements Serializable {
    private static final long serialVersionUID = 1L;

    private final boolean useExact;
    private final boolean useMinHash;
    private final MinHasher minHasher;
    private final int numBands;
    private final int rowsPerBand;
    private final double jaccardThreshold;

    // State for streaming mode
    private final Set<String> seenHashes = new HashSet<>();
    private final List<int[]> signatures = new ArrayList<>();
    private final Map<String, List<Integer>> lshBuckets = new HashMap<>();

    private TextDeduplicator(Builder builder) {
        this.useExact = builder.useExact;
        this.useMinHash = builder.useMinHash;
        this.jaccardThreshold = builder.jaccardThreshold;

        if (useMinHash) {
            int numHashes = builder.numBands * builder.rowsPerBand;
            this.minHasher = new MinHasher(builder.shingleSize, numHashes, builder.seed);
            this.numBands = builder.numBands;
            this.rowsPerBand = builder.rowsPerBand;
        } else {
            this.minHasher = null;
            this.numBands = 0;
            this.rowsPerBand = 0;
        }
    }

    /**
     * Deduplicate a list of texts, returning unique texts.
     */
    public List<String> deduplicate(List<String> texts) {
        Set<Integer> dupIndices = findDuplicateIndices(texts);
        List<String> result = new ArrayList<>();
        for (int i = 0; i < texts.size(); i++) {
            if (!dupIndices.contains(i)) {
                result.add(texts.get(i));
            }
        }
        return result;
    }

    /**
     * Find indices of duplicate texts. For each duplicate group, the first occurrence is kept.
     */
    public Set<Integer> findDuplicateIndices(List<String> texts) {
        Set<Integer> duplicates = new HashSet<>();

        if (useExact) {
            Map<String, Integer> hashToFirst = new HashMap<>();
            for (int i = 0; i < texts.size(); i++) {
                String hash = sha256(normalize(texts.get(i)));
                Integer first = hashToFirst.putIfAbsent(hash, i);
                if (first != null) {
                    duplicates.add(i);
                }
            }
        }

        if (useMinHash) {
            List<int[]> sigs = new ArrayList<>();
            Map<String, List<Integer>> buckets = new HashMap<>();

            for (int i = 0; i < texts.size(); i++) {
                if (duplicates.contains(i)) {
                    sigs.add(null);
                    continue;
                }
                int[] sig = minHasher.signature(texts.get(i));
                sigs.add(sig);

                // LSH banding
                for (int b = 0; b < numBands; b++) {
                    String bucketKey = bandKey(b, sig);
                    buckets.computeIfAbsent(bucketKey, k -> new ArrayList<>()).add(i);
                }
            }

            // Check candidate pairs
            Set<String> checkedPairs = new HashSet<>();
            for (List<Integer> bucket : buckets.values()) {
                if (bucket.size() < 2) continue;
                for (int i = 0; i < bucket.size(); i++) {
                    for (int j = i + 1; j < bucket.size(); j++) {
                        int idx1 = bucket.get(i);
                        int idx2 = bucket.get(j);
                        if (duplicates.contains(idx2)) continue;

                        String pairKey = idx1 + ":" + idx2;
                        if (checkedPairs.contains(pairKey)) continue;
                        checkedPairs.add(pairKey);

                        double sim = minHasher.estimatedJaccard(sigs.get(idx1), sigs.get(idx2));
                        if (sim >= jaccardThreshold) {
                            duplicates.add(idx2);
                        }
                    }
                }
            }
        }

        return duplicates;
    }

    /**
     * Streaming mode: check if a new text is a duplicate of previously seen texts.
     * Returns true if the text is unique (not a duplicate).
     */
    public boolean addIfUnique(String text) {
        if (useExact) {
            String hash = sha256(normalize(text));
            if (!seenHashes.add(hash)) {
                return false;
            }
        }

        if (useMinHash) {
            int[] sig = minHasher.signature(text);
            int newIdx = signatures.size();

            // Check against existing LSH buckets
            for (int b = 0; b < numBands; b++) {
                String key = bandKey(b, sig);
                List<Integer> bucket = lshBuckets.get(key);
                if (bucket != null) {
                    for (int existIdx : bucket) {
                        double sim = minHasher.estimatedJaccard(signatures.get(existIdx), sig);
                        if (sim >= jaccardThreshold) {
                            return false;
                        }
                    }
                }
            }

            // Not a duplicate: add to index
            signatures.add(sig);
            for (int b = 0; b < numBands; b++) {
                String key = bandKey(b, sig);
                lshBuckets.computeIfAbsent(key, k -> new ArrayList<>()).add(newIdx);
            }
        }

        return true;
    }

    /**
     * Reset streaming state.
     */
    public void reset() {
        seenHashes.clear();
        signatures.clear();
        lshBuckets.clear();
    }

    private String bandKey(int band, int[] sig) {
        StringBuilder sb = new StringBuilder();
        sb.append(band).append(':');
        int start = band * rowsPerBand;
        for (int r = 0; r < rowsPerBand; r++) {
            if (r > 0) sb.append(',');
            sb.append(sig[start + r]);
        }
        return sb.toString();
    }

    private static String normalize(String text) {
        return text.toLowerCase().replaceAll("\\s+", " ").trim();
    }

    private static String sha256(String text) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(text.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 not available", e);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private boolean useExact = true;
        private boolean useMinHash = false;
        private int shingleSize = 5;
        private int numBands = 16;
        private int rowsPerBand = 8;
        private double jaccardThreshold = 0.8;
        private long seed = 42L;

        public Builder useExact(boolean use) { this.useExact = use; return this; }
        public Builder useMinHash(boolean use) { this.useMinHash = use; return this; }
        public Builder shingleSize(int size) { this.shingleSize = size; return this; }
        public Builder numBands(int bands) { this.numBands = bands; return this; }
        public Builder rowsPerBand(int rows) { this.rowsPerBand = rows; return this; }
        public Builder jaccardThreshold(double threshold) { this.jaccardThreshold = threshold; return this; }
        public Builder seed(long seed) { this.seed = seed; return this; }

        public TextDeduplicator build() {
            return new TextDeduplicator(this);
        }
    }
}
