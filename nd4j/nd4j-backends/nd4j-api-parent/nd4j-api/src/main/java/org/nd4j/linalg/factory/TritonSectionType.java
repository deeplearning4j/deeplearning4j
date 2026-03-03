package org.nd4j.linalg.factory;

/**
 * Mirror of the C++ KernelSectionType enum used by the Triton GPU compiler backend.
 * Each value corresponds to a bit position in the fallback/mergeable section bitmasks
 * configurable via {@link Environment#setTritonFallbackSections(int)} and
 * {@link Environment#setTritonMergeableSections(int)}.
 *
 * <p>Usage example (from a test):
 * <pre>
 * // Compile only ELEMENTWISE and REDUCTION; everything else falls back to native
 * int fallback = TritonSectionType.allBits()
 *     & ~TritonSectionType.ELEMENTWISE.bit()
 *     & ~TritonSectionType.REDUCTION.bit();
 * Nd4j.getEnvironment().setTritonFallbackSections(fallback);
 * </pre>
 */
public enum TritonSectionType {
    ELEMENTWISE(0),
    MATMUL(1),
    FUSED_ATTENTION(2),
    REDUCTION(3),
    NORMALIZATION(4),
    GATHER(5),
    GATHER_ND(6),
    CONCAT(7),
    SPLIT(8),
    SPLIT_V(9),
    STACK(10),
    STRIDED_SLICE(11),
    TILE(12),
    SCATTER_ND(13),
    SCATTER_ND_UPDATE(14),
    SHAPE_MANIPULATION(15),
    CONSTANT_GENERATION(16),
    CONVOLUTION(17),
    IDENTITY(18);

    private final int ordinalValue;

    TritonSectionType(int ordinalValue) {
        this.ordinalValue = ordinalValue;
    }

    /** The bit position (0-18) corresponding to this section type. */
    public int ordinalValue() {
        return ordinalValue;
    }

    /** The bitmask with just this type's bit set. */
    public int bit() {
        return 1 << ordinalValue;
    }

    /** Bitmask with all section type bits set. */
    public static int allBits() {
        int mask = 0;
        for (TritonSectionType t : values()) {
            mask |= t.bit();
        }
        return mask;
    }

    /** Build a bitmask from multiple section types. */
    public static int mask(TritonSectionType... types) {
        int mask = 0;
        for (TritonSectionType t : types) {
            mask |= t.bit();
        }
        return mask;
    }

    /**
     * Default fallback bitmask: MATMUL, FUSED_ATTENTION, CONVOLUTION, and all DATA_MOVEMENT types.
     * These fall back to native execution (cuBLAS, cuDNN, etc.).
     */
    public static final int DEFAULT_FALLBACK = mask(
            MATMUL, FUSED_ATTENTION, CONVOLUTION,
            GATHER, GATHER_ND, CONCAT, SPLIT, SPLIT_V, STACK,
            STRIDED_SLICE, TILE, SCATTER_ND, SCATTER_ND_UPDATE
    );

    /**
     * Default mergeable bitmask: types that can be grouped into element-wise chains for fusion.
     */
    public static final int DEFAULT_MERGEABLE = mask(
            ELEMENTWISE, IDENTITY, CONSTANT_GENERATION,
            SHAPE_MANIPULATION, REDUCTION, NORMALIZATION
    );
}
