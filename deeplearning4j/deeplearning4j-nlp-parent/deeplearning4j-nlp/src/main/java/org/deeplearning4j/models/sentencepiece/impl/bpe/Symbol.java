package org.deeplearning4j.models.sentencepiece.impl.bpe;

import lombok.*;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class Symbol {

    private Symbol left;
    private Symbol right;

    @Builder.Default
    private boolean unknown = false;

    @Builder.Default
    private long fingerprint = 0L;

    @Builder.Default
    private long frequency = 0L;

    @Builder.Default
    private Set<Long> positions = new LinkedHashSet<>();

    @Builder.Default
    private List<Integer> chars = new ArrayList<>();

    public boolean isBigram() {
        return left != null && right != null;
    }

    public void incrementFrequency(long value) {
        frequency += value;
    }

    @Override
    public String toString() {
        return null;
    }


    public static long fingerprintCat(long a, long c) {
        val b = 0xe08c1d668b756f82L;  // more of the golden ratio
        a -= b;
        a -= c;
        a ^= (c >> 43);
        b -= c;
        b -= a;
        b ^= (a << 9);
        c -= a;
        c -= b;
        c ^= (b >> 8);
        a -= b;
        a -= c;
        a ^= (c >> 38);
        b -= c;
        b -= a;
        b ^= (a << 23);
        c -= a;
        c -= b;
        c ^= (b >> 5);
        a -= b;
        a -= c;
        a ^= (c >> 35);
        b -= c;
        b -= a;
        b ^= (a << 49);
        c -= a;
        c -= b;
        c ^= (b >> 11);
        a -= b;
        a -= c;
        a ^= (c >> 12);
        b -= c;
        b -= a;
        b ^= (a << 18);
        c -= a;
        c -= b;
        c ^= (b >> 22);
        return c;
    }
}
