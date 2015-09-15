package org.nd4j.linalg.api.shape;

import com.google.common.primitives.Ints;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Metadata for striding
 * mainly meant for internal use
 *
 * @author Adam Gibson
 */
public @Data
@AllArgsConstructor
@NoArgsConstructor
class StridePermutation implements Comparable<StridePermutation> {
    private int permutation;
    private int stride;


    @Override
    public int compareTo(StridePermutation o) {
        int astride = this.stride,
                bstride = o.getStride();

    /* Sort the absolute value of the strides */
        if (astride < 0) {
            astride = -astride;
        }
        if (bstride < 0) {
            bstride = -bstride;
        }

        if (astride == bstride) {
        /*
         * Make the qsort stable by next comparing the perm order.
         * (Note that two perm entries will never be equal)
         */
            int aperm = permutation,
                    bperm = o.getPermutation();
            return (aperm < bperm) ? -1 : 1;
        }
        if (astride > bstride) {
            return -1;
        }

        return 1;
    }

    /**
     * Create an array of these from the given stride
     * @param stride the stride
     * @return the stride permutation array
     */
    public static StridePermutation[] create(int[] stride) {
        StridePermutation[] ret = new StridePermutation[stride.length];
        for(int i = 0; i < stride.length; i++) {
            ret[i] = new StridePermutation(i,stride[i]);
        }

        return ret;
    }

}
