package org.nd4j.linalg.api.shape;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

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
     * A port of numpy's multiple strid permutation
     * resolution algorithm
     * @param ndim the number of dimensions
     * @param arrays the arrays to create the strides for
     * @return the stdes for this set of arrays
     */
    public static int[] create(int ndim,INDArray[] arrays) {
        int i0, i1, ipos, ax_j0, ax_j1, iarrays;
        int narrays = arrays.length;
        int[] out_strideperm = new int[ndim];
    /* Initialize the strideperm values to the identity. */
        for (i0 = 0; i0 < ndim; ++i0) {
            out_strideperm[i0] = i0;
        }

    /*
     * This is the same as the custom stable insertion sort in
     * the NpyIter object, but sorting in the reverse order as
     * in the iterator. The iterator sorts from smallest stride
     * to biggest stride (Fortran order), whereas here we sort
     * from biggest stride to smallest stride (C order).
     */
        for (i0 = 1; i0 < ndim; ++i0) {

            ipos = i0;
            ax_j0 = out_strideperm[i0];

            for (i1 = i0 - 1; i1 >= 0; --i1) {
                boolean ambig = true, shouldswap = false;

                ax_j1 = out_strideperm[i1];

                for (iarrays = 0; iarrays < narrays; ++iarrays) {
                    if (arrays[iarrays].size(ax_j0) != 1 &&
                            arrays[iarrays].size(ax_j1) != 1) {
                        if (Math.abs(arrays[iarrays].stride(ax_j0)) <=
                                Math.abs(arrays[iarrays].stride(ax_j1))) {
                        /*
                         * Set swap even if it's not ambiguous already,
                         * because in the case of conflicts between
                         * different operands, C-order wins.
                         */
                            shouldswap = false;
                        }
                        else {
                        /* Only set swap if it's still ambiguous */
                            if (ambig) {
                                shouldswap = true;
                            }
                        }

                    /*
                     * A comparison has been done, so it's
                     * no longer ambiguous
                     */
                        ambig = false;
                    }
                }
            /*
             * If the comparison was unambiguous, either shift
             * 'ipos' to 'i1' or stop looking for an insertion point
             */
                if (!ambig) {
                    if (shouldswap) {
                        ipos = i1;
                    }
                    else {
                        break;
                    }
                }
            }

        /* Insert out_strideperm[i0] into the right place */
            if (ipos != i0) {
                for (i1 = i0; i1 > ipos; --i1) {
                    out_strideperm[i1] = out_strideperm[i1-1];
                }
                out_strideperm[ipos] = ax_j0;
            }
        }

        return out_strideperm;
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
