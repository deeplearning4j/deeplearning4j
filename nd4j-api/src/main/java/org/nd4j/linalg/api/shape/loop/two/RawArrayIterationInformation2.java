package org.nd4j.linalg.api.shape.loop.two;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.StridePermutation;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.Serializable;

/**
 * Raw array iteration information
 * Used in preparing
 * for linear array raw
 * iteration
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public  class RawArrayIterationInformation2 implements Serializable {
    private int nDim;
    private int aOffset = -1;
    private int bOffset = -1;
    private int[] aStrides;
    private int[] bStrides;
    private int[] shape;
    private DataBuffer a;
    private DataBuffer b;

    /**
     * Resolve the new
     * strides/shapes
     * @return
     */
    public RawArrayIterationInformation2 computeOut() {
        int aOffset = this.aOffset;
        int bOffset = this.bOffset;
        int[] aStrides = ArrayUtil.copy(this.aStrides);
        int[] bStrides = ArrayUtil.copy(this.bStrides);
        int[] shape = ArrayUtil.copy(this.shape);
        int nDim = this.nDim;
        StridePermutation[] perms = Shape.createSortedStrides(aStrides);


        for(int i = 0; i < nDim; i++) {
            int iPerm = perms[nDim - i - 1].getPermutation();
            shape[i] = this.shape[iPerm];
            aStrides[i] = aStrides[iPerm];
            bStrides[i] = bStrides[iPerm];
        }

        for(int i = 0; i < nDim; i++) {
            int outStrideA = aStrides[i];
            int outStrideB = bStrides[i];
            int shapeI = shape[i];

            if(outStrideA < 0) {
                aOffset += outStrideA * shapeI - 1;
                bOffset += outStrideB * shapeI - 1;
                aStrides[i] -= outStrideA;
                bStrides[i] -= outStrideB;
            }
        }

        int i = 0;
        for(int j = 1; j < nDim; j++) {
            if(shape[i] == 1) {
                shape[i] = shape[j];
                aStrides[i] =  aStrides[j];
                bStrides[i] = aStrides[j];
            }
            else if(shape[j] == 1) {
                //drops axis j
            }
            else if(aStrides[i] * shape[i] == aStrides[j] && bStrides[i] * shape[i] == bStrides[j]) {
                shape[i] *= shape[j];
            }

            else {
                i++;
                shape[i] = shape[j];
                aStrides[i] = aStrides[j];
                bStrides[i] = bStrides[j];
            }


        }

        nDim = i + 1;
        //need to force vectors and scalars to be 2d
        if(nDim == 1) {
            nDim = 2;
            shape = this.shape;
            //reset
            aStrides = this.aStrides;
            bStrides = this.bStrides;
        }


        return  RawArrayIterationInformation2.builder().aOffset(aOffset).a(a)
                .b(b)
                .bOffset(bOffset).aStrides(aStrides).bStrides(bStrides)
                .shape(shape).nDim(nDim).build();
    }
}
