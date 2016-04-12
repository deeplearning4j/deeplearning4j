package org.nd4j.linalg.api.shape.loop.one;

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
public  class RawArrayIterationInformation1 implements Serializable {
    private int nDim;
    private int aOffset = -1;
    private int[] aStrides;
    private int[] shape;
    private DataBuffer a;

    public RawArrayIterationInformation1 computeOut() {
        int aOffset = this.aOffset;
        int[] aStrides = ArrayUtil.copy(this.aStrides);
        int[] shape = ArrayUtil.copy(this.shape);
        int nDim = this.nDim;
        StridePermutation[] perms = Shape.createSortedStrides(aStrides);


        for(int i = 0; i < nDim; i++) {
            int iPerm = perms[nDim - i - 1].getPermutation();
            shape[i] = this.shape[iPerm];
            aStrides[i] = aStrides[iPerm];

        }

        for(int i = 0; i < nDim; i++) {
            int outStrideA = aStrides[i];
            int shapeI = shape[i];

            if(outStrideA < 0) {
                aOffset += outStrideA * shapeI - 1;
                aStrides[i] -= outStrideA;
            }
        }

        int i = 0;
        for(int j = 1; j < nDim; j++) {
            if(shape[i] == 1) {
                shape[i] = shape[j];
                aStrides[i] =  aStrides[j];

            }
            else if(shape[j] == 1) {
                //drops axis j
            }
            else if(aStrides[i] * shape[i] == aStrides[j]) {
                shape[i] *= shape[j];
            }

            else {
                i++;
                shape[i] = shape[j];
                aStrides[i] = aStrides[j];
            }


        }

        nDim = i + 1;

        return  RawArrayIterationInformation1.builder().aOffset(aOffset)
                .a(a).aStrides(aStrides)
                .shape(shape).nDim(nDim).build();
    }
}
