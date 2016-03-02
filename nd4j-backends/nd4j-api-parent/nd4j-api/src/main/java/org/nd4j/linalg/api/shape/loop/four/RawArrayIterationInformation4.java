package org.nd4j.linalg.api.shape.loop.four;

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
public  class RawArrayIterationInformation4 implements Serializable {
    private int nDim;
    private int aOffset = -1;
    private int bOffset = -1;
    private int cOffset = -1;
    private int dOffset = -1;
    private int[] aStrides;
    private int[] bStrides;
    private int[] cStrides;
    private int[] dStrides;
    private int[] shape;
    private DataBuffer a,b,c,d;

    public RawArrayIterationInformation4 computeOut() {
        int aOffset = this.aOffset;
        int bOffset = this.bOffset;
        int[] aStrides = ArrayUtil.copy(this.aStrides);
        int[] bStrides = ArrayUtil.copy(this.bStrides);
        int[] cStrides = ArrayUtil.copy(this.cStrides);
        int[] dStrides = ArrayUtil.copy(this.dStrides);
        int[] shape = ArrayUtil.copy(this.shape);
        int nDim = this.nDim;
        StridePermutation[] perms = Shape.createSortedStrides(aStrides);


        for(int i = 0; i < nDim; i++) {
            int iPerm = perms[nDim - i - 1].getPermutation();
            shape[i] = this.shape[iPerm];
            aStrides[i] = aStrides[iPerm];
            bStrides[i] = bStrides[iPerm];
            cStrides[i] = cStrides[iPerm];
            dStrides[i] = dStrides[iPerm];

        }

        for(int i = 0; i < nDim; i++) {
            int outStrideA = aStrides[i];
            int outStrideB = bStrides[i];
            int outStrideC = cStrides[i];
            int outStrideD = dStrides[i];

            int shapeI = shape[i];

            if(outStrideA < 0) {
                aOffset += outStrideA * shapeI - 1;
                bOffset += outStrideB * shapeI - 1;
                aStrides[i] -= outStrideA;
                bStrides[i] -= outStrideB;
                cStrides[i] -= outStrideC;
                dStrides[i] -= outStrideD;
            }
        }

        int i = 0;
        for(int j = 1; j < nDim; j++) {
            if(shape[i] == 1) {
                shape[i] = shape[j];
                aStrides[i] =  aStrides[j];
                bStrides[i] = bStrides[j];
                cStrides[i] = cStrides[j];
                dStrides[i] = dStrides[j];

            }
            else if(shape[j] == 1) {
                //drops axis j
            }
            else if(aStrides[i] * shape[i] == aStrides[j] && bStrides[i] * shape[i] == bStrides[j] && bStrides[i] * shape[i] == cStrides[j] && dStrides[i] * shape[i] == dStrides[j]) {
                shape[i] *= shape[j];
            }

            else {
                i++;
                shape[i] = shape[j];
                aStrides[i] = aStrides[j];
                bStrides[i] = bStrides[j];
                cStrides[i] = cStrides[j];
                dStrides[i] = dStrides[j];

            }


        }

        nDim = i + 1;

        return  RawArrayIterationInformation4.builder().aOffset(aOffset)
                .a(a).b(b).c(c).d(d)
                .bOffset(bOffset).aStrides(aStrides).bStrides(bStrides)
                .cStrides(cStrides).dStrides(dStrides)
                .shape(shape).nDim(nDim).build();
    }
}
