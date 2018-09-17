/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.blas.params;

import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

@Getter
@EqualsAndHashCode
public class MMulTranspose implements Serializable {
    //    private INDArray a,b;
    private static MMulTranspose allFalse = MMulTranspose.builder().build();
    private boolean transposeA;
    private boolean transposeB;
    private boolean transposeResult;


    @Builder
    public MMulTranspose(boolean transposeA, boolean transposeB, boolean transposeResult) {
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.transposeResult = transposeResult;
//        if(transposeResult) {
//            //Here: relying on (a x b)T = bT x aT
//            //(a x bT) = b x aT
//            //etc
//            this.transposeA = !transposeB;
//            this.transposeB = !transposeA;
//            this.a = b;
//            this.b = a;
//        } else {
//            this.a = a;
//            this.b = b;
//            this.transposeA = transposeA;
//            this.transposeB = transposeB;
//        }
//
//        if(this.transposeA && this.a != null) {
//            if (this.a.rank() == 2)
//                this.a = this.a.transpose();
//            if (this.a.rank() == 3)
//                this.a = this.a.permute(0, 2, 1);
//        }
//
//        if(this.transposeB && this.b != null) {
//            if (this.b.rank() == 2)
//                this.b = this.b.transpose();
//            if (b.rank() == 3)
//                this.b = this.b.permute(0, 2, 1);
//        }
    }

    /**
     * Returns the default transpose
     * where all are false
     *
     * @return
     */
    public static MMulTranspose allFalse() {
        return allFalse;
    }

    /**
     * Execute the matrix multiplication: A x B
     * Note that if a or b have transposeA/B == true, then this is done internally.
     * Also, if transposeResult == true, then this is also done internally - i.e., the result array - if present -
     * should not be transposed beforehand.
     * @param a      A array
     * @param b      B array
     * @param result Result array (pre resultArrayTranspose if required). May be null.
     * @return Result array
     */
    public INDArray exec(INDArray a, INDArray b, INDArray result) {
        a = transposeIfReq(transposeA, a);
        b = transposeIfReq(transposeB, b);
        if(result == null) {
            INDArray ret = a.mmul(b);
            return transposeIfReq(transposeResult, ret);
        } else {

            if(!transposeResult){
                return a.mmuli(b, result);
            } else {
                return a.mmuli(b, result).transpose();
            }
        }
    }

    private static INDArray transposeIfReq(boolean transpose, INDArray x){
        if (transpose) {
            if (x.rank() == 2)
                return x.transpose();
            if (x.rank() == 3)
                return x.permute(0, 2, 1);
        }
        return x;
    }
}
