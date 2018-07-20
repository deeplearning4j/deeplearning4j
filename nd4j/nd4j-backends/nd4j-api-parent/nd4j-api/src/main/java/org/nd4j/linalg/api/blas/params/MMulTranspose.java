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
    private boolean transposeA = false;
    private boolean transposeB = false;
    private boolean transposeResult = false;
    private INDArray a,b;
    private static MMulTranspose allFalse = MMulTranspose.builder().build();


    /**
     * Returns the default transpose
     * where all are false
     * @return
     */
    public static MMulTranspose allFalse() {
        return allFalse;
    }



    @Builder
    public MMulTranspose(boolean transposeA,
                         boolean transposeB,
                         boolean transposeResult,
                         INDArray a,
                         INDArray b) {
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.transposeResult = transposeResult;
        if(transposeResult) {
            this.transposeA = !transposeA;
            this.transposeB = !transposeB;
            this.a = b;
            this.b = a;
        }

        this.transposeA = transposeA;
        this.transposeB = transposeB;

        if(this.transposeA && a != null) {
            if (a.rank() == 2)
                this.a = a.transpose();
            if (a.rank() == 3)
                this.a = a.permute(0, 2, 1);
        }
        else
            this.a = a;
        if(this.transposeB && b != null) {
            if (b.rank() == 2)
                this.b = b.transpose();
            if (b.rank() == 3)
                this.b = b.permute(0, 2, 1);
        }
        else
            this.b = b;
    }
}
