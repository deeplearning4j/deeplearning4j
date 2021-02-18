/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_EIGENVALSANDVECS_H
#define LIBND4J_EIGENVALSANDVECS_H

#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

// this class calculates eigenvalues and eigenvectors of given input matrix
template <typename T>
class EigenValsAndVecs {

    public:
        // suppose we got input square NxN matrix

        NDArray _Vals;      	// {N,2} matrix of eigenvalues, 2 means real and imaginary part
        NDArray _Vecs;      	// {N,N,2} matrix, whose columns are the eigenvectors (complex), 2 means real and imaginary part

        explicit EigenValsAndVecs(const NDArray& matrix);


		//////////////////////////////////////////////////////////////////////////
		FORCEINLINE static void divideComplexNums(const T& a1, const T& b1, const T& a2, const T& b2, T& a3, T& b3) {

    		T norm2 = a2*a2 + b2*b2;

    		a3 = (a1*a2 + b1*b2) / norm2;
    		b3 = (a2*b1 - a1*b2) / norm2;
		}

		//////////////////////////////////////////////////////////////////////////
		FORCEINLINE static void multiplyComplexNums(const T& a1, const T& b1, const T& a2, const T& b2, T& a3, T& b3) {

    		a3 = (a1*a2 - b1*b2);
    		b3 = (a1*b2 + b1*a2);
		}

		//////////////////////////////////////////////////////////////////////////
		FORCEINLINE static void sqrtComplexNum(T& a, T& b) {

			T norm = math::nd4j_sqrt<T,T>(a*a + b*b);

    		if(b < (T)0)
                b = -math::nd4j_sqrt<T,T>((T)0.5 * (norm - a));
            else
                b = math::nd4j_sqrt<T,T>((T)0.5 * (norm - a));
    		a = math::nd4j_sqrt<T,T>((T)0.5 * (norm + a));
		}


    private:

        void calcEigenVals(const NDArray& schurMatrixT);						// calculates _Vals
        void calcPseudoEigenVecs(NDArray& schurMatrixT, NDArray& schurMatrixU);	// makes changes both in schurMatrixT(NxN) and schurMatrixU(NxN), also calculates and stores pseudo-eigenvectors (real) in schurMatrixU columns
        void calcEigenVecs(const NDArray& schurMatrixU);						// calculates _Vecs

};


}
}
}


#endif //LIBND4J_EIGENVALSANDVECS_H
