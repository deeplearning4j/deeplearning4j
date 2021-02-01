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

#ifndef LIBND4J_HESSENBERGANDSCHUR_H
#define LIBND4J_HESSENBERGANDSCHUR_H

#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

// this class implements Hessenberg decomposition of square matrix using orthogonal similarity transformation
// A = Q H Q^T
// Q - orthogonal matrix
// H - Hessenberg matrix
template <typename T>
class Hessenberg {
    // suppose we got input square NxN matrix

    public:

        NDArray _Q;     // {N,N}
        NDArray _H;     // {N,N}

        explicit Hessenberg(const NDArray& matrix);

    private:
        void evalData();
};


// this class implements real Schur decomposition of square matrix using orthogonal similarity transformation
// A = U T U^T
// T - real quasi-upper-triangular matrix - block upper triangular matrix where the blocks on the diagonal are 1×1 or 2×2 with complex eigenvalues
// U - real orthogonal matrix

template <typename T>
class Schur {
        // suppose we got input square NxN matrix

    public:

        NDArray _T;     // {N,N}
        NDArray _U;     // {N,N}

        explicit Schur(const NDArray& matrix);

        void splitTwoRows(const int ind, const T shift);

        void calcShift(const int ind, const int iter, T& shift, NDArray& shiftInfo);

        void initFrancisQR(const int ind1,  const int ind2, const NDArray& shiftVec, int& ind3, NDArray& householderVec);

        void doFrancisQR(const int ind1, const int ind2, const int ind3, const NDArray& householderVec);

        void calcFromHessenberg();

    private:

    	static const int _maxItersPerRow = 40;

        void evalData(const NDArray& matrix);

	    //////////////////////////////////////////////////////////////////////////
		FORCEINLINE int getSmallSubdiagEntry(const int inInd) {

			int outInd = inInd;
			while (outInd > 0) {
		    	T factor = math::nd4j_abs<T>(_T.t<T>(outInd-1, outInd-1)) + math::nd4j_abs<T>(_T.t<T>(outInd, outInd));
		    	if (math::nd4j_abs<T>(_T.t<T>(outInd, outInd-1)) <= DataTypeUtils::eps<T>() * factor)
		      		break;
				outInd--;
		  	}
			return outInd;
		}
};


}
}
}


#endif //LIBND4J_HESSENBERGANDSCHUR_H
