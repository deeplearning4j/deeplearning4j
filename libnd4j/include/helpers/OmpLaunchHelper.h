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

//
// @author raver119@gmail.com, created on 6/30/2018
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_OMPLAUNCHHELPER_H
#define LIBND4J_OMPLAUNCHHELPER_H

#include <vector>
#include <pointercast.h>
#include <op_boilerplate.h>

namespace nd4j {

class ND4J_EXPORT OmpLaunchHelper {
	
    public:				
        
		OmpLaunchHelper() = delete;
        
        OmpLaunchHelper(const Nd4jLong N, float desiredNumThreads = -1);

        FORCEINLINE Nd4jLong getThreadOffset(const int threadNum);
        FORCEINLINE Nd4jLong getItersPerThread(const int threadNum);

        static Nd4jLong betterSpan(Nd4jLong N);
        static Nd4jLong betterSpan(Nd4jLong N, Nd4jLong numThreads);
        
        static int betterThreads(Nd4jLong N);
        static int betterThreads(Nd4jLong N, int maxThreads);

        static int tadThreads(Nd4jLong tadLength, Nd4jLong numTads);

        int _numThreads;
		unsigned int _itersPerThread;
        unsigned int _remainder;
};

////////////////////////////////////////////////////////////////////////////////
FORCEINLINE Nd4jLong OmpLaunchHelper::getThreadOffset(const int threadNum) {
	
		return threadNum * _itersPerThread;
}

////////////////////////////////////////////////////////////////////////////////
FORCEINLINE Nd4jLong OmpLaunchHelper::getItersPerThread(const int threadNum) {
	
	return (threadNum == _numThreads - 1) ? _itersPerThread + _remainder : _itersPerThread;		// last thread may contain bigger number of iterations    	 
}

}


#endif //LIBND4J_OMPLAUNCHHELPER_H
