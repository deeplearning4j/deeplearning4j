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
// @author raver119@gmail.com, created on 6/30/2018
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_OMPLAUNCHHELPER_H
#define LIBND4J_OMPLAUNCHHELPER_H
#include <system/op_boilerplate.h>

#include <vector>

namespace sd {

class SD_LIB_EXPORT OmpLaunchHelper {
 public:
  OmpLaunchHelper() = delete;

  OmpLaunchHelper(const LongType N, float desiredNumThreads = -1);

  SD_INLINE LongType getThreadOffset(const int threadNum);
  SD_INLINE LongType getItersPerThread(const int threadNum);

  static LongType betterSpan(LongType N);
  static LongType betterSpan(LongType N, LongType numThreads);

  static int betterThreads(LongType N);
  static int betterThreads(LongType N, int maxThreads);

  static int tadThreads(LongType tadLength, LongType numTads);

  int _numThreads;
  unsigned int _itersPerThread;
  unsigned int _remainder;
};

////////////////////////////////////////////////////////////////////////////////
SD_INLINE LongType OmpLaunchHelper::getThreadOffset(const int threadNum) { return threadNum * _itersPerThread; }

////////////////////////////////////////////////////////////////////////////////
SD_INLINE LongType OmpLaunchHelper::getItersPerThread(const int threadNum) {
  return (threadNum == _numThreads - 1) ? _itersPerThread + _remainder
                                        : _itersPerThread;  // last thread may contain bigger number of iterations
}

}  // namespace sd

#endif  // LIBND4J_OMPLAUNCHHELPER_H
