/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Paul Dubs
//

#ifndef LIBND4J_ATTENTIONHELPER_H
#define LIBND4J_ATTENTIONHELPER_H



#include "array/NDArray.h"






namespace sd {
class SD_LIB_EXPORT AttentionHelper {
 public:
  static NDArray multiHeadProject(const NDArray * input, const NDArray * projectionMatrix,
                                  LaunchContext * context = LaunchContext ::defaultContext());
  static void multiHeadProjectBp(const NDArray * input, const NDArray * projectionMatrix, const NDArray * eps,
                                 NDArray * dLdInput, NDArray * dLdProjectionMatrix,
                                 LaunchContext * context = LaunchContext ::defaultContext());

  /**
   * @param shape
   * @return
   */
  static NDArray * lowerTriangularMask(std::vector<LongType> *shape);

  /**
   *
   * @param query
   * @param value
   * @return
   */
  static NDArray *computeCasualMask(NDArray *query, NDArray *value, bool multiHead);


  static NDArray * mergeMasks(NDArray *x, NDArray *y);

  /**
   * @param query
   * @param value
   * @param attentionMask
   * @param useCausalMask
   * @return
   */
  static NDArray *computeAttentionMask(NDArray *query, NDArray *value, NDArray *queryMask, NDArray *valueMask,
                                       NDArray *attentionMask, bool useCausalMask);


  /**
   *
   * @return
   */
  static void applyAttentionScores(NDArray *scores, NDArray *value, NDArray *scoresMask, double dropout,
                                   int randomSeed,
                                   NDArray *applyScoresOut, NDArray *attentionLogits, NDArray *dropoutMask);




  /**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
   */
  static void attentionHelper(NDArray *query, NDArray *key, double scale, NDArray *attentionLogits);

  /**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @param concatWeights
   * @return
   */
  static void attentionBpHelper(NDArray *query, NDArray *key, NDArray *values, double scale, NDArray *dLdq,
                                NDArray *dLdk, NDArray *dLdv, NDArray *eps,
                                LongType dropoutSeed, NDArray *qMask,
                                NDArray *vMask, bool useCausalMask,
                                double dropout, bool training, NDArray *attentionScoresOut,
                                NDArray *attentionScoresWeights,
                                NDArray *attentionScoresLogits,
                                NDArray *dropoutMask);



  /**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @param concatWeights
   * @return
   */
  static void additiveAttentionBpHelper(NDArray *query, NDArray *key, NDArray *values, double scale,
                                        NDArray *concatWeights, NDArray *dLdq, NDArray *dLdk, NDArray *dLdv,
                                        NDArray *eps, LongType dropoutSeed, NDArray *qMask, NDArray *vMask, bool useCausalMask, double dropout, bool training);

  /**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @param concatWeights
   * @return
   */
  static void dotProductAttentionBpHelper(NDArray *query, NDArray *key, NDArray *values, double scale, NDArray *dLdq,
                                          NDArray *dLdk, NDArray *dLdv, NDArray *eps,
                                          LongType dropoutSeed,
                                          NDArray *qMask, NDArray *vMask,
                                          bool useCausalMask, double dropout, bool training,
                                          NDArray *attentionScoresWeights, NDArray *attentionLogits,
                                          NDArray *dropoutMask);



  /**
   *
   * @param inputs
   * @param mask
   * @param training
   * @param returnAttentionScores
   * @param useCausalMask
   */
  static void doAttention(std::vector<NDArray *> &inputs, std::vector<NDArray *> &masks, bool training,
                          bool useCausalMask, double dropout, double scale, NDArray *attentionScores,
                          int dropoutSeed,
                          NDArray *applyScoresOut, NDArray *attentionLogits, NDArray *dropoutMask);



  /**
   *
   * @param inputs
   * @param mask
   * @param training
   * @param returnAttentionScores
   * @param useCausalMask
   */
  static void doAttentionBp(std::vector<NDArray *> &inputs, std::vector<NDArray *> &masks, bool training,
                            bool useCausalMask, double dropout, double scale, std::vector<NDArray *> outputs,
                            LongType dropoutSeed);


};
}  // namespace sd

#endif
