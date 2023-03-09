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

#define ATTENTION_TYPE_DOT_PRODUCT 0
#define ATTENTION_TYPE_ADDITIVE 1

#define ATTENTION_SCORE_MODE_DOT 0
#define ATTENTION_SCORE_MODE_CONCAT 1


namespace sd {
class SD_LIB_EXPORT AttentionHelper {
 public:
  static sd::NDArray multiHeadProject(const sd::NDArray* input, const sd::NDArray* projectionMatrix,
                                      sd::LaunchContext* context = sd::LaunchContext ::defaultContext());
  static void multiHeadProjectBp(const sd::NDArray* input, const sd::NDArray* projectionMatrix, const sd::NDArray* eps,
                                 sd::NDArray* dLdInput, sd::NDArray* dLdProjectionMatrix,
                                 sd::LaunchContext* context = sd::LaunchContext ::defaultContext());

  /**
   * @param shape
   * @return
   */
  static sd::NDArray * lowerTriangularMask(std::vector<sd::LongType> *shape);

  /**
   *
   * @param query
   * @param value
   * @return
   */
  static NDArray *computeCasualMask(sd::NDArray *query, sd::NDArray *value, bool multiHead);


  static sd::NDArray * mergeMasks(sd::NDArray *x,sd::NDArray *y);

  /**
   * @param query
   * @param value
   * @param attentionMask
   * @param useCausalMask
   * @return
   */
  static NDArray *computeAttentionMask(sd::NDArray *query, sd::NDArray *value, sd::NDArray *queryMask,
                                       sd::NDArray *valueMask, sd::NDArray *attentionMask, bool useCausalMask);


  /**
   *
   * @return
   */
  static void applyAttentionScores(sd::NDArray *scores, sd::NDArray *value, sd::NDArray *scoresMask, double dropout,
                                   int randomSeed, sd::NDArray *applyScoresOut);




  /**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
   */
  static void attentionHelper(sd::NDArray *query, sd::NDArray *key, int scoreMode, double scale,
                                     sd::NDArray *concatWeights, sd::NDArray *attentionScoresOut);

  /**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @param concatWeights
   * @return
   */
  static void attentionBpHelper(sd::NDArray *query, sd::NDArray *key, sd::NDArray *values, double scale,
                                sd::NDArray *concatWeights, int scoreMode, sd::NDArray *dLdq, sd::NDArray *dLdk,
                                sd::NDArray *dLdv, sd::NDArray *eps, LongType dropoutSeed, sd::NDArray *qMask,
                                sd::NDArray *vMask, bool useCausalMask, double dropout, bool training);



  /**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @param concatWeights
   * @return
   */
  static void additiveAttentionBpHelper(sd::NDArray *query, sd::NDArray *key, sd::NDArray *values, double scale,
                                        sd::NDArray *concatWeights, sd::NDArray *dLdq, sd::NDArray *dLdk,
                                        sd::NDArray *dLdv, sd::NDArray *eps, LongType dropoutSeed, sd::NDArray *qMask,
                                        sd::NDArray *vMask, bool useCausalMask, double dropout, bool training);

  /**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @param concatWeights
   * @return
   */
  static void dotProductAttentionBpHelper(sd::NDArray *query, sd::NDArray *key, sd::NDArray *values, double scale,
                                          sd::NDArray *dLdq, sd::NDArray *dLdk, sd::NDArray *dLdv, sd::NDArray *eps,
                                          LongType dropoutSeed, sd::NDArray *qMask, sd::NDArray *vMask,
                                          bool useCausalMask, double dropout, bool training);



  /**
   *
   * @param inputs
   * @param mask
   * @param training
   * @param returnAttentionScores
   * @param useCausalMask
   */
  static void doAttention(std::vector<NDArray *> &inputs, std::vector<sd::NDArray *> &masks, bool training,
                          bool returnAttentionScores, bool useCausalMask, double dropout, int attentionType,
                          double scale, sd::NDArray *attentionScores, int dropoutSeed, sd::NDArray *applyScoresOut);



  /**
   *
   * @param inputs
   * @param mask
   * @param training
   * @param returnAttentionScores
   * @param useCausalMask
   */
  static void doAttentionBp(std::vector<NDArray *> &inputs, std::vector<sd::NDArray *> &masks, bool training,
                            bool returnAttentionScores, bool useCausalMask, double dropout, int attentionType,
                            double scale, std::vector<NDArray *> outputs,
                            LongType dropoutSeed = 0);


};
}  // namespace sd

#endif
