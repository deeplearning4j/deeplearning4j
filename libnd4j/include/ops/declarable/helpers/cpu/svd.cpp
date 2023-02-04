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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 03.01.2018
//
#include <array/NDArrayFactory.h>
#include <helpers/biDiagonalUp.h>
#include <helpers/jacobiSVD.h>
#include <helpers/svd.h>
#if NOT_EXCLUDED(OP_svd)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// svd operation, this function is not method of SVD class, it is standalone function
template <typename T>
static void svd_(const NDArray* x, const std::vector<NDArray*>& outArrs, const bool fullUV, const bool calcUV,
                 const int switchNum) {
  sd_printf("In svd helper\n",0);
  auto s = outArrs[0];
  auto u = outArrs[1];
  auto v = outArrs[2];
  sd_printf("After s,u,v\n",0);
  const int rank = x->rankOf();
  const int sRank = rank - 1;

  sd_printf("before listX\n",0);
  auto listX = x->allTensorsAlongDimension({rank - 2, rank - 1});
  sd_printf("before listS\n",0);
  auto listS = s->allTensorsAlongDimension({sRank - 1});
  ResultSet *listU(nullptr), *listV(nullptr);

  if (calcUV) {
    sd_printf("in calcUV\n",0);
    listU = new ResultSet(u->allTensorsAlongDimension({rank - 2, rank - 1}));
    listV = new ResultSet(v->allTensorsAlongDimension({rank - 2, rank - 1}));
    sd_printf("after calcUV\n",0);
  }

  for (int i = 0; i < listX.size(); ++i) {
    sd_printf("calculating svd at %d\n",i);
    helpers::SVD<T> svdObj(*(listX.at(i)), switchNum, calcUV, calcUV, fullUV);
    sd_printf("Before assign in calculating svd\n",0);
    listS.at(i)->assign(svdObj._s);

    if (calcUV) {
      sd_printf("before assign in calcUV %d\n",i);
      listU->at(i)->assign(svdObj._u);
      listV->at(i)->assign(svdObj._v);
      sd_printf("after assign in calcUV %d\n",i);
    }
  }

  if (calcUV) {
    delete listU;
    delete listV;
  }
}

//////////////////////////////////////////////////////////////////////////
void svd(sd::LaunchContext* context, const NDArray* x, const std::vector<NDArray*>& outArrs, const bool fullUV,
         const bool calcUV, const int switchNum) {
  BUILD_SINGLE_SELECTOR(x->dataType(), svd_, (x, outArrs, fullUV, calcUV, switchNum), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif