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
// @author Alex Black
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_gruCell)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/gru.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gruCell, 6, 4, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);  // input [bS, nIn], nIn - input size
  auto hLast =
      INPUT_VARIABLE(1);  // previous cell output [bS, nU],  that is at previous time step t-1, nU - number of units
  auto Wru = INPUT_VARIABLE(2);  // RU weights - [nIn+nU, 2*nU] - reset and update gates (input/recurrent weights)
  auto Wc = INPUT_VARIABLE(3);   // C weights - [nIn+nU, nU] - cell gate (input/recurrent weights)
  auto bru = INPUT_VARIABLE(4);  // reset and update biases, [2*nU] - reset and update gates
  auto bc = INPUT_VARIABLE(5);   // cell biases, [nU]

  auto r = OUTPUT_VARIABLE(0);  // Reset gate output [bS, nU]
  auto u = OUTPUT_VARIABLE(1);  // Update gate output [bS, nU]
  auto c = OUTPUT_VARIABLE(2);  // Cell gate output [bS, nU]
  auto h = OUTPUT_VARIABLE(3);  // current cell output [bS, nU]

  REQUIRE_TRUE(x->rankOf() == 2 && hLast->rankOf() == 2, 0,
               "gruCell: Input ranks must be 2 for inputs 0 and 1 (x, hLast) - got %i, %i", x->rankOf(),
               hLast->rankOf());

  const int rank = x->rankOf();
  const auto bS = x->sizeAt(0);
  const auto nIn = x->sizeAt(1);
  const auto nU = hLast->sizeAt(1);

  REQUIRE_TRUE(x->sizeAt(0) == hLast->sizeAt(0), 0,
               "gruCell: Input minibatch sizes (dimension 0) must be same for x and hLast");
  REQUIRE_TRUE(Wru->rankOf() == 2 && Wc->rankOf() == 2, 0,
               "gruCell: weight arrays (Wru, Wc) arrays must be 2, got %i and %i", Wru->rankOf(), Wc->rankOf());
  REQUIRE_TRUE(Wru->sizeAt(0) == (nIn + nU) && Wc->sizeAt(0) == (nIn + nU), 0,
               "gruCell: Weights size(0) must be equal to nIn + nU, got %i", Wru->sizeAt(0));
  REQUIRE_TRUE(Wru->sizeAt(1) == (2 * nU), 0,
               "gruCell: Weights (reset and update) size(1) must be equal to 2*nU, got %i", Wru->sizeAt(1));
  REQUIRE_TRUE(Wc->sizeAt(1) == nU, 0, "gruCell: Weights (cell) size(1) must be equal to nU, got %i", Wc->sizeAt(1));
  REQUIRE_TRUE(bru->rankOf() == 1 && bru->sizeAt(0) == (2 * nU), 0,
               "gruCell: reset/update biases must be rank 1, size 2*nU");
  REQUIRE_TRUE(bc->rankOf() == 1 && bc->sizeAt(0) == nU, 0, "gruCell: cell biases must be rank 1, size nU");
  REQUIRE_TRUE(r->rankOf() == 2 && u->rankOf() == 2 && c->rankOf() == 2 && h->rankOf() == 2 && r->sizeAt(0) == bS &&
                   u->sizeAt(0) == bS && c->sizeAt(0) == bS && h->sizeAt(0) == bS && r->sizeAt(1) == nU &&
                   u->sizeAt(1) == nU && c->sizeAt(1) == nU && h->sizeAt(1) == nU,
               0, "gruCell: Output arrays must all be rank 2 with size(0) == batchSize and size(1) == nU");

  helpers::gruCell(block.launchContext(), x, hLast, Wru, Wc, bru, bc, r, u, c, h);

  return Status::OK;
}

DECLARE_TYPES(gruCell) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedInputTypes(4, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(gruCell) {
  auto x = inputShape->at(0);      // input [bS x nIn]
  auto hLast = inputShape->at(1);  // previous cell output [bS x nU],  that is at previous time step t-1
  auto Wru = inputShape->at(2);    // RU weights - [(nIn+nU), 2*nU] - reset and update gates (input/recurrent weights)
  auto Wc = inputShape->at(3);     // C weights - [(nIn+nU), nU] - cell gate (input/recurrent weights)
  auto bru = inputShape->at(4);    // reset and update biases, [2*nU] - reset and update gates
  auto bc = inputShape->at(5);     // cell biases, [nU]

  REQUIRE_TRUE(shape::rank(x) == 2 && shape::rank(hLast) == 2, 0,
               "gruCell: Input ranks must be 2 for inputs 0 and 1 (x, hLast) - got %i, %i", shape::rank(x),
               shape::rank(hLast));

  const int rank = x[0];
  const auto bS = x[1];
  const auto nIn = x[2];
  const auto nU = hLast[2];

  REQUIRE_TRUE(x[1] == hLast[1], 0, "gruCell: Input minibatch sizes (dimension 0) must be same for x and hLast");
  REQUIRE_TRUE(shape::rank(Wru) == 2 && shape::rank(Wc) == 2, 0,
               "gruCell: weight arrays (Wru, Wc) arrays must be 2, got %i and %i", shape::rank(Wru), shape::rank(Wc));
  REQUIRE_TRUE(Wru[1] == (nIn + nU) && Wc[1] == (nIn + nU), 0,
               "gruCell: Weights size(0) must be equal to nIn + nU, got %i and %i", Wru[1], Wc[1]);
  REQUIRE_TRUE(Wru[2] == (2 * nU), 0, "gruCell: Weights (reset and update) size(1) must be equal to 2*nU, got %i",
               Wru[2]);
  REQUIRE_TRUE(Wc[2] == nU, 0, "gruCell: Weights (cell) size(1) must be equal to nU, got %i", Wc[2]);
  REQUIRE_TRUE(shape::rank(bru) == 1 && bru[1] == (2 * nU), 0,
               "gruCell: reset/update biases must be rank 1, size 2*nU");
  REQUIRE_TRUE(shape::rank(bc) == 1 && bc[1] == nU, 0, "gruCell: cell biases must be rank 1, size nU");

  LongType *s0(nullptr);
  ALLOCATE(s0, block.getWorkspace(), shape::shapeInfoLength(rank), sd::LongType);  // [bS x nU]

  s0[0] = rank;
  s0[1] = bS;
  s0[2] = nU;

  ShapeUtils::updateStridesAndType(s0, x, shape::order(hLast));
  auto ts0 = ConstantShapeHelper::getInstance().createFromExisting(s0);

  // 4 output shapes, all [bs, nU]
  return SHAPELIST(ts0, ts0, ts0, ts0);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gruCell_bp, 10, 6, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);     // input [bS x iS]
  auto hi = INPUT_VARIABLE(1);    // previous cell output [bS x nU]
  auto W = INPUT_VARIABLE(2);     // weights, [iS+nU x 2*nU]
  auto Wc = INPUT_VARIABLE(3);    // c weights, [iS+nU x nU]
  auto b = INPUT_VARIABLE(4);     // biases, [2*nU]
  auto bc = INPUT_VARIABLE(5);    // biases, [nU]
  auto dLdr = INPUT_VARIABLE(6);  // gradient wrt reset gate, [bS, nU]
  auto dLdu = INPUT_VARIABLE(7);  // gradient wrt update gate, [bS, nU]
  auto dLdc = INPUT_VARIABLE(8);  // gradient wrt cell state, [bS, nU]
  auto dLdh = INPUT_VARIABLE(9);  // gradient wrt current cell output, [bS, nU]

  auto dLdx = OUTPUT_VARIABLE(0);   // gradient wrt x,  [bS, iS]
  auto dLdhi = OUTPUT_VARIABLE(1);  // gradient wrt hi, [bS, nU]
  auto dLdW = OUTPUT_VARIABLE(2);   // gradient wrt W,  [iS+nU x 2*nU]
  auto dLdWc = OUTPUT_VARIABLE(3);  // gradient wrt Wc, [iS+nU x nU]
  auto dLdb = OUTPUT_VARIABLE(4);   // gradient wrt biases, [2*nU]
  auto dLdbc = OUTPUT_VARIABLE(5);  // gradient wrt c biases, [nU]

  const LongType bS = x->sizeAt(0);
  const LongType iS = x->sizeAt(1);
  const LongType nU = hi->sizeAt(1);

  REQUIRE_TRUE(x->rankOf() == 2, 0, "GRU_CELL_BP: rank of input array x must be 2, but got %i instead", x->rankOf());

  const std::vector<LongType> hiCorrectShape = {bS, nU};
  const std::vector<LongType> wCorrectShape = {iS + nU, 2 * nU};
  const std::vector<LongType> wcCorrectShape = {iS + nU, nU};
  const std::vector<LongType> bCorrectShape = {2 * nU};
  const std::vector<LongType> bcCorrectShape = {nU};

  REQUIRE_TRUE(hi->isSameShape(hiCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of previous cell output array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(hiCorrectShape).c_str(), ShapeUtils::shapeAsString(hi).c_str());
  REQUIRE_TRUE(W->isSameShape(wCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(wCorrectShape).c_str(), ShapeUtils::shapeAsString(W).c_str());
  REQUIRE_TRUE(Wc->isSameShape(wcCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of c weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(wcCorrectShape).c_str(), ShapeUtils::shapeAsString(Wc).c_str());
  REQUIRE_TRUE(b->isSameShape(bCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of biases array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(b).c_str());
  REQUIRE_TRUE(bc->isSameShape(bcCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of c biases array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(bcCorrectShape).c_str(), ShapeUtils::shapeAsString(bc).c_str());
  REQUIRE_TRUE(
      dLdr->isSameShape(hiCorrectShape), 0,
      "GRU_CELL_BP op: wrong shape of dLdr array (gradient wrt reset gate), expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(hiCorrectShape).c_str(), ShapeUtils::shapeAsString(dLdr).c_str());
  REQUIRE_TRUE(
      dLdu->isSameShape(hiCorrectShape), 0,
      "GRU_CELL_BP op: wrong shape of dLdu array (gradient wrt update gate), expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(hiCorrectShape).c_str(), ShapeUtils::shapeAsString(dLdu).c_str());
  REQUIRE_TRUE(
      dLdc->isSameShape(hiCorrectShape), 0,
      "GRU_CELL_BP op: wrong shape of dLdc array (gradient wrt cell state), expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(hiCorrectShape).c_str(), ShapeUtils::shapeAsString(dLdc).c_str());
  REQUIRE_TRUE(dLdh->isSameShape(hiCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of dLdh array (gradient wrt current cell output), expected is %s, but got "
               "%s instead !",
               ShapeUtils::shapeAsString(hiCorrectShape).c_str(), ShapeUtils::shapeAsString(dLdh).c_str());

  helpers::gruCellBp(block.launchContext(), x, hi, W, Wc, b, bc, dLdr, dLdu, dLdc, dLdh, dLdx, dLdhi, dLdW, dLdWc, dLdb,
                     dLdbc);

  return Status::OK;
}

DECLARE_TYPES(gruCell_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedInputTypes(4, {ALL_FLOATS})
      ->setAllowedInputTypes(5, {ALL_FLOATS})
      ->setAllowedInputTypes(6, {ALL_FLOATS})
      ->setAllowedInputTypes(7, {ALL_FLOATS})
      ->setAllowedInputTypes(8, {ALL_FLOATS})
      ->setAllowedInputTypes(9, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(gruCell_bp) {
  auto xShapeInfo = inputShape->at(0);     // [bS x iS]
  auto hiShapeInfo = inputShape->at(1);    // [bS x nU]
  auto wShapeInfo = inputShape->at(2);     // [iS+nU x 2*nU]
  auto wcShapeInfo = inputShape->at(3);    // [iS+nU x nU]
  auto bShapeInfo = inputShape->at(4);     // [2*nU]
  auto bcShapeInfo = inputShape->at(5);    // [nU]
  auto dLdrShapeInfo = inputShape->at(6);  // [bS, nU]
  auto dLduShapeInfo = inputShape->at(7);  // [bS, nU]
  auto dLdcShapeInfo = inputShape->at(8);  // [bS, nU]
  auto dLdhShapeInfo = inputShape->at(9);  // [bS, nU]

  const int rank = xShapeInfo[0];  // = 2
  const LongType bS = xShapeInfo[1];
  const LongType iS = xShapeInfo[2];
  const LongType nU = hiShapeInfo[2];

  REQUIRE_TRUE(xShapeInfo[0] == 2, 0, "GRU_CELL_BP: rank of input array x must be 2, but got %i instead",
               xShapeInfo[0]);

  const std::vector<LongType> hiCorrectShape = {bS, nU};
  const std::vector<LongType> wCorrectShape = {iS + nU, 2 * nU};
  const std::vector<LongType> wcCorrectShape = {iS + nU, nU};
  const std::vector<LongType> bCorrectShape = {2 * nU};
  const std::vector<LongType> bcCorrectShape = {nU};

  REQUIRE_TRUE(ShapeUtils::areShapesEqual(hiShapeInfo, hiCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of previous cell output array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(hiCorrectShape).c_str(), ShapeUtils::shapeAsString(hiShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(wShapeInfo, wCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(wCorrectShape).c_str(), ShapeUtils::shapeAsString(wShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(wcShapeInfo, wcCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of c weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(wcCorrectShape).c_str(), ShapeUtils::shapeAsString(wcShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(bShapeInfo, bCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of biases array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(bShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(bcShapeInfo, bcCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of c biases array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(bcCorrectShape).c_str(), ShapeUtils::shapeAsString(bcShapeInfo).c_str());
  REQUIRE_TRUE(
      ShapeUtils::areShapesEqual(dLdrShapeInfo, hiCorrectShape), 0,
      "GRU_CELL_BP op: wrong shape of dLdr array (gradient wrt reset gate), expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(hiCorrectShape).c_str(), ShapeUtils::shapeAsString(dLdrShapeInfo).c_str());
  REQUIRE_TRUE(
      ShapeUtils::areShapesEqual(dLduShapeInfo, hiCorrectShape), 0,
      "GRU_CELL_BP op: wrong shape of dLdu array (gradient wrt update gate), expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(hiCorrectShape).c_str(), ShapeUtils::shapeAsString(dLduShapeInfo).c_str());
  REQUIRE_TRUE(
      ShapeUtils::areShapesEqual(dLdcShapeInfo, hiCorrectShape), 0,
      "GRU_CELL_BP op: wrong shape of dLdc array (gradient wrt cell state), expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(hiCorrectShape).c_str(), ShapeUtils::shapeAsString(dLdcShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(dLdhShapeInfo, hiCorrectShape), 0,
               "GRU_CELL_BP op: wrong shape of dLdh array (gradient wrt current cell output), expected is %s, but got "
               "%s instead !",
               ShapeUtils::shapeAsString(hiCorrectShape).c_str(), ShapeUtils::shapeAsString(dLdhShapeInfo).c_str());
  return SHAPELIST(CONSTANT(xShapeInfo), CONSTANT(hiShapeInfo), CONSTANT(wShapeInfo), CONSTANT(wcShapeInfo),
                   CONSTANT(bShapeInfo), CONSTANT(bcShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
