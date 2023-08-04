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
// implementation of operations for Simple Recurrent Unit: arXiv:1709.02755v2 [cs.CL] 12 Sep 2017
//
//@author Yurii Shyrma
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_sru)

#include <helpers/MmulHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/sru.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru, 5, 2, false, 0, 0) {
  auto x = INPUT_VARIABLE(0);  // X, input 3d tensor [bS x inSize x time], time - number of time steps, bS - batch size,
  // inSize - number of features
  auto w = INPUT_VARIABLE(1);  // W, 2d tensor of weights [3*inSize x inSize]
  auto b = INPUT_VARIABLE(2);  // B, row of biases with twice length [2*inSize]
  auto c0 = INPUT_VARIABLE(3);  // C_{0}, 2d tensor of initial state [bS x inSize] at time t=0
  auto mask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;  // optional,  2d tensor of dropout mask [bS x inSize]

  auto h = OUTPUT_VARIABLE(0);  // cell outputs, [bS x inSize x time]
  auto c = OUTPUT_VARIABLE(1);  // cell states,  [bS x inSize x time]

  const int rank = x->rankOf();  // = 3
  const auto bS = x->sizeAt(0);
  const auto inSize = x->sizeAt(1);
  const auto time = x->sizeAt(2);

  // input shapes validation
  REQUIRE_TRUE(w->rankOf() == rank - 1, 0,
               "SRU operation: wrong rank of weights array, expected is %i, but got %i instead !", rank - 1,
               w->rankOf());
  REQUIRE_TRUE(b->rankOf() == 1, 0, "SRU operation: wrong rank of biases  array, expected is %i, but got %i instead !",
               1, b->rankOf());
  REQUIRE_TRUE(c0->rankOf() == rank - 1, 0,
               "SRU operation: wrong rank of initial state array, expected is %i, but got %i instead !", rank - 1,
               c0->rankOf());
  if (mask)
  REQUIRE_TRUE(mask->rankOf() == rank - 1, 0,
               "SRU operation: wrong rank of mask array, expected is %i, but got %i instead !", rank - 1,
               mask->rankOf());

  const std::vector<sd::LongType> wCorrectShape = {3 * inSize, inSize};
  const std::vector<sd::LongType> bCorrectShape = {2 * inSize};
  const std::vector<sd::LongType> c0CorrectShape = {bS, inSize};

  REQUIRE_TRUE(w->isSameShape(wCorrectShape), 0,
               "SRU operation: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(wCorrectShape).c_str(), ShapeUtils::shapeAsString(w).c_str());
  REQUIRE_TRUE(b->isSameShape(bCorrectShape), 0,
               "SRU operation: wrong shape of biases  array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(b).c_str());
  REQUIRE_TRUE(c0->isSameShape(c0CorrectShape), 0,
               "SRU operation: wrong shape of initial state array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(c0).c_str());
  if (mask)
  REQUIRE_TRUE(mask->isSameShape(c0CorrectShape), 0,
               "SRU operation: wrong shape of mask array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(mask).c_str());

  //  xm = x * mask
  auto xm = x;
  if (mask) {
    xm = new NDArray(x->shapeInfo(), true, block.launchContext());
    std::vector<sd::LongType> dims = {0, 1};
    x->applyBroadcast(broadcast::Multiply,&dims , *mask, *xm);
  }

  // time loop
  helpers::sruTimeLoop(block.launchContext(), xm, c0, w, b, h, c);

  if (mask) delete xm;

  return sd::Status::OK;
}

DECLARE_TYPES(sru) { getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS}); }

DECLARE_SHAPE_FN(sru) {
  auto xShapeInfo = inputShape->at(0);   // X, input 3d tensor [bS x inSize x time], time - number of time steps, bS -
  // batch size, inSize - number of features
  auto wShapeInfo = inputShape->at(1);   // W, 2d tensor of weights [3*inSize x inSize]
  auto bShapeInfo = inputShape->at(2);   // B, row of biases with twice length [2*inSize]
  auto c0ShapeInfo = inputShape->at(3);  // C_{0}, 2d tensor of initial state [bS x inSize] at time t=0
  auto maskShapeInfo =
      block.width() > 4 ? inputShape->at(4) : nullptr;  // optional,  2d tensor of dropout mask [bS x inSize]

  const int rank = xShapeInfo[0];  // = 3
  const int bS = xShapeInfo[1];
  const int inSize = xShapeInfo[2];
  const int time = xShapeInfo[3];

  // input shapes validation
  REQUIRE_TRUE(wShapeInfo[0] == rank - 1, 0,
               "SRU operation: wrong rank of weights array, expected is %i, but got %i instead !", rank - 1,
               wShapeInfo[0]);
  REQUIRE_TRUE(bShapeInfo[0] == 1, 0,
               "SRU operation: wrong rank of biases  array, expected is %i, but got %i instead !", 1, bShapeInfo[0]);
  REQUIRE_TRUE(c0ShapeInfo[0] == rank - 1, 0,
               "SRU operation: wrong rank of initial state array, expected is %i, but got %i instead !", rank - 1,
               c0ShapeInfo[0]);
  if (maskShapeInfo)
  REQUIRE_TRUE(maskShapeInfo[0] == rank - 1, 0,
               "SRU operation: wrong rank of mask array, expected is %i, but got %i instead !", rank - 1,
               maskShapeInfo[0]);

  const std::vector<sd::LongType> wCorrectShape = {3 * inSize, inSize};
  const std::vector<sd::LongType> bCorrectShape = {2 * inSize};
  const std::vector<sd::LongType> c0CorrectShape = {bS, inSize};

  REQUIRE_TRUE(ShapeUtils::areShapesEqual(wShapeInfo, wCorrectShape), 0,
               "SRU operation: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(wCorrectShape).c_str(), ShapeUtils::shapeAsString(wShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(bShapeInfo, bCorrectShape), 0,
               "SRU operation: wrong shape of biases  array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(bShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(c0ShapeInfo, c0CorrectShape), 0,
               "SRU operation: wrong shape of initial state array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(c0ShapeInfo).c_str());
  if (maskShapeInfo)
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(maskShapeInfo, c0CorrectShape), 0,
               "SRU operation: wrong shape of mask array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(maskShapeInfo).c_str());

  sd::LongType* newShapeInfo1 = nullptr;
  ALLOCATE(newShapeInfo1, block.getWorkspace(), shape::shapeInfoLength(rank), sd::LongType);  // [bS x inSize x time]

  newShapeInfo1[0] = rank;
  newShapeInfo1[1] = bS;
  newShapeInfo1[2] = inSize;
  newShapeInfo1[3] = time;

  ShapeUtils::updateStridesAndType(newShapeInfo1, xShapeInfo, shape::order(xShapeInfo));
  ShapeDescriptor *descriptor = new ShapeDescriptor(newShapeInfo1);
  RELEASE(newShapeInfo1, block.getWorkspace());
  auto result = ConstantShapeHelper::getInstance().createShapeInfo(descriptor);
  delete descriptor;
  return SHAPELIST(result, result);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_bp, 8, 4, true, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  // X, input 3d tensor [bS x K x N], N - number of time steps, bS - batch size, K - number of features
  auto w = INPUT_VARIABLE(1);         // W, 2d tensor of weights [3K x K]
  auto b = INPUT_VARIABLE(2);         // B, row of biases with twice length [1 x 2*K]
  auto c0 = INPUT_VARIABLE(3);        // C_{0}, 2d tensor of initial state [bS x K] at time t=0
  auto c = INPUT_VARIABLE(4);         // C, [bS x K x N]
  auto inGradCt = INPUT_VARIABLE(5);  // [bS x K]
  auto inGradH = INPUT_VARIABLE(6);   // [bS x K x N]
  NDArray* mask = nullptr;            // optional,  2d tensor of dropout mask [bS x K]

  bool applyMask = false;
  if (block.width() > 7) {
    mask = INPUT_VARIABLE(7);
    applyMask = true;
  }

  auto gradX = OUTPUT_VARIABLE(0);     // [bS x K x N]
  auto gradW = OUTPUT_VARIABLE(1);     // [bS x 3K x K]
  auto gradB = OUTPUT_VARIABLE(2);     // [1 x 2K]
  auto gradInit = OUTPUT_VARIABLE(3);  // [bS x K]

  const int bS = x->shapeOf()[0];
  const int K = x->shapeOf()[1];
  const int N = x->shapeOf()[2];  // N - number of time steps

  auto gradBias = NDArrayFactory::create_(x->ordering(), {bS, 2 * K, N}, gradX->dataType(), block.launchContext());
  auto gradU = NDArrayFactory::create_(x->ordering(), {bS, 3 * K, N}, gradX->dataType(), block.launchContext());
  auto gradHX = NDArrayFactory::create_(x->ordering(), {bS, K, N}, gradX->dataType(), block.launchContext());
  auto gct = NDArrayFactory::create_(c->ordering(), {bS, K}, gradX->dataType(), block.launchContext());
  auto gradTanh = NDArrayFactory::create_(c->ordering(), {bS, K}, gradX->dataType(), block.launchContext());
  auto gradCt = NDArrayFactory::create_(c->ordering(), {bS, K}, gradX->dataType(), block.launchContext());
  auto ftMinus = NDArrayFactory::create_(c->ordering(), {bS, K}, gradX->dataType(), block.launchContext());
  auto rtMinus = NDArrayFactory::create_(c->ordering(), {bS, K}, gradX->dataType(), block.launchContext());
  auto temp1 = NDArrayFactory::create_(c->ordering(), {bS, K}, gradX->dataType(), block.launchContext());
  auto temp2 = NDArrayFactory::create_(c->ordering(), {bS, K}, gradX->dataType(), block.launchContext());

  std::vector<sd::LongType> axes = {0, 1};
  //  x = x * mask
  if (applyMask) x->applyBroadcast(broadcast::Multiply, &axes, *mask, *x);  // apply mask
  // multiplication matrix wi = matmul(w,x), U = WX
  auto wi = MmulHelper::mmul(w, x, nullptr, 1., 0.);  // U [bS x 3K x N]

  auto wiZ = (*wi)({0, 0, 0, K, 0, 0}, true);                // [bS x K x N]
  auto wiF = (*wi)({0, 0, K, 2 * K, 0, 0}, true);            // forget gate [bS x K x N]
  auto wiR = (*wi)({0, 0, 2 * K, 3 * K, 0, 0}, true);        // reset gate [bS x K x N]
  auto bF = (*b)({0, 0, 0, K}, true);                        // biases for forget gate [1 x K]
  auto bR = (*b)({0, 0, K, 2 * K}, true);                    // biases for reset gate [1 x K]
  auto gradBF = (*gradBias)({0, 0, 0, K, 0, 0}, true);       // [bS x K x N]
  auto gradBR = (*gradBias)({0, 0, K, 2 * K, 0, 0}, true);   // [bS x K x N]
  auto gradUZ = (*gradU)({0, 0, 0, K, 0, 0}, true);          // [bS x K x N]
  auto gradUF = (*gradU)({0, 0, K, 2 * K, 0, 0}, true);      // [bS x K x N]
  auto gradUR = (*gradU)({0, 0, 2 * K, 3 * K, 0, 0}, true);  // [bS x K x N]

  NDArray* ct_1 = nullptr;

  std::vector<sd::LongType> idx = {0, 0, 0, 0, 0, 0};

  for (int t = N - 1; t >= 0; --t) {
    // initialization
    idx[4] = t;
    idx[5] = t + 1;
    auto xt = (*x)(idx);              // [bS x K x N] -> [bS x K x 1] -> [bS x K]
    auto zt = wiZ(idx);               // [bS x K x N] -> [bS x K x 1] -> [bS x K]
    auto ft = wiF(idx);               // [bS x K x N] -> [bS x K x 1] -> [bS x K]
    auto rt = wiR(idx);               // [bS x K x N] -> [bS x K x 1] -> [bS x K]
    auto ct = (*c)(idx);              // [bS x K x N] -> [bS x K x 1] -> [bS x K]
    auto inGradHt = (*inGradH)(idx);  // [bS x K x N] -> [bS x K x 1] -> [bS x K]
    auto gradBRt = gradBR(idx);       // [bS x K x N] -> [bS x K x 1] -> [bS x K]
    auto gradBFt = gradBF(idx);       // [bS x K x N] -> [bS x K x 1] -> [bS x K]
    auto gradHXt = (*gradHX)(idx);    // [bS x K x N] -> [bS x K x 1] -> [bS x K]
    auto gradUZt = gradUZ(idx);       // [bS x K x N] -> [bS x K x 1] -> [bS x K]
    auto gradUFt = gradUF(idx);       // [bS x K x N] -> [bS x K x 1] -> [bS x K]
    auto gradURt = gradUR(idx);       // [bS x K x N] -> [bS x K x 1] -> [bS x K]

    if (t != 0) {
      idx[4] = t - 1;
      idx[5] = t;
      ct_1 = new NDArray((*c)(idx));  // previous c_{t-1}
    } else
      ct_1 = c0;

    ///////////////// forward
    // ft = sigmoid(ft + bf), rt = sigmoid(rt + bR)
    ft.addRowVector(bF, ft);
    rt.addRowVector(bR, rt);
    ft.applyTransform(transform::Sigmoid, ft);
    rt.applyTransform(transform::Sigmoid, rt);

    // TODO T val = (activation_type == 1) ? tanh(cur) : ((activation_type == 2) ? reluf(cur) : cur );
    ct.applyTransform(transform::Tanh, *gct);
    // ftMinus = 1-ft,  rtMinus = 1-rt
    ft.applyTransform(transform::OneMinus, *ftMinus);
    rt.applyTransform(transform::OneMinus, *rtMinus);

    ///////////////// backward
    // bR, *grad_brt_ptr = inGradHt * (g_ct - xt) * (1.0f - rt) * rt;
    gct->applyPairwiseTransform(pairwise::Subtract, xt, *temp1);      // temp1 = (g_ct - xt)
    rtMinus->applyPairwiseTransform(pairwise::Multiply, rt, *temp2);  // temp2 = (1.0f - rt) * rt;
    temp1->applyPairwiseTransform(pairwise::Multiply, *temp2);        // temp1 = (g_ct - xt) * (1.0f - rt) * rt;
    inGradHt.applyPairwiseTransform(pairwise::Multiply, *temp1,
                                    gradBRt);  // = inGradHt * (g_ct - xt) * (1.0f - rt) * rt;

    // bF, TODO - tanh
    // gradTanh = (1.0f - g_ct * g_ct);
    gct->applyPairwiseTransform(pairwise::Multiply, *gct, *gradTanh);  // gradTanh = g_ct * g_ct
    gradTanh->applyTransform(transform::OneMinus, *gradTanh);          // gradTanh = (1.0f - g_ct * g_ct)
    // gradCt  = inGradHt * rt * gradTanh
    rt.applyPairwiseTransform(pairwise::Multiply, *gradTanh, *gradCt);      // gradCt = rt * gradTanh
    inGradHt.applyPairwiseTransform(pairwise::Multiply, *gradCt, *gradCt);  // gradCt = inGradHt * rt * gradTanh
    // gradBFt = (gradCt + inGradCt) * (ct_1 - zt) * (1 - ft) * ft;
    gradCt->applyPairwiseTransform(pairwise::Add, *inGradCt, *temp1);     // temp1 = (gradCt + inGradCt)
    ct_1->applyPairwiseTransform(pairwise::Subtract, zt, *temp2);         // temp2 = (ct_1 - zt)
    temp1->applyPairwiseTransform(pairwise::Multiply, *ftMinus, *temp1);  // temp1 = (gradCt + inGradCt)*(1-ft)
    temp1->applyPairwiseTransform(pairwise::Multiply, ft, *temp1);        // temp1 = (gradCt + inGradCt)*(1-ft)*ft
    temp1->applyPairwiseTransform(pairwise::Multiply, *temp2,
                                  gradBFt);  // gradBFt = (gradCt + inGradCt) * (ct_1 - zt) * (1 - ft) * ft;

    // x_t (highway connection), gradHXt = inGradHt * (1.0f - rt);
    inGradHt.applyPairwiseTransform(pairwise::Multiply, *rtMinus, gradHXt);

    // U_t, gradUZt = (inGradHt * rt * grad_tanh + inGradCt) * (1.0f - ft);
    rt.applyPairwiseTransform(pairwise::Multiply, *gradTanh, *temp1);     // temp1 = rt * grad_tanh
    inGradHt.applyPairwiseTransform(pairwise::Multiply, *temp1, *temp1);  // temp1 = inGradHt * rt * grad_tanh
    temp1->applyPairwiseTransform(pairwise::Add, *inGradCt, *temp1);  // temp1 = inGradHt * rt * grad_tanh + inGradCt
    temp1->applyPairwiseTransform(pairwise::Multiply, *ftMinus,
                                  gradUZt);  // gradUZt = (inGradHt * rt * grad_tanh + inGradCt) * (1.0f - ft);
    gradUFt.assign(&gradBFt);
    gradURt.assign(&gradBRt);

    // c_{t-1}, inGradCt = (gradCt + inGradCt) * ft;
    gradCt->applyPairwiseTransform(pairwise::Add, *inGradCt, *temp1);  // temp1 = (gradCt + inGradCt)
    temp1->applyPairwiseTransform(pairwise::Multiply, ft, *inGradCt);  // inGradCt = (gradCt + inGradCt) * ft;

    if (t != 0) delete ct_1;
  }

  // gradInit
  gradInit->assign(inGradCt);

  // gradX
  auto weightsT = w->transpose();                                                    // [K x 3K]
  MmulHelper::mmul(&weightsT, gradU, gradX, 1., 0.);                                 // [bS x K x N]
  gradX->applyPairwiseTransform(pairwise::Add, *gradHX, *gradX);
  std::vector<sd::LongType> axes3 = {0, 1};
  // + grad_highway_x
  if (applyMask) gradX->applyBroadcast(broadcast::Multiply, &axes3, *mask, *gradX);  // apply mask

  // gradB
  auto gradB2 = gradB->reshape(gradB->ordering(), {2 * K});
  std::vector<sd::LongType> axes2;
  axes.push_back(0);
  axes.push_back(2);
  gradBias->reduceAlongDimension(reduce::Sum, gradB2, &axes2);  // [1 x 2K]

  // gradW [bS x 3K x K]
  x->permutei({0, 2, 1});                     // [bS x N x K]
  MmulHelper::mmul(gradU, x, gradW, 1., 0.);  // [bS x 3K x K]

  delete gct;
  delete gradU;
  delete gradHX;
  delete temp1;
  delete temp2;
  delete gradCt;
  delete wi;
  delete gradTanh;
  delete ftMinus;
  delete rtMinus;
  delete gradBias;

  return sd::Status::OK;
}

DECLARE_TYPES(sru_bp) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(sru_bp) {
  auto inShape = inputShape->at(0);  // [bS x inSize x time]
  auto bS = inShape[1];
  auto inSize = inShape[2];
  auto time = inShape[3];
  char order = (char)(inShape[9]);

  ShapeDescriptor *descriptor1 = new ShapeDescriptor(ArrayOptions::dataType(inShape), order, {bS, inSize, time});
  ShapeDescriptor *descriptor2 = new ShapeDescriptor(ArrayOptions::dataType(inShape), order, {bS, 3 * inSize, inSize});
  ShapeDescriptor *descriptor3 = new ShapeDescriptor(ArrayOptions::dataType(inShape), order, {1, 2 * inSize});
  ShapeDescriptor *descriptor4 = new ShapeDescriptor(ArrayOptions::dataType(inShape), order, {bS, inSize});

  auto ret =  SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(descriptor1),
                        ConstantShapeHelper::getInstance().createShapeInfo(descriptor2),
                        ConstantShapeHelper::getInstance().createShapeInfo(descriptor3),
                        ConstantShapeHelper::getInstance().createShapeInfo(descriptor4));
  delete descriptor1;
  delete descriptor2;
  delete descriptor3;
  delete descriptor4;
  return ret;
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_bi, 5, 2, true, 0, 0) {
  auto x = INPUT_VARIABLE(0);   // X, input 3d tensor [time x bS x 2*inSize], time - number of time steps, bS - batch
  // size, inSize - number of features
  auto w = INPUT_VARIABLE(1);   // W, 2d tensor of weights [2*inSize x 6*inSize]
  auto b = INPUT_VARIABLE(2);   // B, row of biases with twice length [1 x 4*inSize]
  auto c0 = INPUT_VARIABLE(3);  // C_{0}, 2d tensor of initial state [bS x 2*inSize] at time t=0
  NDArray* mask =
      block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;  // optional, 2d tensor of dropout mask [bS x 2*inSize]

  auto ht = OUTPUT_VARIABLE(0);  // h_t, [time x bS x 2*inSize]
  auto ct = OUTPUT_VARIABLE(1);  // c_t, [time x bS x 2*inSize]

  // input shapes validation
  const int rank = x->rankOf();
  const sd::LongType bS = x->sizeAt(1);
  const sd::LongType inSize = x->sizeAt(2) / 2;

  REQUIRE_TRUE(x->rankOf() == rank, 0,
               "SRU_BI operation: wrong rank of input array, expected is %i, but got %i instead !", rank, x->rankOf());
  REQUIRE_TRUE(w->rankOf() == rank - 1, 0,
               "SRU_BI operation: wrong rank of weights array, expected is %i, but got %i instead !", rank - 1,
               w->rankOf());
  REQUIRE_TRUE(b->rankOf() == 1, 0, "SRU_BI operation: wrong rank of biases array, expected is 1, but got %i instead !",
               b->rankOf());
  REQUIRE_TRUE(c0->rankOf() == rank - 1, 0,
               "SRU_BI operation: wrong rank of initial state array, expected is %i, but got %i instead !", rank - 1,
               c0->rankOf());
  if (mask)
  REQUIRE_TRUE(mask->rankOf() == rank - 1, 0,
               "SRU_BI operation: wrong rank of mask array, expected is %i, but got %i instead !", rank - 1,
               mask->rankOf());

  const std::vector<sd::LongType> wCorrectShape = {2 * inSize, 6 * inSize};
  const std::vector<sd::LongType> bCorrectShape = {4 * inSize};
  const std::vector<sd::LongType> c0CorrectShape = {bS, 2 * inSize};

  REQUIRE_TRUE(w->isSameShape(wCorrectShape), 0,
               "SRU_BI operation: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(wCorrectShape).c_str(), ShapeUtils::shapeAsString(w).c_str());
  REQUIRE_TRUE(b->isSameShape(bCorrectShape), 0,
               "SRU_BI operation: wrong shape of biases array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(b).c_str());
  REQUIRE_TRUE(c0->isSameShape(c0CorrectShape), 0,
               "SRU_BI operation: wrong shape of initial state array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(c0).c_str());
  if (mask)
  REQUIRE_TRUE(mask->isSameShape(c0CorrectShape), 0,
               "SRU_BI operation: wrong shape of mask array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(mask).c_str());

  helpers::sruBI(block.launchContext(), x, w, b, c0, mask, ht, ct);

  return sd::Status::OK;
}

DECLARE_TYPES(sru_bi) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(sru_bi) {
  auto xShapeInfo = inputShape->at(0);  // [time x bS x 2K ]
  auto wShapeInfo = inputShape->at(1);
  auto bShapeInfo = inputShape->at(2);
  auto c0ShapeInfo = inputShape->at(3);
  auto maskShapeInfo =
      block.width() > 4 ? inputShape->at(4) : nullptr;  // optional,  2d tensor of dropout mask [bS x inSize]

  const int rank = xShapeInfo[0];  // = 3
  const sd::LongType time = xShapeInfo[1];
  const sd::LongType bS = xShapeInfo[2];
  const sd::LongType inSize = xShapeInfo[3] / 2;

  // input shapes validation
  REQUIRE_TRUE(wShapeInfo[0] == rank - 1, 0,
               "SRU_BI operation: wrong rank of weights array, expected is %i, but got %i instead !", rank - 1,
               wShapeInfo[0]);
  REQUIRE_TRUE(bShapeInfo[0] == 1, 0,
               "SRU_BI operation: wrong rank of biases  array, expected is 1, but got %i instead !", bShapeInfo[0]);
  REQUIRE_TRUE(c0ShapeInfo[0] == rank - 1, 0,
               "SRU_BI operation: wrong rank of initial state array, expected is %i, but got %i instead !", rank - 1,
               c0ShapeInfo[0]);
  if (maskShapeInfo)
  REQUIRE_TRUE(maskShapeInfo[0] == rank - 1, 0,
               "SRU_BI operation: wrong rank of mask array, expected is %i, but got %i instead !", rank - 1,
               maskShapeInfo[0]);

  const std::vector<sd::LongType> wCorrectShape = {2 * inSize, 6 * inSize};
  const std::vector<sd::LongType> bCorrectShape = {4 * inSize};
  const std::vector<sd::LongType> c0CorrectShape = {bS, 2 * inSize};

  REQUIRE_TRUE(ShapeUtils::areShapesEqual(wShapeInfo, wCorrectShape), 0,
               "SRU_BI operation: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(wCorrectShape).c_str(), ShapeUtils::shapeAsString(wShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(bShapeInfo, bCorrectShape), 0,
               "SRU_BI operation: wrong shape of biases array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(bShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(c0ShapeInfo, c0CorrectShape), 0,
               "SRU_BI operation: wrong shape of initial state array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(c0ShapeInfo).c_str());
  if (maskShapeInfo)
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(maskShapeInfo, c0CorrectShape), 0,
               "SRU_BI operation: wrong shape of mask array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(maskShapeInfo).c_str());

  char order = shape::order(xShapeInfo);

  ShapeDescriptor *descriptor = new ShapeDescriptor(ArrayOptions::dataType(xShapeInfo), order, {time, bS, 2 * inSize});
  auto result = ConstantShapeHelper::getInstance().createShapeInfo(descriptor);
  return SHAPELIST(result, result);
}

DECLARE_TYPES(sru_bi_bp) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sru_bi_bp, 8, 4, true, 0, 0) {
  auto x = INPUT_VARIABLE(0);   // X, input 3d tensor [time x bS x 2*inSize], time - number of time steps, bS - batch
  // size, inSize - number of features
  auto w = INPUT_VARIABLE(1);   // W, 2d tensor of weights [2*inSize x 6*inSize]
  auto b = INPUT_VARIABLE(2);   // B, row of biases with twice length [4*inSize]
  auto c0 = INPUT_VARIABLE(3);  // C_{0}, 2d tensor of initial state [bS x 2*inSize] at time t=0
  auto ct = INPUT_VARIABLE(4);  // C, [time x bS x 2*inSize]
  auto inGradC0 = INPUT_VARIABLE(5);  // [bS x 2*inSize]
  auto inGradHt = INPUT_VARIABLE(6);  // [time x bS x 2*inSize]
  NDArray* mask =
      block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;  // optional,  2d tensor of dropout mask [bS x 2*inSize]

  // input shapes validation
  const int rank = x->rankOf();
  const sd::LongType time = x->sizeAt(0);
  const sd::LongType bS = x->sizeAt(1);
  const sd::LongType inSize = x->sizeAt(2) / 2;

  REQUIRE_TRUE(w->rankOf() == rank - 1, 0,
               "SRU_BI_BP operation: wrong rank of weights array, expected is %i, but got %i instead !", rank - 1,
               w->rankOf());
  REQUIRE_TRUE(b->rankOf() == 1, 0,
               "SRU_BI_BP operation: wrong rank of biases array, expected is 1, but got %i instead !", b->rankOf());
  REQUIRE_TRUE(c0->rankOf() == rank - 1, 0,
               "SRU_BI_BP operation: wrong rank of initial state array, expected is %i, but got %i instead !", rank - 1,
               c0->rankOf());
  REQUIRE_TRUE(ct->rankOf() == rank, 0,
               "SRU_BI_BP operation: wrong rank of state array, expected is %i, but got %i instead !", rank,
               ct->rankOf());
  REQUIRE_TRUE(inGradC0->rankOf() == rank - 1, 0,
               "SRU_BI_BP operation: wrong rank of gradient c0, expected is %i, but got %i instead !", rank - 1,
               inGradC0->rankOf());
  REQUIRE_TRUE(inGradHt->rankOf() == rank, 0,
               "SRU_BI_BP operation: wrong rank of gradient ht, expected is %i, but got %i instead !", rank,
               inGradHt->rankOf());
  if (mask)
  REQUIRE_TRUE(mask->rankOf() == rank - 1, 0,
               "SRU_BI_BP operation: wrong rank of mask array, expected is %i, but got %i instead !", rank - 1,
               mask->rankOf());

  const std::vector<sd::LongType> wCorrectShape = {2 * inSize, 6 * inSize};
  const std::vector<sd::LongType> bCorrectShape = {4 * inSize};
  const std::vector<sd::LongType> c0CorrectShape = {bS, 2 * inSize};
  const std::vector<sd::LongType> ctCorrectShape = {time, bS, 2 * inSize};

  REQUIRE_TRUE(w->isSameShape(wCorrectShape), 0,
               "SRU_BI operation: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(wCorrectShape).c_str(), ShapeUtils::shapeAsString(w).c_str());
  REQUIRE_TRUE(b->isSameShape(bCorrectShape), 0,
               "SRU_BI operation: wrong shape of biases  array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(b).c_str());
  REQUIRE_TRUE(c0->isSameShape(c0CorrectShape), 0,
               "SRU_BI operation: wrong shape of initial state array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(c0).c_str());
  REQUIRE_TRUE(ct->isSameShape(ctCorrectShape), 0,
               "SRU_BI operation: wrong shape of state array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(ctCorrectShape).c_str(), ShapeUtils::shapeAsString(ct).c_str());
  if (mask)
  REQUIRE_TRUE(mask->isSameShape(c0CorrectShape), 0,
               "SRU_BI operation: wrong shape of mask array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(mask).c_str());

  auto gradI = OUTPUT_VARIABLE(0);   // [time x bS x 2*inSize]
  auto gradW = OUTPUT_VARIABLE(1);   // [time x 2*inSize x 6*inSize]
  auto gradB = OUTPUT_VARIABLE(2);   // [1 x 4*inSize]
  auto gradC0 = OUTPUT_VARIABLE(3);  // [bS x 2*inSize]

  helpers::sruBIBP(block.launchContext(), x, w, b, c0, ct, inGradC0, inGradHt, mask, gradI, gradW, gradB, gradC0);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(sru_bi_bp) {
  auto xShapeInfo = inputShape->at(0);  // [time x bS x 2K ]
  auto wShapeInfo = inputShape->at(1);
  auto bShapeInfo = inputShape->at(2);
  auto c0ShapeInfo = inputShape->at(3);
  auto ctShapeInfo = inputShape->at(4);
  auto inGradC0ShapeInfo = inputShape->at(5);
  auto inGradHtShapeInfo = inputShape->at(6);
  auto maskShapeInfo =
      block.width() > 7 ? inputShape->at(7) : nullptr;  // optional,  2d tensor of dropout mask [bS x inSize]

  // input shapes validation
  const int rank = xShapeInfo[0];
  const sd::LongType time = xShapeInfo[1];
  const sd::LongType bS = xShapeInfo[2];
  const sd::LongType inSize = xShapeInfo[3] / 2;

  REQUIRE_TRUE(wShapeInfo[0] == rank - 1, 0,
               "SRU_BI_BP operation: wrong rank of weights array, expected is %i, but got %i instead !", rank - 1,
               wShapeInfo[0]);
  REQUIRE_TRUE(bShapeInfo[0] == 1, 0,
               "SRU_BI_BP operation: wrong rank of biases  array, expected is 1, but got %i instead !", bShapeInfo[0]);
  REQUIRE_TRUE(c0ShapeInfo[0] == rank - 1, 0,
               "SRU_BI_BP operation: wrong rank of initial state array, expected is %i, but got %i instead !", rank - 1,
               c0ShapeInfo);
  REQUIRE_TRUE(ctShapeInfo[0] == rank, 0,
               "SRU_BI_BP operation: wrong rank of state array, expected is %i, but got %i instead !", rank,
               ctShapeInfo);
  REQUIRE_TRUE(inGradC0ShapeInfo[0] == rank - 1, 0,
               "SRU_BI_BP operation: wrong rank of gradient c0, expected is %i, but got %i instead !", rank - 1,
               inGradC0ShapeInfo[0]);
  REQUIRE_TRUE(inGradHtShapeInfo[0] == rank, 0,
               "SRU_BI_BP operation: wrong rank of gradient ht, expected is %i, but got %i instead !", rank,
               inGradHtShapeInfo[0]);
  if (maskShapeInfo)
  REQUIRE_TRUE(maskShapeInfo[0] == rank - 1, 0,
               "SRU_BI_BP operation: wrong rank of mask array, expected is %i, but got %i instead !", rank - 1,
               maskShapeInfo[0]);

  const std::vector<sd::LongType> wCorrectShape = {2 * inSize, 6 * inSize};
  const std::vector<sd::LongType> bCorrectShape = {4 * inSize};
  const std::vector<sd::LongType> c0CorrectShape = {bS, 2 * inSize};
  const std::vector<sd::LongType> ctCorrectShape = {time, bS, 2 * inSize};
  const std::vector<sd::LongType> inGradC0CorrectShape = {bS, 2 * inSize};
  const std::vector<sd::LongType> inGradHtCorrectShape = {time, bS, 2 * inSize};

  REQUIRE_TRUE(ShapeUtils::areShapesEqual(wShapeInfo, wCorrectShape), 0,
               "SRU_BI operation: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(wCorrectShape).c_str(), ShapeUtils::shapeAsString(wShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(bShapeInfo, bCorrectShape), 0,
               "SRU_BI operation: wrong shape of biases  array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(bShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(c0ShapeInfo, c0CorrectShape), 0,
               "SRU_BI operation: wrong shape of initial state array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(c0ShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(ctShapeInfo, ctCorrectShape), 0,
               "SRU_BI operation: wrong shape of state array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(ctCorrectShape).c_str(), ShapeUtils::shapeAsString(ctShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(inGradC0ShapeInfo, inGradC0CorrectShape), 0,
               "SRU_BI operation: wrong shape of gradient c0 array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(inGradC0CorrectShape).c_str(),
               ShapeUtils::shapeAsString(inGradC0ShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(inGradHtShapeInfo, inGradHtCorrectShape), 0,
               "SRU_BI operation: wrong shape of gradient ht array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(inGradHtCorrectShape).c_str(),
               ShapeUtils::shapeAsString(inGradHtShapeInfo).c_str());
  if (maskShapeInfo)
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(maskShapeInfo, c0CorrectShape), 0,
               "SRU_BI operation: wrong shape of mask array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(c0CorrectShape).c_str(), ShapeUtils::shapeAsString(maskShapeInfo).c_str());

  const char order = shape::order(xShapeInfo);

  ShapeDescriptor *descriptor1 = new ShapeDescriptor(ArrayOptions::dataType(xShapeInfo), order, {time, bS, 2 * inSize});
  ShapeDescriptor *descriptor2 = new ShapeDescriptor(ArrayOptions::dataType(xShapeInfo), order, {time, 2 * inSize, 6 * inSize});
  ShapeDescriptor *descriptor3 = new ShapeDescriptor(ArrayOptions::dataType(xShapeInfo), order, {4 * inSize});
  ShapeDescriptor *descriptor4 = new ShapeDescriptor(ArrayOptions::dataType(xShapeInfo), order, {bS, 2 * inSize});

  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(descriptor1),
                   ConstantShapeHelper::getInstance().createShapeInfo(descriptor2),
                   ConstantShapeHelper::getInstance().createShapeInfo(descriptor3),
                   ConstantShapeHelper::getInstance().createShapeInfo(descriptor4));
}

}  // namespace ops
}  // namespace sd

#endif
