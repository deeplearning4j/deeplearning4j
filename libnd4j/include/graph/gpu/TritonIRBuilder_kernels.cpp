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
// TritonIRBuilder — High-level kernel patterns:
//   emitMatmulKernel, emitFusedAttentionKernel
//

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonIRBuilder.h>
#include <graph/gpu/TritonIRBuilder_internal.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include <algorithm>
#include <cmath>

// MLIR core
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

// Triton MLIR dialect
#include <triton/Dialect/Triton/IR/Dialect.h>
#include <triton/Dialect/Triton/IR/Types.h>

// Standard MLIR dialects
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

namespace sd {
namespace graph {

using namespace ir_builder_internal;

// ─── Matmul op emission ─────────────────────────────────────────────────────

// No-epilogue overload — delegates to the full version with empty vectors
void TritonIRBuilder::emitMatmulKernel(mlir::OpBuilder& builder, mlir::Location loc,
                                        mlir::Value aPtr, mlir::Value bPtr, mlir::Value cPtr,
                                        int M, int N, int K,
                                        int blockM, int blockN, int blockK) {
  std::vector<EpilogueOp> noEpi;
  std::vector<mlir::Value> noPtrs;
  emitMatmulKernel(builder, loc, aPtr, bPtr, cPtr, M, N, K, blockM, blockN, blockK, noEpi, noPtrs);
}

void TritonIRBuilder::emitMatmulKernel(mlir::OpBuilder& builder, mlir::Location loc,
                                        mlir::Value aPtr, mlir::Value bPtr, mlir::Value cPtr,
                                        int M, int N, int K,
                                        int blockM, int blockN, int blockK,
                                        const std::vector<EpilogueOp>& epilogueOps,
                                        const std::vector<mlir::Value>& epiloguePtrs) {
  DSP_DIAG(JIT, "emitMatmulKernel: M=%d N=%d K=%d block=(%d,%d,%d) epilogues=%d",
           M, N, K, blockM, blockN, blockK, (int)epilogueOps.size());
  auto f32Type = builder.getF32Type();
  auto i32Type = builder.getI32Type();
  auto i1Type = builder.getI1Type();

  // Extract element types from pointer args for mixed-precision support.
  // Inputs (A, B) may be f16/bf16/int8; accumulator is always f32;
  // output (C) stores in its native type with cast from f32 if needed.
  auto aPtrType = mlir::cast<mlir::triton::PointerType>(aPtr.getType());
  auto bPtrType = mlir::cast<mlir::triton::PointerType>(bPtr.getType());
  auto cPtrType = mlir::cast<mlir::triton::PointerType>(cPtr.getType());
  auto aElemType = aPtrType.getPointeeType();
  auto bElemType = bPtrType.getPointeeType();
  auto cElemType = cPtrType.getPointeeType();

  // Determine InputPrecision for DotOp based on input types and Environment flag.
  // TF32 (10-bit mantissa) gives ~2x throughput on sm_80+ but compounds precision
  // loss across thousands of ops per decode step. Default OFF for correctness.
  bool inputIsF32 = mlir::isa<mlir::Float32Type>(aElemType);
  bool useTf32 = inputIsF32 && sd::Environment::getInstance().tritonTf32Enabled();
  auto dotPrecision = useTf32 ? mlir::triton::InputPrecision::TF32
                               : mlir::triton::InputPrecision::IEEE;

  DSP_DIAG(JIT, "TritonIRBuilder::emitMatmulKernel: A elem=%s, B elem=%s, C elem=%s, precision=%s",
            inputIsF32 ? "f32" : "non-f32", inputIsF32 ? "f32" : "non-f32",
            mlir::isa<mlir::Float32Type>(cElemType) ? "f32" : "non-f32",
            useTf32 ? "TF32" : "IEEE");

  // Program IDs for 2D grid
  auto pidM = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);
  auto pidN = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::Y);

  // Tile index offsets
  auto blockMConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockM, 32);
  auto blockNConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockN, 32);
  auto mOffset = builder.create<mlir::arith::MulIOp>(loc, pidM, blockMConst);
  auto nOffset = builder.create<mlir::arith::MulIOp>(loc, pidN, blockNConst);

  // Create range vectors for tile offsets
  auto i32BmType = mlir::RankedTensorType::get({blockM}, i32Type);
  auto i32BnType = mlir::RankedTensorType::get({blockN}, i32Type);
  auto i32BkType = mlir::RankedTensorType::get({blockK}, i32Type);

  auto rangeM = builder.create<mlir::triton::MakeRangeOp>(loc, i32BmType, 0, blockM);
  auto rangeN = builder.create<mlir::triton::MakeRangeOp>(loc, i32BnType, 0, blockN);
  auto rangeK = builder.create<mlir::triton::MakeRangeOp>(loc, i32BkType, 0, blockK);

  auto splatMOffset = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mOffset);
  auto mIndices = builder.create<mlir::arith::AddIOp>(loc, splatMOffset, rangeM);
  auto splatNOffset = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nOffset);
  auto nIndices = builder.create<mlir::arith::AddIOp>(loc, splatNOffset, rangeN);

  // Initialize accumulator to zeros: always f32 (tensor cores accumulate in f32)
  auto accType = mlir::RankedTensorType::get({blockM, blockN}, f32Type);
  auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);
  auto zeroScalar = builder.create<mlir::arith::ConstantOp>(loc, f32Type, zeroAttr);
  auto accInit = builder.create<mlir::triton::SplatOp>(loc, accType, zeroScalar);

  // K-loop bounds (i32 — Triton convention, NOT index type)
  auto kStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto kEnd = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto kStep = builder.create<mlir::arith::ConstantIntOp>(loc, blockK, 32);

  // K-loop via scf.for (i32 bounds)
  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, kStart, kEnd, kStep, mlir::ValueRange{accInit});

  // Inside the K-loop body
  builder.setInsertionPointToStart(forOp.getBody());
  auto kIdxI32 = forOp.getInductionVar();  // i32 induction variable
  auto accIter = forOp.getBody()->getArgument(1);  // loop-carried accumulator

  // Splat k offset for pointer arithmetic
  auto splatKOffset = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kIdxI32);
  auto kIndices = builder.create<mlir::arith::AddIOp>(loc, splatKOffset, rangeK);

  // Load A tile [BM, BK] in native dtype
  auto kConst = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);

  // Compute 2D pointer offsets for A: mIndices[:, None] * K + kIndices[None, :]
  auto mExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);  // [BM, 1]
  auto kExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 0);  // [1, BK]

  auto i32BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i32Type);
  auto kSplat = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, 1}, i32Type), kConst);
  auto mTimesK = builder.create<mlir::arith::MulIOp>(loc, mExpanded, kSplat);
  auto mTimesKBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, mTimesK);
  auto kBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, kExpanded);
  auto aOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesKBroadcast, kBroadcast);

  auto aPtrTensorType = mlir::RankedTensorType::get({blockM, blockK},
      mlir::triton::PointerType::get(aElemType, 1));
  auto aSplat = builder.create<mlir::triton::SplatOp>(loc, aPtrTensorType, aPtr);
  auto aPtrs = builder.create<mlir::triton::AddPtrOp>(loc, aPtrTensorType, aSplat, aOffsets);

  // Create 2D mask for A tile: mIndices < M && kIndices < K
  auto mConst = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto kConst2 = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto mConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConst);
  auto kConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kConst2);
  auto mMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      builder.create<mlir::arith::AddIOp>(loc, splatMOffset, rangeM), mConstSplat);
  auto kMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      kIndices, kConstSplat);
  auto i1BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i1Type);
  auto mMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1D, 1);
  auto kMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 0);
  auto mMask2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkType, mMaskExp);
  auto kMask2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkType, kMaskExp);
  auto aMask = builder.create<mlir::arith::AndIOp>(loc, mMask2D, kMask2D);

  auto aLoaded = builder.create<mlir::triton::LoadOp>(loc,
      /*ptr=*/aPtrs.getResult(), /*mask=*/aMask.getResult(), /*other=*/mlir::Value(),
      /*cache=*/mlir::triton::CacheModifier::NONE,
      /*evict=*/mlir::triton::EvictionPolicy::NORMAL,
      /*isVolatile=*/false);

  // Load B tile [BK, BN] in native dtype
  auto nConst = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);

  auto kExpandedB = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 1);  // [BK, 1]
  auto nExpandedB = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);  // [1, BN]

  auto i32BkBnType = mlir::RankedTensorType::get({blockK, blockN}, i32Type);
  auto nSplat = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockK, 1}, i32Type), nConst);
  auto kTimesN = builder.create<mlir::arith::MulIOp>(loc, kExpandedB, nSplat);
  auto kTimesNBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, kTimesN);
  auto nBroadcastB = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, nExpandedB);
  auto bOffsets = builder.create<mlir::arith::AddIOp>(loc, kTimesNBroadcast, nBroadcastB);

  auto bPtrTensorType = mlir::RankedTensorType::get({blockK, blockN},
      mlir::triton::PointerType::get(bElemType, 1));
  auto bSplat = builder.create<mlir::triton::SplatOp>(loc, bPtrTensorType, bPtr);
  auto bPtrs = builder.create<mlir::triton::AddPtrOp>(loc, bPtrTensorType, bSplat, bOffsets);

  // Create 2D mask for B tile: kIndices < K && nIndices < N
  auto nConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nConst);
  auto nMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      builder.create<mlir::arith::AddIOp>(loc, splatNOffset, rangeN), nConstSplat);
  auto i1BkBnType = mlir::RankedTensorType::get({blockK, blockN}, i1Type);
  auto kMaskExpB = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 1);
  auto nMaskExpB = builder.create<mlir::triton::ExpandDimsOp>(loc, nMask1D, 0);
  auto kMask2DB = builder.create<mlir::triton::BroadcastOp>(loc, i1BkBnType, kMaskExpB);
  auto nMask2DB = builder.create<mlir::triton::BroadcastOp>(loc, i1BkBnType, nMaskExpB);
  auto bMask = builder.create<mlir::arith::AndIOp>(loc, kMask2DB, nMask2DB);

  auto bLoaded = builder.create<mlir::triton::LoadOp>(loc,
      /*ptr=*/bPtrs.getResult(), /*mask=*/bMask.getResult(), /*other=*/mlir::Value(),
      /*cache=*/mlir::triton::CacheModifier::NONE,
      /*evict=*/mlir::triton::EvictionPolicy::NORMAL,
      /*isVolatile=*/false);

  // Matrix multiply: acc += dot(A_tile, B_tile)
  // tt.dot requires A and B to have same element type.
  // Accumulator is always f32. Tensor cores handle f16/bf16→f32 natively.
  // When FP16 weight quantization is active, A may be f32 while B is f16.
  // Truncate the wider operand to the narrower for tensor core compatibility.
  mlir::Value aForDot = aLoaded.getResult();
  mlir::Value bForDot = bLoaded.getResult();
  if (aElemType != bElemType) {
    if (mlir::isa<mlir::Float32Type>(aElemType) && !mlir::isa<mlir::Float32Type>(bElemType)) {
      auto aCastType = mlir::RankedTensorType::get({blockM, blockK}, bElemType);
      aForDot = builder.create<mlir::arith::TruncFOp>(loc, aCastType, aForDot);
    } else if (!mlir::isa<mlir::Float32Type>(aElemType) && mlir::isa<mlir::Float32Type>(bElemType)) {
      auto bCastType = mlir::RankedTensorType::get({blockK, blockN}, aElemType);
      bForDot = builder.create<mlir::arith::TruncFOp>(loc, bCastType, bForDot);
    }
  }
  auto dotResult = builder.create<mlir::triton::DotOp>(
      loc, accType, aForDot, bForDot, accIter,
      dotPrecision, /*maxNumImpreciseAcc=*/0);

  // Yield accumulator for next K-iteration
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{dotResult});

  // After the K-loop — apply epilogue ops to accumulator IN REGISTERS,
  // then store. This is the mega-kernel fusion point: data never leaves
  // registers between matmul and epilogue (bias add, activation, cast).
  builder.setInsertionPointAfter(forOp);
  auto finalAcc = forOp.getResult(0);  // f32 accumulator [blockM, blockN]
  mlir::Value acc = finalAcc;

  // ── Apply epilogue ops in-register ──
  for (size_t epiIdx = 0; epiIdx < epilogueOps.size(); epiIdx++) {
    const auto& epi = epilogueOps[epiIdx];
    auto accTensorType = mlir::cast<mlir::RankedTensorType>(acc.getType());

    switch (epi.type) {
      case EpilogueOp::BIAS_ADD: {
        // Load bias vector [N] and broadcast to [blockM, blockN]
        if (epi.biasArgIndex >= 0 && epi.biasArgIndex < static_cast<int>(epiloguePtrs.size())) {
          auto biasPtr = epiloguePtrs[epi.biasArgIndex];
          auto biasPtrType = mlir::cast<mlir::triton::PointerType>(biasPtr.getType());
          auto biasElemType = biasPtrType.getPointeeType();
          auto biasPtrTensorType = mlir::RankedTensorType::get({blockN},
              mlir::triton::PointerType::get(biasElemType, 1));
          auto biasSplat = builder.create<mlir::triton::SplatOp>(loc, biasPtrTensorType, biasPtr);
          // nIndices already exists from the matmul body above
          auto biasPtrs = builder.create<mlir::triton::AddPtrOp>(loc, biasPtrTensorType, biasSplat, nIndices);
          auto biasLoaded = builder.create<mlir::triton::LoadOp>(loc,
              biasPtrs.getResult(), /*mask=*/mlir::Value(), /*other=*/mlir::Value(),
              mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
          // Cast bias to f32 if needed
          mlir::Value biasF32 = biasLoaded.getResult();
          if (biasElemType != f32Type) {
            auto f32VecType = mlir::RankedTensorType::get({blockN}, f32Type);
            biasF32 = builder.create<mlir::arith::ExtFOp>(loc, f32VecType, biasF32);
          }
          // Broadcast [blockN] → [blockM, blockN]
          auto biasExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, biasF32, 0); // [1, blockN]
          auto biasBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, accTensorType, biasExpanded);
          acc = builder.create<mlir::arith::AddFOp>(loc, acc, biasBroadcast);
        }
        break;
      }
      case EpilogueOp::RESIDUAL_ADD: {
        // Element-wise add of a [M, N] tensor (e.g., transformer residual).
        // Unlike BIAS_ADD which loads bias[n] and broadcasts across M, this
        // loads residual[m, n] for each tile element. Offsets: m*N + n.
        if (epi.biasArgIndex >= 0 && epi.biasArgIndex < static_cast<int>(epiloguePtrs.size())) {
          auto resPtr = epiloguePtrs[epi.biasArgIndex];
          auto resPtrType = mlir::cast<mlir::triton::PointerType>(resPtr.getType());
          auto resElemType = resPtrType.getPointeeType();

          // Compute 2D offsets: m*N + n for each (m, n) in [blockM, blockN].
          // Reuse mIndices, nIndices, nConst from the matmul body above.
          auto mExpandedRes = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);  // [blockM, 1]
          auto nExpandedRes = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);  // [1, blockN]
          auto i32BmBnTypeRes = mlir::RankedTensorType::get({blockM, blockN}, i32Type);
          auto nSplatRes = builder.create<mlir::triton::SplatOp>(loc,
              mlir::RankedTensorType::get({blockM, 1}, i32Type), nConst);
          auto mTimesNRes = builder.create<mlir::arith::MulIOp>(loc, mExpandedRes, nSplatRes);
          auto mTimesNBroadcastRes = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnTypeRes, mTimesNRes);
          auto nBroadcastRes = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnTypeRes, nExpandedRes);
          auto resOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesNBroadcastRes, nBroadcastRes);

          // Build pointer tensor and load
          auto resPtrTensorType = mlir::RankedTensorType::get({blockM, blockN},
              mlir::triton::PointerType::get(resElemType, 1));
          auto resSplat = builder.create<mlir::triton::SplatOp>(loc, resPtrTensorType, resPtr);
          auto resPtrs = builder.create<mlir::triton::AddPtrOp>(loc, resPtrTensorType, resSplat, resOffsets);
          auto resLoaded = builder.create<mlir::triton::LoadOp>(loc,
              resPtrs.getResult(), /*mask=*/mlir::Value(), /*other=*/mlir::Value(),
              mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

          // Cast to f32 if needed
          mlir::Value resF32 = resLoaded.getResult();
          if (resElemType != f32Type) {
            resF32 = builder.create<mlir::arith::ExtFOp>(loc, accTensorType, resF32);
          }
          acc = builder.create<mlir::arith::AddFOp>(loc, acc, resF32);
        }
        break;
      }
      case EpilogueOp::RELU: {
        auto zeroAttr2 = builder.getFloatAttr(f32Type, 0.0);
        auto zeroScalar2 = builder.create<mlir::arith::ConstantOp>(loc, f32Type, zeroAttr2);
        auto zeroTensor = builder.create<mlir::triton::SplatOp>(loc, accTensorType, zeroScalar2);
        acc = builder.create<mlir::arith::MaximumFOp>(loc, acc, zeroTensor);
        break;
      }
      case EpilogueOp::GELU: {
        // Match native GELU: x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 * x))
        auto coeffScalar = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 1.702));
        auto coeffT = builder.create<mlir::triton::SplatOp>(loc, accTensorType, coeffScalar);
        auto scaled = builder.create<mlir::arith::MulFOp>(loc, coeffT, acc);
        auto neg = builder.create<mlir::arith::NegFOp>(loc, scaled);
        auto expNeg = builder.create<mlir::math::ExpOp>(loc, neg);
        auto oneT = builder.create<mlir::triton::SplatOp>(loc, accTensorType,
            builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 1.0)));
        auto denom = builder.create<mlir::arith::AddFOp>(loc, oneT, expNeg);
        auto sigmoid = builder.create<mlir::arith::DivFOp>(loc, oneT, denom);
        acc = builder.create<mlir::arith::MulFOp>(loc, acc, sigmoid);
        break;
      }
      case EpilogueOp::SILU: {
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        auto negAcc = builder.create<mlir::arith::NegFOp>(loc, acc);
        auto expNeg = builder.create<mlir::math::ExpOp>(loc, negAcc);
        auto oneT = builder.create<mlir::triton::SplatOp>(loc, accTensorType,
            builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 1.0)));
        auto denom = builder.create<mlir::arith::AddFOp>(loc, oneT, expNeg);
        auto sigmoid = builder.create<mlir::arith::DivFOp>(loc, oneT, denom);
        acc = builder.create<mlir::arith::MulFOp>(loc, acc, sigmoid);
        break;
      }
      case EpilogueOp::TANH: {
        acc = builder.create<mlir::math::TanhOp>(loc, acc);
        break;
      }
      case EpilogueOp::SIGMOID: {
        auto negAcc = builder.create<mlir::arith::NegFOp>(loc, acc);
        auto expNeg = builder.create<mlir::math::ExpOp>(loc, negAcc);
        auto oneT = builder.create<mlir::triton::SplatOp>(loc, accTensorType,
            builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 1.0)));
        auto denom = builder.create<mlir::arith::AddFOp>(loc, oneT, expNeg);
        acc = builder.create<mlir::arith::DivFOp>(loc, oneT, denom);
        break;
      }
      default:
        // HARD ERROR: an unknown epilogue op type was added to the list
        // but no handler exists. Silently dropping it would corrupt output
        // (the op would be skipped, downstream consumers see wrong data).
        // Instead fail loudly so the missing handler is added.
        THROW_EXCEPTION("TritonIRBuilder: unhandled EpilogueOp type — would silently drop op and corrupt output");
    }
  }

  // Cast epilogue-processed accumulator to output type if needed.
  //  cast from `acc` (post-epilogue), NOT `finalAcc` (raw matmul).
  // Using finalAcc here would silently bypass all in-register epilogue ops
  // (relu, gelu, bias_add, etc.), causing 3.0+ output divergence.
  mlir::Value storeVal = acc;
  if (cElemType != f32Type) {
    auto cTileType = mlir::RankedTensorType::get({blockM, blockN}, cElemType);
    if (mlir::isa<mlir::FloatType>(cElemType)) {
      storeVal = builder.create<mlir::arith::TruncFOp>(loc, cTileType, acc);
    } else if (mlir::isa<mlir::IntegerType>(cElemType)) {
      storeVal = builder.create<mlir::arith::FPToSIOp>(loc, cTileType, acc);
    }
  }

  // Compute C pointers: c_ptr + mIndices * N + nIndices
  auto mExpandedC = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);  // [BM, 1]
  auto nExpandedC = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);  // [1, BN]

  auto i32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i32Type);
  auto nSplatC = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, 1}, i32Type), nConst);
  auto mTimesNC = builder.create<mlir::arith::MulIOp>(loc, mExpandedC, nSplatC);
  auto mTimesNCBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, mTimesNC);
  auto nBroadcastC = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, nExpandedC);
  auto cOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesNCBroadcast, nBroadcastC);

  auto cPtrTensorType = mlir::RankedTensorType::get({blockM, blockN},
      mlir::triton::PointerType::get(cElemType, 1));
  auto cSplat = builder.create<mlir::triton::SplatOp>(loc, cPtrTensorType, cPtr);
  auto cPtrs = builder.create<mlir::triton::AddPtrOp>(loc, cPtrTensorType, cSplat, cOffsets);

  // Create 2D mask for C tile: mIndices < M && nIndices < N
  auto mConstC = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto nConstC = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto mConstSplatC = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConstC);
  auto nConstSplatC = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nConstC);
  auto mMask1DC = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, mIndices, mConstSplatC);
  auto nMask1DC = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, nIndices, nConstSplatC);
  auto i1BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i1Type);
  auto mMaskExpC = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1DC, 1);
  auto nMaskExpC = builder.create<mlir::triton::ExpandDimsOp>(loc, nMask1DC, 0);
  auto mMask2DC = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, mMaskExpC);
  auto nMask2DC = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, nMaskExpC);
  auto cMask = builder.create<mlir::arith::AndIOp>(loc, mMask2DC, nMask2DC);

  builder.create<mlir::triton::StoreOp>(loc, cPtrs, storeVal, cMask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  DSP_DIAG(JIT, "TritonIRBuilder: emitted matmul kernel M=%d N=%d K=%d BM=%d BN=%d BK=%d",
            M, N, K, blockM, blockN, blockK);
}

// ─── Fused RMSNorm + Linear (Mirage-style single-pass) ──────────────────────
//
// Mirage's key insight: RMSNorm(x) @ W can be computed in a single tiled loop.
// Instead of: (1) compute Σx² → rsqrt → normalize → store, (2) load → matmul → store
// We do: in ONE K-loop, accumulate BOTH Σx² (scalar per row) AND Σ(x*gamma)*W (the matmul).
// After the loop, divide the matmul accumulator by RMS. One read of x, one write of output.
//
// This is algebraically equivalent because:
//   rms_norm(x) @ W = (x * gamma / rms(x)) @ W = (x * gamma @ W) / rms(x)
// The division by rms(x) commutes out of the matmul since it's a per-row scalar.
//
// Grid: 2D, (ceil(M/blockM), ceil(N/blockN))
// Per tile: K-loop loads x[blockM, blockK], gamma[blockK], W[blockK, blockN]
//   acc_matmul[blockM, blockN] += (x_tile * gamma_tile) @ W_tile   (via tt.dot)
//   acc_sq[blockM] += sum_k(x_tile * x_tile)                       (per-row reduction)
// After loop: rsqrt_rms = rsqrt(acc_sq / K + eps), output = acc_matmul * rsqrt_rms

void TritonIRBuilder::emitRmsNormLinearKernel(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::Value xPtr, mlir::Value gammaPtr, mlir::Value wPtr, mlir::Value outPtr,
    int M, int N, int K, float epsilon,
    int blockM, int blockN, int blockK) {

  DSP_DIAG(JIT, "emitRmsNormLinearKernel: M=%d N=%d K=%d eps=%.6f block=(%d,%d,%d)",
           M, N, K, epsilon, blockM, blockN, blockK);
  auto f32Type = builder.getF32Type();
  auto i32Type = builder.getI32Type();
  auto i1Type = builder.getI1Type();

  auto xPtrType = mlir::cast<mlir::triton::PointerType>(xPtr.getType());
  auto xElemType = xPtrType.getPointeeType();
  auto wPtrType = mlir::cast<mlir::triton::PointerType>(wPtr.getType());
  auto wElemType = wPtrType.getPointeeType();
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outPtr.getType());
  auto outElemType = outPtrType.getPointeeType();
  auto gammaPtrType = mlir::cast<mlir::triton::PointerType>(gammaPtr.getType());
  auto gammaElemType = gammaPtrType.getPointeeType();

  bool useTf32 = mlir::isa<mlir::Float32Type>(xElemType) &&
                 sd::Environment::getInstance().tritonTf32Enabled();
  auto dotPrecision = useTf32 ? mlir::triton::InputPrecision::TF32
                               : mlir::triton::InputPrecision::IEEE;

  // Program IDs for 2D grid
  auto pidM = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);
  auto pidN = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::Y);

  auto blockMConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockM, 32);
  auto blockNConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockN, 32);
  auto mOffset = builder.create<mlir::arith::MulIOp>(loc, pidM, blockMConst);
  auto nOffset = builder.create<mlir::arith::MulIOp>(loc, pidN, blockNConst);

  auto i32BmType = mlir::RankedTensorType::get({blockM}, i32Type);
  auto i32BnType = mlir::RankedTensorType::get({blockN}, i32Type);
  auto i32BkType = mlir::RankedTensorType::get({blockK}, i32Type);

  auto rangeM = builder.create<mlir::triton::MakeRangeOp>(loc, i32BmType, 0, blockM);
  auto rangeN = builder.create<mlir::triton::MakeRangeOp>(loc, i32BnType, 0, blockN);
  auto rangeK = builder.create<mlir::triton::MakeRangeOp>(loc, i32BkType, 0, blockK);

  auto splatMOffset = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mOffset);
  auto mIndices = builder.create<mlir::arith::AddIOp>(loc, splatMOffset, rangeM);
  auto splatNOffset = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nOffset);
  auto nIndices = builder.create<mlir::arith::AddIOp>(loc, splatNOffset, rangeN);

  // ── Row boundary mask: prevent OOB reads/writes when M < blockM ──
  // When M=1 (decode) with blockM=128, rows 1..127 are out-of-bounds.
  // Without masking, the kernel reads arbitrary GPU memory for those rows
  // which can contain NaN (e.g. from a prior seqLen=16 plan's buffers),
  // causing rsqrt(sqAcc+eps) = rsqrt(NaN+eps) = NaN in the output.
  // The mask zeros OOB contributions to sqAcc and the matmul accumulator.
  auto mBoundConst = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto mBoundSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mBoundConst);
  auto i1BmType = mlir::RankedTensorType::get({blockM}, i1Type);
  // mRowMask[i] = (mIndices[i] < M) — true for in-bounds rows
  auto mRowMask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, mIndices.getResult(), mBoundSplat.getResult());

  // Two accumulators:
  // 1. matmulAcc[blockM, blockN] — accumulates (x*gamma) @ W
  // 2. sqAcc[blockM] — accumulates Σ x² per row (for RMS denominator)
  auto accType = mlir::RankedTensorType::get({blockM, blockN}, f32Type);
  auto sqAccType = mlir::RankedTensorType::get({blockM}, f32Type);
  auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 0.0));
  auto matmulAccInit = builder.create<mlir::triton::SplatOp>(loc, accType, zeroF);
  auto sqAccInit = builder.create<mlir::triton::SplatOp>(loc, sqAccType, zeroF);

  // K-loop bounds
  auto kStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto kEnd = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto kStep = builder.create<mlir::arith::ConstantIntOp>(loc, blockK, 32);

  // K-loop with TWO carried accumulators
  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, kStart, kEnd, kStep,
      mlir::ValueRange{matmulAccInit.getResult(), sqAccInit.getResult()});

  builder.setInsertionPointToStart(forOp.getBody());
  auto kIdxI32 = forOp.getInductionVar();
  auto matmulAccIter = forOp.getBody()->getArgument(1);
  auto sqAccIter = forOp.getBody()->getArgument(2);

  auto splatKOffset = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kIdxI32);
  auto kIndices = builder.create<mlir::arith::AddIOp>(loc, splatKOffset, rangeK);

  // ── Load x tile [blockM, blockK] ──
  auto kConst = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto mExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);
  auto kExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 0);
  auto i32BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i32Type);
  auto kSplat = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockM, 1}, i32Type), kConst);
  auto mTimesK = builder.create<mlir::arith::MulIOp>(loc, mExpanded, kSplat);
  auto mTimesKBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, mTimesK);
  auto kBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, kExpanded);
  auto xOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesKBroadcast, kBroadcast);

  auto xPtrTensorType = mlir::RankedTensorType::get({blockM, blockK},
      mlir::triton::PointerType::get(xElemType, 1));
  auto xSplat = builder.create<mlir::triton::SplatOp>(loc, xPtrTensorType, xPtr);
  auto xPtrs = builder.create<mlir::triton::AddPtrOp>(loc, xPtrTensorType, xSplat, xOffsets);
  // Expand row mask [blockM] → [blockM, 1] → [blockM, blockK] for masked x load.
  // OOB rows (mIndices >= M) load zeros so they don't corrupt sqAcc or matmulAcc.
  auto i1BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i1Type);
  auto mRowMaskExpX = builder.create<mlir::triton::ExpandDimsOp>(loc, mRowMask, 1);
  auto mRowMask2DX = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkType, mRowMaskExpX);
  auto xZeroScalar = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(xElemType, 0.0));
  auto xZeroTile = builder.create<mlir::triton::SplatOp>(
      loc, mlir::RankedTensorType::get({blockM, blockK}, xElemType), xZeroScalar);
  auto xLoaded = builder.create<mlir::triton::LoadOp>(loc,
      xPtrs.getResult(), mRowMask2DX.getResult(), xZeroTile.getResult(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // ── Load gamma tile [blockK] and broadcast to [blockM, blockK] ──
  auto gammaPtrTensorType = mlir::RankedTensorType::get({blockK},
      mlir::triton::PointerType::get(gammaElemType, 1));
  auto gammaSplat = builder.create<mlir::triton::SplatOp>(loc, gammaPtrTensorType, gammaPtr);
  auto gammaPtrs = builder.create<mlir::triton::AddPtrOp>(loc, gammaPtrTensorType, gammaSplat, kIndices);
  auto gammaLoaded = builder.create<mlir::triton::LoadOp>(loc,
      gammaPtrs.getResult(), /*mask=*/mlir::Value(), /*other=*/mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  // Broadcast gamma [blockK] → [blockM, blockK]
  auto gammaExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, gammaLoaded, 0);
  auto gammaTileType = mlir::RankedTensorType::get({blockM, blockK}, gammaElemType);
  auto gammaBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, gammaTileType, gammaExpanded);

  // ── Compute x_scaled = x * gamma [blockM, blockK] ──
  // MulFOp requires both operands to have the same type.
  // If gamma is f16 (FP16 weight quantization), extend to f32 to preserve precision.
  mlir::Value gammaForMul = gammaBroadcast.getResult();
  if (gammaElemType != xElemType) {
    if (mlir::isa<mlir::Float32Type>(xElemType) && !mlir::isa<mlir::Float32Type>(gammaElemType)) {
      auto gammaCastType = mlir::RankedTensorType::get({blockM, blockK}, xElemType);
      gammaForMul = builder.create<mlir::arith::ExtFOp>(loc, gammaCastType, gammaForMul);
    } else if (!mlir::isa<mlir::Float32Type>(xElemType) && mlir::isa<mlir::Float32Type>(gammaElemType)) {
      auto gammaCastType = mlir::RankedTensorType::get({blockM, blockK}, xElemType);
      gammaForMul = builder.create<mlir::arith::TruncFOp>(loc, gammaCastType, gammaForMul);
    }
  }
  auto xScaled = builder.create<mlir::arith::MulFOp>(loc, xLoaded, gammaForMul);

  // ── Load W tile [blockK, blockN] ──
  auto nConst = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto kExpandedW = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 1);
  auto nExpandedW = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);
  auto i32BkBnType = mlir::RankedTensorType::get({blockK, blockN}, i32Type);
  auto nSplatW = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockK, 1}, i32Type), nConst);
  auto kTimesN = builder.create<mlir::arith::MulIOp>(loc, kExpandedW, nSplatW);
  auto kTimesNBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, kTimesN);
  auto nBroadcastW = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, nExpandedW);
  auto wOffsets = builder.create<mlir::arith::AddIOp>(loc, kTimesNBroadcast, nBroadcastW);

  auto wPtrTensorType = mlir::RankedTensorType::get({blockK, blockN},
      mlir::triton::PointerType::get(wElemType, 1));
  auto wSplat = builder.create<mlir::triton::SplatOp>(loc, wPtrTensorType, wPtr);
  auto wPtrs = builder.create<mlir::triton::AddPtrOp>(loc, wPtrTensorType, wSplat, wOffsets);
  auto wLoaded = builder.create<mlir::triton::LoadOp>(loc,
      wPtrs.getResult(), /*mask=*/mlir::Value(), /*other=*/mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // ── Accumulate matmul: acc += (x * gamma) @ W ──
  // tt.dot requires both inputs to have same element type.
  // xScaled is xElemType (f32), wLoaded is wElemType (f16 if quantized).
  // Truncate to the narrower type for tensor core compatibility.
  mlir::Value xForDot = xScaled.getResult();
  if (xElemType != wElemType) {
    if (mlir::isa<mlir::Float32Type>(xElemType) && !mlir::isa<mlir::Float32Type>(wElemType)) {
      auto xCastType = mlir::RankedTensorType::get({blockM, blockK}, wElemType);
      xForDot = builder.create<mlir::arith::TruncFOp>(loc, xCastType, xForDot);
    }
  }
  auto dotResult = builder.create<mlir::triton::DotOp>(
      loc, accType, xForDot, wLoaded, matmulAccIter,
      dotPrecision, /*maxNumImpreciseAcc=*/0);

  // ── Accumulate Σx² per row: sqAcc += sum_k(x * x) ──
  // x*x element-wise [blockM, blockK], then reduce over K dimension
  auto xSquared = builder.create<mlir::arith::MulFOp>(loc, xLoaded, xLoaded);
  // Cast to f32 if needed for accumulation
  mlir::Value xSqF32 = xSquared.getResult();
  if (!mlir::isa<mlir::Float32Type>(xElemType)) {
    auto f32BmBkType = mlir::RankedTensorType::get({blockM, blockK}, f32Type);
    xSqF32 = builder.create<mlir::arith::ExtFOp>(loc, f32BmBkType, xSqF32);
  }
  // Reduce over K: sum each row → [blockM]
  // Use tt.reduce with add combiner
  auto reduceOp = builder.create<mlir::triton::ReduceOp>(
      loc, mlir::ValueRange{xSqF32}, /*axis=*/1);
  {
    auto& reduceBody = reduceOp.getRegion().emplaceBlock();
    reduceBody.addArgument(f32Type, loc);
    reduceBody.addArgument(f32Type, loc);
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&reduceBody);
    auto addResult = builder.create<mlir::arith::AddFOp>(loc,
        reduceBody.getArgument(0), reduceBody.getArgument(1));
    builder.create<mlir::triton::ReduceReturnOp>(loc, mlir::ValueRange{addResult});
  }
  auto kSumSq = reduceOp.getResults()[0];  // [blockM]
  auto newSqAcc = builder.create<mlir::arith::AddFOp>(loc, sqAccIter, kSumSq);

  // Yield both accumulators
  mlir::Value yieldVals[] = {dotResult.getResult(), newSqAcc.getResult()};
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange(yieldVals));

  // ── After K-loop: compute rsqrt(mean_sq + eps) and scale the matmul result ──
  builder.setInsertionPointAfter(forOp);
  auto finalMatmulAcc = forOp.getResult(0);  // [blockM, blockN]
  auto finalSqAcc = forOp.getResult(1);       // [blockM]

  // mean_sq = sqAcc / K
  auto kF32 = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, static_cast<float>(K)));
  auto kF32Splat = builder.create<mlir::triton::SplatOp>(loc, sqAccType, kF32);
  auto meanSq = builder.create<mlir::arith::DivFOp>(loc, finalSqAcc, kF32Splat);

  // rsqrt(mean_sq + eps)
  auto epsConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, epsilon));
  auto epsSplat = builder.create<mlir::triton::SplatOp>(loc, sqAccType, epsConst);
  auto meanSqEps = builder.create<mlir::arith::AddFOp>(loc, meanSq, epsSplat);
  auto rsqrtVal = builder.create<mlir::math::RsqrtOp>(loc, meanSqEps);  // [blockM]

  // Broadcast rsqrt [blockM] → [blockM, blockN] and multiply with matmul accumulator
  auto rsqrtExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, rsqrtVal, 1);  // [blockM, 1]
  auto rsqrtBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, accType, rsqrtExpanded);
  auto scaledResult = builder.create<mlir::arith::MulFOp>(loc, finalMatmulAcc, rsqrtBroadcast);

  // Cast and store
  mlir::Value storeVal = scaledResult;
  if (outElemType != f32Type) {
    auto outTileType = mlir::RankedTensorType::get({blockM, blockN}, outElemType);
    if (mlir::isa<mlir::FloatType>(outElemType)) {
      storeVal = builder.create<mlir::arith::TruncFOp>(loc, outTileType, scaledResult);
    }
  }

  // Store to output
  auto mExpandedOut = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);
  auto nExpandedOut = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);
  auto i32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i32Type);
  auto nSplatOut = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockM, 1}, i32Type), nConst);
  auto mTimesN = builder.create<mlir::arith::MulIOp>(loc, mExpandedOut, nSplatOut);
  auto mTimesNBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, mTimesN);
  auto nBroadcastOut = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, nExpandedOut);
  auto outOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesNBroadcast, nBroadcastOut);

  auto outPtrTensorType = mlir::RankedTensorType::get({blockM, blockN},
      mlir::triton::PointerType::get(outElemType, 1));
  auto outSplat = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, outSplat, outOffsets);

  // Masked output store: only write rows where mIndices[i] < M.
  // Without the mask, rows blockM-1..M+1 write beyond the output allocation,
  // corrupting adjacent GPU allocations and causing NaN on subsequent executes.
  auto i1BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i1Type);
  auto mRowMaskExpOut = builder.create<mlir::triton::ExpandDimsOp>(loc, mRowMask, 1);
  auto mRowMask2DOut = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, mRowMaskExpOut);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mRowMask2DOut.getResult(),
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  DSP_DIAG(JIT, "TritonIRBuilder: emitted RmsNormLinear single-pass kernel M=%d N=%d K=%d eps=%f",
            M, N, K, epsilon);
}

// ─── Fused GatedMLP (concatenated GEMM + SiLU + mul) ────────────────────────
//
// GatedMLP: out = silu(x @ W_gate) * (x @ W_up)
// Optimization: concatenate W_concat = [W_gate | W_up] along N dimension → [K, 2N]
// Single wider matmul: concat_out = x @ W_concat → [M, 2N]
// Split: gate = concat_out[:, :N], up = concat_out[:, N:]
// Apply: out = silu(gate) * up
//
// One read of x (instead of two), one wider tt.dot, split+activation in epilogue.
// Grid: 2D, (ceil(M/blockM), ceil(N/blockN)) — each block computes both gate[j] and up[j].

void TritonIRBuilder::emitGatedMLPKernel(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::Value xPtr, mlir::Value wGatePtr, mlir::Value wUpPtr, mlir::Value outPtr,
    int M, int N, int K,
    int blockM, int blockN, int blockK) {

  DSP_DIAG(JIT, "emitGatedMLPKernel: M=%d N=%d K=%d block=(%d,%d,%d)", M, N, K, blockM, blockN, blockK);
  auto f32Type = builder.getF32Type();
  auto i32Type = builder.getI32Type();

  auto xPtrType = mlir::cast<mlir::triton::PointerType>(xPtr.getType());
  auto xElemType = xPtrType.getPointeeType();
  auto wPtrType = mlir::cast<mlir::triton::PointerType>(wGatePtr.getType());
  auto wElemType = wPtrType.getPointeeType();
  auto outPtrTypeVal = mlir::cast<mlir::triton::PointerType>(outPtr.getType());
  auto outElemType = outPtrTypeVal.getPointeeType();

  bool useTf32 = mlir::isa<mlir::Float32Type>(xElemType) &&
                 sd::Environment::getInstance().tritonTf32Enabled();
  auto dotPrecision = useTf32 ? mlir::triton::InputPrecision::TF32
                               : mlir::triton::InputPrecision::IEEE;

  // Program IDs
  auto pidM = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);
  auto pidN = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::Y);

  auto blockMConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockM, 32);
  auto blockNConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockN, 32);
  auto mOffset = builder.create<mlir::arith::MulIOp>(loc, pidM, blockMConst);
  auto nOffset = builder.create<mlir::arith::MulIOp>(loc, pidN, blockNConst);

  auto i32BmType = mlir::RankedTensorType::get({blockM}, i32Type);
  auto i32BnType = mlir::RankedTensorType::get({blockN}, i32Type);
  auto i32BkType = mlir::RankedTensorType::get({blockK}, i32Type);

  auto rangeM = builder.create<mlir::triton::MakeRangeOp>(loc, i32BmType, 0, blockM);
  auto rangeN = builder.create<mlir::triton::MakeRangeOp>(loc, i32BnType, 0, blockN);
  auto rangeK = builder.create<mlir::triton::MakeRangeOp>(loc, i32BkType, 0, blockK);

  auto splatMOffset = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mOffset);
  auto mIndices = builder.create<mlir::arith::AddIOp>(loc, splatMOffset, rangeM);
  auto splatNOffset = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nOffset);
  auto nIndices = builder.create<mlir::arith::AddIOp>(loc, splatNOffset, rangeN);

  // TWO accumulators: gate [blockM, blockN] and up [blockM, blockN]
  auto accType = mlir::RankedTensorType::get({blockM, blockN}, f32Type);
  auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 0.0));
  auto gateAccInit = builder.create<mlir::triton::SplatOp>(loc, accType, zeroF);
  auto upAccInit = builder.create<mlir::triton::SplatOp>(loc, accType, zeroF);

  auto kStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto kEnd = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto kStep = builder.create<mlir::arith::ConstantIntOp>(loc, blockK, 32);

  // K-loop with two accumulators: gate and up
  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, kStart, kEnd, kStep,
      mlir::ValueRange{gateAccInit.getResult(), upAccInit.getResult()});

  builder.setInsertionPointToStart(forOp.getBody());
  auto kIdxI32 = forOp.getInductionVar();
  auto gateAccIter = forOp.getBody()->getArgument(1);
  auto upAccIter = forOp.getBody()->getArgument(2);

  auto splatKOffset = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kIdxI32);
  auto kIndices = builder.create<mlir::arith::AddIOp>(loc, splatKOffset, rangeK);

  // Load x tile [blockM, blockK] — loaded ONCE, used for BOTH gate and up GEMMs
  auto kConst = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto mExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);
  auto kExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 0);
  auto i32BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i32Type);
  auto kSplat = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockM, 1}, i32Type), kConst);
  auto mTimesK = builder.create<mlir::arith::MulIOp>(loc, mExpanded, kSplat);
  auto mTimesKBcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, mTimesK);
  auto kBcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, kExpanded);
  auto xOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesKBcast, kBcast);

  auto xPtrTensorType = mlir::RankedTensorType::get({blockM, blockK},
      mlir::triton::PointerType::get(xElemType, 1));
  auto xSplat = builder.create<mlir::triton::SplatOp>(loc, xPtrTensorType, xPtr);
  auto xPtrs = builder.create<mlir::triton::AddPtrOp>(loc, xPtrTensorType, xSplat, xOffsets);
  // Boundary mask for x tile [blockM, blockK]: mIndices < M && kIndices < K.
  // Without this, decode (M=1 < blockM) reads OOB rows of x -> illegal memory access.
  // Mirrors emitMatmulKernel's A-tile masking. Garbage in masked rows is discarded
  // by the masked output store below, so `other` is left default (matches matmul).
  auto i1TypeXm = builder.getI1Type();
  auto mConstXm = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto mConstSplatXm = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConstXm);
  auto mMask1Dxm = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      mIndices, mConstSplatXm);
  auto kConstSplatXm = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kConst);
  auto kMask1Dxm = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      kIndices, kConstSplatXm);
  auto i1BmBkTypeXm = mlir::RankedTensorType::get({blockM, blockK}, i1TypeXm);
  auto mMaskExpXm = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1Dxm, 1);
  auto kMaskExpXm = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1Dxm, 0);
  auto mMask2Dxm = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkTypeXm, mMaskExpXm);
  auto kMask2Dxm = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkTypeXm, kMaskExpXm);
  auto xMask = builder.create<mlir::arith::AndIOp>(loc, mMask2Dxm, kMask2Dxm);
  auto xLoaded = builder.create<mlir::triton::LoadOp>(loc,
      xPtrs.getResult(), /*mask=*/xMask.getResult(), /*other=*/mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Load W_gate tile [blockK, blockN]
  auto nConst = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto kExpandedW = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 1);
  auto nExpandedW = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);
  auto i32BkBnType = mlir::RankedTensorType::get({blockK, blockN}, i32Type);
  auto nSplatW = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockK, 1}, i32Type), nConst);
  auto kTimesN = builder.create<mlir::arith::MulIOp>(loc, kExpandedW, nSplatW);
  auto kTimesNBcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, kTimesN);
  auto nBcastW = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, nExpandedW);
  auto wGateOffsets = builder.create<mlir::arith::AddIOp>(loc, kTimesNBcast, nBcastW);

  auto wPtrTensorType = mlir::RankedTensorType::get({blockK, blockN},
      mlir::triton::PointerType::get(wElemType, 1));
  auto wGateSplat = builder.create<mlir::triton::SplatOp>(loc, wPtrTensorType, wGatePtr);
  auto wGatePtrs = builder.create<mlir::triton::AddPtrOp>(loc, wPtrTensorType, wGateSplat, wGateOffsets);
  auto wGateLoaded = builder.create<mlir::triton::LoadOp>(loc,
      wGatePtrs.getResult(), /*mask=*/mlir::Value(), /*other=*/mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Load W_up tile [blockK, blockN] — SAME offsets, different base pointer
  auto wUpSplat = builder.create<mlir::triton::SplatOp>(loc, wPtrTensorType, wUpPtr);
  auto wUpPtrs = builder.create<mlir::triton::AddPtrOp>(loc, wPtrTensorType, wUpSplat, wGateOffsets);
  auto wUpLoaded = builder.create<mlir::triton::LoadOp>(loc,
      wUpPtrs.getResult(), /*mask=*/mlir::Value(), /*other=*/mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // gate_acc += x @ W_gate (reuses same x tile)
  // tt.dot requires both inputs to have same element type.
  // If x is f32 and weights are f16 (FP16 quantization), truncate x to match.
  mlir::Value xForDot = xLoaded.getResult();
  if (xElemType != wElemType) {
    if (mlir::isa<mlir::Float32Type>(xElemType) && !mlir::isa<mlir::Float32Type>(wElemType)) {
      auto xCastType = mlir::RankedTensorType::get({blockM, blockK}, wElemType);
      xForDot = builder.create<mlir::arith::TruncFOp>(loc, xCastType, xForDot);
    }
  }
  auto gateResult = builder.create<mlir::triton::DotOp>(
      loc, accType, xForDot, wGateLoaded, gateAccIter,
      dotPrecision, /*maxNumImpreciseAcc=*/0);

  // up_acc += x @ W_up (reuses same x tile — ONE read, TWO tt.dots)
  auto upResult = builder.create<mlir::triton::DotOp>(
      loc, accType, xForDot, wUpLoaded, upAccIter,
      dotPrecision, /*maxNumImpreciseAcc=*/0);

  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{gateResult, upResult});

  // After K-loop: apply silu(gate) * up in registers
  builder.setInsertionPointAfter(forOp);
  auto finalGate = forOp.getResult(0);  // [blockM, blockN] f32
  auto finalUp = forOp.getResult(1);    // [blockM, blockN] f32

  // SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
  auto negGate = builder.create<mlir::arith::NegFOp>(loc, finalGate);
  auto expNegGate = builder.create<mlir::math::ExpOp>(loc, negGate);
  auto oneT = builder.create<mlir::triton::SplatOp>(loc, accType,
      builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 1.0)));
  auto denom = builder.create<mlir::arith::AddFOp>(loc, oneT, expNegGate);
  auto sigmoid = builder.create<mlir::arith::DivFOp>(loc, oneT, denom);
  auto siluGate = builder.create<mlir::arith::MulFOp>(loc, finalGate, sigmoid);

  // output = silu(gate) * up — still in registers
  auto fusedResult = builder.create<mlir::arith::MulFOp>(loc, siluGate, finalUp);

  // Cast and store
  mlir::Value storeVal = fusedResult;
  if (outElemType != f32Type) {
    auto outTileType = mlir::RankedTensorType::get({blockM, blockN}, outElemType);
    if (mlir::isa<mlir::FloatType>(outElemType)) {
      storeVal = builder.create<mlir::arith::TruncFOp>(loc, outTileType, fusedResult);
    }
  }

  // Store output
  auto mExpandedOut = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);
  auto nExpandedOut = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);
  auto i32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i32Type);
  auto nSplatOut = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockM, 1}, i32Type), nConst);
  auto mTimesNOut = builder.create<mlir::arith::MulIOp>(loc, mExpandedOut, nSplatOut);
  auto mTimesNBcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, mTimesNOut);
  auto nBcastOut = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, nExpandedOut);
  auto outOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesNBcast, nBcastOut);

  auto outPtrTensorType = mlir::RankedTensorType::get({blockM, blockN},
      mlir::triton::PointerType::get(outElemType, 1));
  auto outSplat = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, outSplat, outOffsets);

  // Boundary mask for output tile [blockM, blockN]: mIndices < M && nIndices < N.
  // Prevents OOB WRITES (memory corruption) when M < blockM (decode) or N not
  // tile-aligned, and discards the garbage rows produced from masked-off x rows.
  // mIndices/nIndices/i32Bm/BnType are defined pre-K-loop so they dominate here;
  // fresh M/N constants are created (the in-loop ones do not dominate after the loop).
  auto i1TypeOm = builder.getI1Type();
  auto mConstOm = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto nConstOm = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto mConstSplatOm = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConstOm);
  auto nConstSplatOm = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nConstOm);
  auto mMask1Dom = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      mIndices, mConstSplatOm);
  auto nMask1Dom = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      nIndices, nConstSplatOm);
  auto i1BmBnTypeOm = mlir::RankedTensorType::get({blockM, blockN}, i1TypeOm);
  auto mMaskExpOm = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1Dom, 1);
  auto nMaskExpOm = builder.create<mlir::triton::ExpandDimsOp>(loc, nMask1Dom, 0);
  auto mMask2Dom = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnTypeOm, mMaskExpOm);
  auto nMask2Dom = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnTypeOm, nMaskExpOm);
  auto outMask = builder.create<mlir::arith::AndIOp>(loc, mMask2Dom, nMask2Dom);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, /*mask=*/outMask.getResult(),
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  DSP_DIAG(JIT, "TritonIRBuilder: emitted GatedMLP kernel M=%d N=%d K=%d (x loaded once, 2x tt.dot)",
            M, N, K);
}

// ─── Fused Two-Layer MLP emission (FastVLA pattern) ─────────────────────────
//
// Computes: out = tanh(ReLU(x @ W1 + b1) @ W2 + b2)
//
// Key insight (from FastVLA _action_fwd_kernel): tile over H (hidden dimension)
// so the intermediate activation h1 = ReLU(x @ W1 + b1) stays in registers.
// For each H-tile:
//   1. Inner K-loop: accumulate x @ W1_tile (D tiles, blockK at a time)
//   2. Add b1 tile, apply ReLU — still in registers
//   3. Multiply h1_tile @ W2_tile, accumulate into output
// After all H-tiles: add b2, apply tanh.
//
// Memory traffic: reads x once per H-tile, W1 once total, W2 once total.
// The intermediate [M, H] activation never touches global memory.
//
// Grid: (ceil(M/blockM), ceil(A/blockA)) — 2D
// Inputs:  x [M, D], W1 [D, H], b1 [H], W2 [H, A], b2 [A]
// Output:  out [M, A]

void TritonIRBuilder::emitFusedTwoLayerMLPKernel(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::Value xPtr, mlir::Value w1Ptr, mlir::Value b1Ptr,
    mlir::Value w2Ptr, mlir::Value b2Ptr, mlir::Value outPtr,
    int M, int D, int H, int A,
    int blockM, int blockH, int blockK, int blockA) {

  DSP_DIAG(JIT, "emitFusedTwoLayerMLPKernel: M=%d D=%d H=%d A=%d block=(%d,%d,%d,%d)",
           M, D, H, A, blockM, blockH, blockK, blockA);

  auto f32Type = builder.getF32Type();
  auto i32Type = builder.getI32Type();

  auto xPtrType = mlir::cast<mlir::triton::PointerType>(xPtr.getType());
  auto xElemType = xPtrType.getPointeeType();
  auto w1PtrType = mlir::cast<mlir::triton::PointerType>(w1Ptr.getType());
  auto w1ElemType = w1PtrType.getPointeeType();
  auto w2PtrType = mlir::cast<mlir::triton::PointerType>(w2Ptr.getType());
  auto w2ElemType = w2PtrType.getPointeeType();
  auto outPtrTypeVal = mlir::cast<mlir::triton::PointerType>(outPtr.getType());
  auto outElemType = outPtrTypeVal.getPointeeType();

  bool useTf32 = mlir::isa<mlir::Float32Type>(xElemType) &&
                 sd::Environment::getInstance().tritonTf32Enabled();
  auto dotPrecision = useTf32 ? mlir::triton::InputPrecision::TF32
                               : mlir::triton::InputPrecision::IEEE;

  // Program IDs: 2D grid (M tiles, A tiles)
  auto pidM = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);
  auto pidA = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::Y);

  auto blockMConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockM, 32);
  auto blockAConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockA, 32);
  auto mOffset = builder.create<mlir::arith::MulIOp>(loc, pidM, blockMConst);
  auto aOffset = builder.create<mlir::arith::MulIOp>(loc, pidA, blockAConst);

  auto i32BmType = mlir::RankedTensorType::get({blockM}, i32Type);
  auto i32BaType = mlir::RankedTensorType::get({blockA}, i32Type);
  auto i32BhType = mlir::RankedTensorType::get({blockH}, i32Type);
  auto i32BkType = mlir::RankedTensorType::get({blockK}, i32Type);

  auto rangeM = builder.create<mlir::triton::MakeRangeOp>(loc, i32BmType, 0, blockM);
  auto rangeA = builder.create<mlir::triton::MakeRangeOp>(loc, i32BaType, 0, blockA);

  auto splatMOffset = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mOffset);
  auto mIndices = builder.create<mlir::arith::AddIOp>(loc, splatMOffset, rangeM);
  auto splatAOffset = builder.create<mlir::triton::SplatOp>(loc, i32BaType, aOffset);
  auto aIndices = builder.create<mlir::arith::AddIOp>(loc, splatAOffset, rangeA);

  // Output accumulator: [blockM, blockA] in f32
  auto accType = mlir::RankedTensorType::get({blockM, blockA}, f32Type);
  auto zeroF = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 0.0));
  auto outAccInit = builder.create<mlir::triton::SplatOp>(loc, accType, zeroF);

  // ── Outer loop over H tiles ──
  // For each H-tile, we compute h1_tile = ReLU(x @ W1_tile + b1_tile),
  // then accumulate out_acc += h1_tile @ W2_tile. The intermediate h1_tile
  // stays in registers across both matmuls — zero global memory traffic.
  auto hStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto hEnd = builder.create<mlir::arith::ConstantIntOp>(loc, H, 32);
  auto hStep = builder.create<mlir::arith::ConstantIntOp>(loc, blockH, 32);

  auto hLoop = builder.create<mlir::scf::ForOp>(
      loc, hStart, hEnd, hStep,
      mlir::ValueRange{outAccInit.getResult()});

  builder.setInsertionPointToStart(hLoop.getBody());
  auto hIdx = hLoop.getInductionVar();
  auto outAccIter = hLoop.getBody()->getArgument(1);

  // H-tile indices
  auto splatHOffset = builder.create<mlir::triton::SplatOp>(loc, i32BhType, hIdx);
  auto rangeH = builder.create<mlir::triton::MakeRangeOp>(loc, i32BhType, 0, blockH);
  auto hIndices = builder.create<mlir::arith::AddIOp>(loc, splatHOffset, rangeH);

  // ── Inner K-loop: accumulate h1_tile = x @ W1_tile ──
  // x is [M, D], W1 is [D, H]. For this H-tile, we compute the
  // [blockM, blockH] matmul by tiling over D in blockK chunks.
  auto h1Type = mlir::RankedTensorType::get({blockM, blockH}, f32Type);
  auto h1Init = builder.create<mlir::triton::SplatOp>(loc, h1Type, zeroF);

  auto kStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto kEnd = builder.create<mlir::arith::ConstantIntOp>(loc, D, 32);
  auto kStep = builder.create<mlir::arith::ConstantIntOp>(loc, blockK, 32);

  auto kLoop = builder.create<mlir::scf::ForOp>(
      loc, kStart, kEnd, kStep,
      mlir::ValueRange{h1Init.getResult()});

  builder.setInsertionPointToStart(kLoop.getBody());
  auto kIdx = kLoop.getInductionVar();
  auto h1AccIter = kLoop.getBody()->getArgument(1);

  // K-tile indices
  auto splatKOffset = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kIdx);
  auto rangeK = builder.create<mlir::triton::MakeRangeOp>(loc, i32BkType, 0, blockK);
  auto kIndices = builder.create<mlir::arith::AddIOp>(loc, splatKOffset, rangeK);

  // Load x tile [blockM, blockK]: x[m, k] at offset m*D + k
  auto dConst = builder.create<mlir::arith::ConstantIntOp>(loc, D, 32);
  auto mExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);
  auto kExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 0);
  auto i32BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i32Type);
  auto dSplat = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockM, 1}, i32Type), dConst);
  auto mTimesD = builder.create<mlir::arith::MulIOp>(loc, mExpanded, dSplat);
  auto mTimesDBC = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, mTimesD);
  auto kBC = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, kExpanded);
  auto xOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesDBC, kBC);

  auto xPtrTensorType = mlir::RankedTensorType::get({blockM, blockK},
      mlir::triton::PointerType::get(xElemType, 1));
  auto xSplat = builder.create<mlir::triton::SplatOp>(loc, xPtrTensorType, xPtr);
  auto xPtrs = builder.create<mlir::triton::AddPtrOp>(loc, xPtrTensorType, xSplat, xOffsets);
  // Boundary mask for x tile [blockM, blockK]: mIndices < M && kIndices < D.
  // Without this, decode (M=1 < blockM) reads OOB rows of x -> illegal memory access.
  // Garbage in masked rows is discarded by the masked output store below.
  auto i1TypeXm2 = builder.getI1Type();
  auto mConstXm2 = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto mConstSplatXm2 = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConstXm2);
  auto mMask1Dxm2 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      mIndices, mConstSplatXm2);
  auto dConstSplatXm2 = builder.create<mlir::triton::SplatOp>(loc, i32BkType, dConst);
  auto kMask1Dxm2 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      kIndices, dConstSplatXm2);
  auto i1BmBkTypeXm2 = mlir::RankedTensorType::get({blockM, blockK}, i1TypeXm2);
  auto mMaskExpXm2 = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1Dxm2, 1);
  auto kMaskExpXm2 = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1Dxm2, 0);
  auto mMask2Dxm2 = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkTypeXm2, mMaskExpXm2);
  auto kMask2Dxm2 = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkTypeXm2, kMaskExpXm2);
  auto xMask2 = builder.create<mlir::arith::AndIOp>(loc, mMask2Dxm2, kMask2Dxm2);
  auto xLoaded = builder.create<mlir::triton::LoadOp>(loc,
      xPtrs.getResult(), /*mask=*/xMask2.getResult(), /*other=*/mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Load W1 tile [blockK, blockH]: W1[k, h] at offset k*H + h
  auto hConst = builder.create<mlir::arith::ConstantIntOp>(loc, H, 32);
  auto kExpandedW1 = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 1);
  auto hExpandedW1 = builder.create<mlir::triton::ExpandDimsOp>(loc, hIndices, 0);
  auto i32BkBhType = mlir::RankedTensorType::get({blockK, blockH}, i32Type);
  auto hSplatW1 = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockK, 1}, i32Type), hConst);
  auto kTimesH = builder.create<mlir::arith::MulIOp>(loc, kExpandedW1, hSplatW1);
  auto kTimesHBC = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBhType, kTimesH);
  auto hBCW1 = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBhType, hExpandedW1);
  auto w1Offsets = builder.create<mlir::arith::AddIOp>(loc, kTimesHBC, hBCW1);

  auto w1PtrTensorType = mlir::RankedTensorType::get({blockK, blockH},
      mlir::triton::PointerType::get(w1ElemType, 1));
  auto w1Splat = builder.create<mlir::triton::SplatOp>(loc, w1PtrTensorType, w1Ptr);
  auto w1Ptrs = builder.create<mlir::triton::AddPtrOp>(loc, w1PtrTensorType, w1Splat, w1Offsets);
  auto w1Loaded = builder.create<mlir::triton::LoadOp>(loc,
      w1Ptrs.getResult(), /*mask=*/mlir::Value(), /*other=*/mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // h1_acc += x @ W1_tile
  auto h1Result = builder.create<mlir::triton::DotOp>(
      loc, h1Type, xLoaded, w1Loaded, h1AccIter,
      dotPrecision, /*maxNumImpreciseAcc=*/0);

  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{h1Result});

  // ── After K-loop: add b1 and apply ReLU ──
  builder.setInsertionPointAfter(kLoop);
  auto h1Final = kLoop.getResult(0);  // [blockM, blockH] f32

  // Load b1 tile [blockH] and broadcast to [blockM, blockH]
  auto b1PtrType = mlir::cast<mlir::triton::PointerType>(b1Ptr.getType());
  auto b1ElemType = b1PtrType.getPointeeType();
  auto b1PtrTensorType = mlir::RankedTensorType::get({blockH},
      mlir::triton::PointerType::get(b1ElemType, 1));
  auto b1Splat = builder.create<mlir::triton::SplatOp>(loc, b1PtrTensorType, b1Ptr);
  auto b1Ptrs = builder.create<mlir::triton::AddPtrOp>(loc, b1PtrTensorType, b1Splat, hIndices);
  auto b1Loaded = builder.create<mlir::triton::LoadOp>(loc,
      b1Ptrs.getResult(), /*mask=*/mlir::Value(), /*other=*/mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Cast b1 to f32 if needed, then broadcast [blockH] -> [blockM, blockH]
  mlir::Value b1F32 = b1Loaded.getResult();
  if (b1ElemType != f32Type) {
    auto b1F32VType = mlir::RankedTensorType::get({blockH}, f32Type);
    b1F32 = builder.create<mlir::arith::ExtFOp>(loc, b1F32VType, b1F32);
  }
  auto b1Expanded = builder.create<mlir::triton::ExpandDimsOp>(loc, b1F32, 0);
  auto b1Broadcast = builder.create<mlir::triton::BroadcastOp>(loc, h1Type, b1Expanded);

  auto h1WithBias = builder.create<mlir::arith::AddFOp>(loc, h1Final, b1Broadcast);

  // ReLU: max(h1 + b1, 0)
  auto zeroSplat = builder.create<mlir::triton::SplatOp>(loc, h1Type, zeroF);
  auto h1Relu = builder.create<mlir::arith::MaximumFOp>(loc, h1WithBias, zeroSplat);

  // ── Multiply h1_relu @ W2_tile and accumulate ──
  // W2 is [H, A]. Load W2 tile [blockH, blockA]: W2[h, a] at offset h*A + a
  auto aConst = builder.create<mlir::arith::ConstantIntOp>(loc, A, 32);
  auto hExpandedW2 = builder.create<mlir::triton::ExpandDimsOp>(loc, hIndices, 1);
  auto aExpandedW2 = builder.create<mlir::triton::ExpandDimsOp>(loc, aIndices, 0);
  auto i32BhBaType = mlir::RankedTensorType::get({blockH, blockA}, i32Type);
  auto aSplatW2 = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockH, 1}, i32Type), aConst);
  auto hTimesA = builder.create<mlir::arith::MulIOp>(loc, hExpandedW2, aSplatW2);
  auto hTimesABC = builder.create<mlir::triton::BroadcastOp>(loc, i32BhBaType, hTimesA);
  auto aBCW2 = builder.create<mlir::triton::BroadcastOp>(loc, i32BhBaType, aExpandedW2);
  auto w2Offsets = builder.create<mlir::arith::AddIOp>(loc, hTimesABC, aBCW2);

  auto w2PtrTensorType = mlir::RankedTensorType::get({blockH, blockA},
      mlir::triton::PointerType::get(w2ElemType, 1));
  auto w2Splat = builder.create<mlir::triton::SplatOp>(loc, w2PtrTensorType, w2Ptr);
  auto w2Ptrs = builder.create<mlir::triton::AddPtrOp>(loc, w2PtrTensorType, w2Splat, w2Offsets);
  auto w2Loaded = builder.create<mlir::triton::LoadOp>(loc,
      w2Ptrs.getResult(), /*mask=*/mlir::Value(), /*other=*/mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Cast h1_relu to match w2 dtype for tt.dot (both inputs must match)
  mlir::Value h1ForDot = h1Relu;
  if (w2ElemType != f32Type) {
    auto h1CastType = mlir::RankedTensorType::get({blockM, blockH}, w2ElemType);
    h1ForDot = builder.create<mlir::arith::TruncFOp>(loc, h1CastType, h1Relu);
  }

  // out_acc += h1_relu @ W2_tile
  auto outAccResult = builder.create<mlir::triton::DotOp>(
      loc, accType, h1ForDot, w2Loaded, outAccIter,
      dotPrecision, /*maxNumImpreciseAcc=*/0);

  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{outAccResult});

  // ── After H-loop: add b2 and apply tanh ──
  builder.setInsertionPointAfter(hLoop);
  auto outFinal = hLoop.getResult(0);  // [blockM, blockA] f32

  // Load b2 tile [blockA] and broadcast to [blockM, blockA]
  auto b2PtrType = mlir::cast<mlir::triton::PointerType>(b2Ptr.getType());
  auto b2ElemType = b2PtrType.getPointeeType();
  auto b2PtrTensorType = mlir::RankedTensorType::get({blockA},
      mlir::triton::PointerType::get(b2ElemType, 1));
  auto b2Splat = builder.create<mlir::triton::SplatOp>(loc, b2PtrTensorType, b2Ptr);
  auto b2Ptrs = builder.create<mlir::triton::AddPtrOp>(loc, b2PtrTensorType, b2Splat, aIndices);
  auto b2Loaded = builder.create<mlir::triton::LoadOp>(loc,
      b2Ptrs.getResult(), /*mask=*/mlir::Value(), /*other=*/mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  mlir::Value b2F32 = b2Loaded.getResult();
  if (b2ElemType != f32Type) {
    auto b2F32VType = mlir::RankedTensorType::get({blockA}, f32Type);
    b2F32 = builder.create<mlir::arith::ExtFOp>(loc, b2F32VType, b2F32);
  }
  auto b2Expanded = builder.create<mlir::triton::ExpandDimsOp>(loc, b2F32, 0);
  auto b2Broadcast = builder.create<mlir::triton::BroadcastOp>(loc, accType, b2Expanded);

  auto outWithBias = builder.create<mlir::arith::AddFOp>(loc, outFinal, b2Broadcast);

  // Tanh via numerically stable sigmoid: tanh(x) = 2 * sigmoid(2x) - 1
  // This matches FastVLA's approach and avoids precision issues with direct tanh.
  auto twoConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 2.0));
  auto twoSplat = builder.create<mlir::triton::SplatOp>(loc, accType, twoConst);
  auto twoX = builder.create<mlir::arith::MulFOp>(loc, twoSplat, outWithBias);
  auto negTwoX = builder.create<mlir::arith::NegFOp>(loc, twoX);
  auto expNegTwoX = builder.create<mlir::math::ExpOp>(loc, negTwoX);
  auto oneConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32Type, 1.0));
  auto oneSplat = builder.create<mlir::triton::SplatOp>(loc, accType, oneConst);
  auto denom = builder.create<mlir::arith::AddFOp>(loc, oneSplat, expNegTwoX);
  auto sigmoid2x = builder.create<mlir::arith::DivFOp>(loc, oneSplat, denom);
  auto twoSigmoid = builder.create<mlir::arith::MulFOp>(loc, twoSplat, sigmoid2x);
  auto tanhResult = builder.create<mlir::arith::SubFOp>(loc, twoSigmoid, oneSplat);

  // Cast and store output
  mlir::Value storeVal = tanhResult;
  if (outElemType != f32Type) {
    auto outTileType = mlir::RankedTensorType::get({blockM, blockA}, outElemType);
    if (mlir::isa<mlir::FloatType>(outElemType)) {
      storeVal = builder.create<mlir::arith::TruncFOp>(loc, outTileType, tanhResult);
    }
  }

  // Store: out[m, a] at offset m*A + a
  auto mExpandedOut = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);
  auto aExpandedOut = builder.create<mlir::triton::ExpandDimsOp>(loc, aIndices, 0);
  auto i32BmBaType = mlir::RankedTensorType::get({blockM, blockA}, i32Type);
  auto aSplatOut = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockM, 1}, i32Type), aConst);
  auto mTimesAOut = builder.create<mlir::arith::MulIOp>(loc, mExpandedOut, aSplatOut);
  auto mTimesABCOut = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBaType, mTimesAOut);
  auto aBCOut = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBaType, aExpandedOut);
  auto outOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesABCOut, aBCOut);

  auto outPtrTensorType = mlir::RankedTensorType::get({blockM, blockA},
      mlir::triton::PointerType::get(outElemType, 1));
  auto outSplat = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, outSplat, outOffsets);

  // Boundary mask for output tile [blockM, blockA]: mIndices < M && aIndices < A.
  // Prevents OOB WRITES (memory corruption) when M < blockM (decode) or A not
  // tile-aligned, and discards garbage rows produced from masked-off x rows.
  auto i1TypeOm2 = builder.getI1Type();
  auto mConstOm2 = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto aConstOm2 = builder.create<mlir::arith::ConstantIntOp>(loc, A, 32);
  auto mConstSplatOm2 = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConstOm2);
  auto aConstSplatOm2 = builder.create<mlir::triton::SplatOp>(loc, i32BaType, aConstOm2);
  auto mMask1Dom2 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      mIndices, mConstSplatOm2);
  auto aMask1Dom2 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      aIndices, aConstSplatOm2);
  auto i1BmBaTypeOm2 = mlir::RankedTensorType::get({blockM, blockA}, i1TypeOm2);
  auto mMaskExpOm2 = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1Dom2, 1);
  auto aMaskExpOm2 = builder.create<mlir::triton::ExpandDimsOp>(loc, aMask1Dom2, 0);
  auto mMask2Dom2 = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBaTypeOm2, mMaskExpOm2);
  auto aMask2Dom2 = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBaTypeOm2, aMaskExpOm2);
  auto outMask2 = builder.create<mlir::arith::AndIOp>(loc, mMask2Dom2, aMask2Dom2);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, /*mask=*/outMask2.getResult(),
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  DSP_DIAG(JIT, "TritonIRBuilder: emitted FusedTwoLayerMLP kernel M=%d D=%d H=%d A=%d "
            "(h1 stays in registers, zero intermediate HBM traffic)", M, D, H, A);
}

// ─── Fused attention (Flash Attention) emission ─────────────────────────────

void TritonIRBuilder::emitFusedAttentionKernel(mlir::OpBuilder& builder, mlir::Location loc,
                                                mlir::Value qPtr, mlir::Value kPtr,
                                                mlir::Value vPtr, mlir::Value outPtr,
                                                int batchSize, int numQHeads, int numKvHeads,
                                                int seqQ, int seqK,
                                                int headDim, float scale,
                                                int blockM, int blockN,
                                                bool qIsBSHD, bool kIsBSHD,
                                                mlir::Value biasPtr,
                                                const std::vector<LongType>& biasShape,
                                                mlir::Value curKPtr, mlir::Value curVPtr,
                                                int pastSeq, int seqKVCur) {
  DSP_DIAG(JIT, "emitFusedAttentionKernel: batch=%d qHeads=%d kvHeads=%d seqQ=%d seqK=%d headDim=%d "
           "block=(%d,%d) dualBuffer=%d", batchSize, numQHeads, numKvHeads, seqQ, seqK, headDim,
           blockM, blockN, curKPtr ? 1 : 0);
  // Dual-buffer mode: when curKPtr is valid, K/V are split across two buffers:
  //   kPtr  = past_key  [B,H,pastSeq,D]   BHSD (positions [0, pastSeq))
  //   curKPtr = current_key [B,seqKVCur,H*D] BSHD (positions [pastSeq, seqK))
  // seqK = pastSeq + seqKVCur (total sequence length for attention)
  bool dualBuffer = curKPtr ? true : false;
  // GQA: numQHeads >= numKvHeads, each KV head serves (numQHeads/numKvHeads) Q heads
  if (numKvHeads <= 0) numKvHeads = numQHeads;
  int kvGroupSize = (numKvHeads > 0) ? (numQHeads / numKvHeads) : 1;
  if (kvGroupSize < 1) kvGroupSize = 1;
  auto f32Type = builder.getF32Type();
  auto i32Type = builder.getI32Type();

  // Triton requires all tensor dimensions (MakeRangeOp) to be power-of-2.
  // Round headDim up and use masking for the padded region.
  int headDimPadded = headDim;
  if (headDimPadded > 0 && (headDimPadded & (headDimPadded - 1)) != 0) {
    int p = 1;
    while (p < headDimPadded) p <<= 1;
    headDimPadded = p;
  }
  bool needsHdMask = (headDimPadded != headDim);

  // Program IDs: pid0 = batch * numQHeads + qHeadIdx, pid1 = query tile index
  auto pid0 = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);
  auto pid1 = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::Y);

  // Decompose pid0 into batch and Q head indices
  auto numQHeadsConst = builder.create<mlir::arith::ConstantIntOp>(loc, numQHeads, 32);
  auto numKvHeadsConst = builder.create<mlir::arith::ConstantIntOp>(loc, numKvHeads, 32);
  auto headIdx = builder.create<mlir::arith::RemSIOp>(loc, pid0, numQHeadsConst);   // Q head index [0, numQHeads)
  auto batchIdx = builder.create<mlir::arith::DivSIOp>(loc, pid0, numQHeadsConst);
  // GQA: map Q head to KV head — kvHeadIdx = headIdx / kvGroupSize
  auto kvGroupSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, kvGroupSize, 32);
  auto kvHeadIdx = builder.create<mlir::arith::DivSIOp>(loc, headIdx, kvGroupSizeConst);

  // Query tile offset
  auto blockMConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockM, 32);
  auto qOffset = builder.create<mlir::arith::MulIOp>(loc, pid1, blockMConst);

  // Create range vectors — use headDimPadded (power-of-2) for tensor sizes
  auto i32BmType = mlir::RankedTensorType::get({blockM}, i32Type);
  auto i32BnType = mlir::RankedTensorType::get({blockN}, i32Type);
  auto i32HdType = mlir::RankedTensorType::get({headDimPadded}, i32Type);

  auto rangeM = builder.create<mlir::triton::MakeRangeOp>(loc, i32BmType, 0, blockM);
  auto rangeN = builder.create<mlir::triton::MakeRangeOp>(loc, i32BnType, 0, blockN);
  auto rangeHd = builder.create<mlir::triton::MakeRangeOp>(loc, i32HdType, 0, headDimPadded);
  mlir::Value rangeHdForPtr = rangeHd;
  if (needsHdMask) {
    auto headDimMinus1 = builder.create<mlir::arith::ConstantIntOp>(loc, std::max(0, headDim - 1), 32);
    auto headDimMinus1Splat = builder.create<mlir::triton::SplatOp>(loc, i32HdType, headDimMinus1);
    rangeHdForPtr = builder.create<mlir::arith::MinSIOp>(loc, rangeHd, headDimMinus1Splat);
  }

  auto splatQOffset = builder.create<mlir::triton::SplatOp>(loc, i32BmType, qOffset);
  auto qIndices = builder.create<mlir::arith::AddIOp>(loc, splatQOffset, rangeM);

  // Compute base offset into Q/K/V/Out buffers.
  // BHSD (4D): [batch, heads, seq, headDim] — base = batch*NH*S*HD + head*S*HD, rowStride=HD
  // BSHD (3D): [batch, seq, NH*HD]         — base = batch*S*NH*HD + head*HD,   rowStride=NH*HD
  auto seqQConst = builder.create<mlir::arith::ConstantIntOp>(loc, seqQ, 32);
  auto seqKConst = builder.create<mlir::arith::ConstantIntOp>(loc, seqK, 32);
  auto headDimConst = builder.create<mlir::arith::ConstantIntOp>(loc, headDim, 32);

  mlir::Value qBase, qRowStride;
  if (qIsBSHD) {
    // BSHD: [batch, seqQ, numQHeads*headDim]
    auto nhTimesHd = builder.create<mlir::arith::MulIOp>(loc, numQHeadsConst, headDimConst);
    auto qStride0 = builder.create<mlir::arith::MulIOp>(loc, seqQConst, nhTimesHd);
    qBase = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::MulIOp>(loc, batchIdx, qStride0),
        builder.create<mlir::arith::MulIOp>(loc, headIdx, headDimConst));
    qRowStride = nhTimesHd;
  } else {
    // BHSD: [batch, numQHeads, seqQ, headDim]
    auto qStride0 = builder.create<mlir::arith::MulIOp>(loc, numQHeadsConst,
        builder.create<mlir::arith::MulIOp>(loc, seqQConst, headDimConst));
    auto qStride1 = builder.create<mlir::arith::MulIOp>(loc, seqQConst, headDimConst);
    qBase = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::MulIOp>(loc, batchIdx, qStride0),
        builder.create<mlir::arith::MulIOp>(loc, headIdx, qStride1));
    qRowStride = headDimConst;
  }

  mlir::Value kvBase, kvRowStride;
  if (kIsBSHD) {
    // BSHD: [batch, seqK, numKvHeads*headDim] — use kvHeadIdx for GQA
    auto kvNhTimesHd = builder.create<mlir::arith::MulIOp>(loc, numKvHeadsConst, headDimConst);
    auto kvStride0 = builder.create<mlir::arith::MulIOp>(loc, seqKConst, kvNhTimesHd);
    kvBase = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::MulIOp>(loc, batchIdx, kvStride0),
        builder.create<mlir::arith::MulIOp>(loc, kvHeadIdx, headDimConst));
    kvRowStride = kvNhTimesHd;
  } else {
    // BHSD: [batch, numKvHeads, actualSeqDim, headDim] — use kvHeadIdx for GQA
    // In dual-buffer mode, the K buffer has pastSeq positions (not total seqK).
    // Using seqK here would shift every KV head's base offset by (seqKVCur * headDim) per head.
    mlir::Value kvSeqDimConst;
    if (dualBuffer) {
      kvSeqDimConst = builder.create<mlir::arith::ConstantIntOp>(loc, pastSeq, 32);
    } else {
      kvSeqDimConst = seqKConst;
    }
    auto kvStride0 = builder.create<mlir::arith::MulIOp>(loc, numKvHeadsConst,
        builder.create<mlir::arith::MulIOp>(loc, kvSeqDimConst, headDimConst));
    auto kvStride1 = builder.create<mlir::arith::MulIOp>(loc, kvSeqDimConst, headDimConst);
    kvBase = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::MulIOp>(loc, batchIdx, kvStride0),
        builder.create<mlir::arith::MulIOp>(loc, kvHeadIdx, kvStride1));
    kvRowStride = headDimConst;
  }

  // Dual-buffer: compute base and stride for current key/value (BSHD layout)
  // current_key [B, seqKVCur, numKvHeads*headDim] = [B, seqKVCur, numKvHeads, headDim] BSHD
  // base = b * seqKVCur * numKvHeads * headDim + kvH * headDim
  // rowStride = numKvHeads * headDim
  mlir::Value curKvBase, curKvRowStride, pastSeqConst;
  if (dualBuffer) {
    auto seqKVCurConst = builder.create<mlir::arith::ConstantIntOp>(loc, seqKVCur, 32);
    pastSeqConst = builder.create<mlir::arith::ConstantIntOp>(loc, pastSeq, 32);
    auto curNhTimesHd = builder.create<mlir::arith::MulIOp>(loc, numKvHeadsConst, headDimConst);
    auto curStride0 = builder.create<mlir::arith::MulIOp>(loc, seqKVCurConst, curNhTimesHd);
    curKvBase = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::MulIOp>(loc, batchIdx, curStride0),
        builder.create<mlir::arith::MulIOp>(loc, kvHeadIdx, headDimConst));
    curKvRowStride = curNhTimesHd;
  }

  // Load Q tile [BLOCK_M, headDim]
  // Q pointer offsets: qBase + qIndicesClamped[:, None] * headDim + clamped(rangeHd)[None, :]
  // Clamp Q indices for pointer computation (same rationale as K clamping above).
  auto seqQMinus1 = builder.create<mlir::arith::ConstantIntOp>(loc, std::max(0, seqQ - 1), 32);
  auto seqQMinus1Splat = builder.create<mlir::triton::SplatOp>(loc, i32BmType, seqQMinus1);
  auto qIndicesClamped = builder.create<mlir::arith::MinSIOp>(loc, qIndices, seqQMinus1Splat);
  auto qMExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, qIndicesClamped, 1);  // [BM, 1]
  auto hdExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, rangeHdForPtr, 0);   // [1, HD]

  auto i32BmHdType = mlir::RankedTensorType::get({blockM, headDimPadded}, i32Type);
  auto f32BmHdType = mlir::RankedTensorType::get({blockM, headDimPadded}, f32Type);
  auto qRowStrideSplat = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockM, 1}, i32Type), qRowStride);
  auto qRowOffsets = builder.create<mlir::arith::MulIOp>(loc, qMExpanded, qRowStrideSplat);
  auto qRowBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmHdType, qRowOffsets);
  auto hdBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmHdType, hdExpanded);
  auto qOffsets2D = builder.create<mlir::arith::AddIOp>(loc, qRowBroadcast, hdBroadcast);

  auto qBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmHdType, qBase);
  auto qFinalOffsets = builder.create<mlir::arith::AddIOp>(loc, qBaseSplat, qOffsets2D);

  // Derive pointer types from actual arguments (NOT hardcoded f32)
  auto qPtrType = mlir::cast<mlir::triton::PointerType>(qPtr.getType());
  auto kPtrTypeAttn = mlir::cast<mlir::triton::PointerType>(kPtr.getType());
  auto vPtrTypeAttn = mlir::cast<mlir::triton::PointerType>(vPtr.getType());
  auto outPtrTypeAttn = mlir::cast<mlir::triton::PointerType>(outPtr.getType());
  auto qPtrTensorType = mlir::RankedTensorType::get({blockM, headDimPadded}, qPtrType);
  auto qSplat = builder.create<mlir::triton::SplatOp>(loc, qPtrTensorType, qPtr);
  auto qPtrs = builder.create<mlir::triton::AddPtrOp>(loc, qPtrTensorType, qSplat, qFinalOffsets);

  // Q mask: qIndices < seqQ (AND rangeHd < headDim if padded)
  auto i1Type = builder.getI1Type();
  auto seqQSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmType, seqQConst);
  auto qMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      qIndices, seqQSplat);
  auto qMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, qMask1D, 1);  // [BM, 1]
  auto i1BmHdType = mlir::RankedTensorType::get({blockM, headDimPadded}, i1Type);
  auto qMask2D_row = builder.create<mlir::triton::BroadcastOp>(loc, i1BmHdType, qMaskExp);
  mlir::Value qMask2D = qMask2D_row;
  if (needsHdMask) {
    auto headDimSplatHd = builder.create<mlir::triton::SplatOp>(loc, i32HdType, headDimConst);
    auto hdMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
        rangeHd, headDimSplatHd);
    auto hdMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, hdMask1D, 0);  // [1, HD]
    auto hdMask2DBm = builder.create<mlir::triton::BroadcastOp>(loc, i1BmHdType, hdMaskExp);
    qMask2D = builder.create<mlir::arith::AndIOp>(loc, qMask2D_row, hdMask2DBm);
  }

  mlir::Value qPtrsVal = qPtrs;
  auto qLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
      qPtrsVal, qMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  // Cast Q to f32 for computation
  auto qLoaded = castTo(builder, loc, qLoadedRaw, f32Type);

  // Apply scale to Q: q_scaled = q * scale
  auto scaleSplat = splatConstantF32(builder, loc, f32BmHdType, scale);
  auto qScaled = builder.create<mlir::arith::MulFOp>(loc, qLoaded, scaleSplat);

  // Initialize accumulators for online softmax:
  // acc = zeros([BLOCK_M, headDim]) — accumulated weighted values
  // m_i = splat(-inf, [BLOCK_M]) — running max
  // l_i = zeros([BLOCK_M]) — running sum of exp
  auto f32BmType = mlir::RankedTensorType::get({blockM}, f32Type);
  auto accInit = splatConstantF32(builder, loc, f32BmHdType, 0.0f);
  auto mInit = splatConstantF32(builder, loc, f32BmType, -3.4028235e+38f);
  auto lInit = splatConstantF32(builder, loc, f32BmType, 0.0f);

  // K-V loop: for j in range(0, seqK, BLOCK_N) — i32 bounds (Triton convention)
  auto jStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto jEnd = builder.create<mlir::arith::ConstantIntOp>(loc, seqK, 32);
  auto jStep = builder.create<mlir::arith::ConstantIntOp>(loc, blockN, 32);

  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, jStart, jEnd, jStep,
      mlir::ValueRange{accInit, mInit, lInit});

  // Inside KV loop
  builder.setInsertionPointToStart(forOp.getBody());
  auto jIdxI32 = forOp.getInductionVar();  // i32 induction variable
  auto accIter = forOp.getBody()->getArgument(1);
  auto mIter = forOp.getBody()->getArgument(2);
  auto lIter = forOp.getBody()->getArgument(3);

  // Compute K indices for this tile
  auto splatJOffset = builder.create<mlir::triton::SplatOp>(loc, i32BnType, jIdxI32);
  auto kIndices = builder.create<mlir::arith::AddIOp>(loc, splatJOffset, rangeN);

  // Clamp K indices for pointer computation: min(kIndices, seqK - 1).
  // Triton masked loads still issue memory transactions for masked-out lanes —
  // the hardware accesses the address then discards the result. During CUDA graph
  // replay, the last tile's OOB pointer addresses (past the K/V buffer) can land
  // on unmapped memory, causing Xid 13 GPU faults. Clamping ensures every pointer
  // stays within the buffer while the mask still zeros the loaded values.
  auto seqKMinus1 = builder.create<mlir::arith::ConstantIntOp>(loc, std::max(0, seqK - 1), 32);
  auto seqKMinus1Splat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, seqKMinus1);
  auto kIndicesClamped = builder.create<mlir::arith::MinSIOp>(loc, kIndices, seqKMinus1Splat);

  // Load K tile [BLOCK_N, headDim]
  // Use clamped indices for pointer computation, original for mask
  auto kNExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndicesClamped, 1);  // [BN, 1]
  auto hdExpandedK = builder.create<mlir::triton::ExpandDimsOp>(loc, rangeHdForPtr, 0);  // [1, HD]

  auto i32BnHdType = mlir::RankedTensorType::get({blockN, headDimPadded}, i32Type);
  auto f32BnHdType = mlir::RankedTensorType::get({blockN, headDimPadded}, f32Type);
  auto kvRowStrideSplat = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockN, 1}, i32Type), kvRowStride);
  auto kRowOffsets = builder.create<mlir::arith::MulIOp>(loc, kNExpanded, kvRowStrideSplat);
  auto kRowBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BnHdType, kRowOffsets);
  auto hdBroadcastK = builder.create<mlir::triton::BroadcastOp>(loc, i32BnHdType, hdExpandedK);
  auto kOffsets2D = builder.create<mlir::arith::AddIOp>(loc, kRowBroadcast, hdBroadcastK);

  auto kvBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnHdType, kvBase);
  auto kFinalOffsets = builder.create<mlir::arith::AddIOp>(loc, kvBaseSplat, kOffsets2D);

  auto kPtrTensorType = mlir::RankedTensorType::get({blockN, headDimPadded}, kPtrTypeAttn);
  auto kSplat = builder.create<mlir::triton::SplatOp>(loc, kPtrTensorType, kPtr);
  auto kPtrs = builder.create<mlir::triton::AddPtrOp>(loc, kPtrTensorType, kSplat, kFinalOffsets);

  // K mask: kIndices < seqK (AND rangeHd < headDim if padded)
  auto seqKSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, seqKConst);
  auto kMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      kIndices, seqKSplat);
  auto kMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 1);
  auto i1BnHdType = mlir::RankedTensorType::get({blockN, headDimPadded}, i1Type);
  auto kMask2D_row = builder.create<mlir::triton::BroadcastOp>(loc, i1BnHdType, kMaskExp);
  mlir::Value kMask2D = kMask2D_row;
  if (needsHdMask) {
    auto headDimSplatHdK = builder.create<mlir::triton::SplatOp>(loc, i32HdType, headDimConst);
    auto hdMask1DK = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
        rangeHd, headDimSplatHdK);
    auto hdMaskExpK = builder.create<mlir::triton::ExpandDimsOp>(loc, hdMask1DK, 0);  // [1, HD]
    auto hdMask2DBn = builder.create<mlir::triton::BroadcastOp>(loc, i1BnHdType, hdMaskExpK);
    kMask2D = builder.create<mlir::arith::AndIOp>(loc, kMask2D_row, hdMask2DBn);
  }

  mlir::Value kLoaded;
  if (dualBuffer) {
    // Dual-buffer K loading: past positions from kPtr (BHSD), current positions from curKPtr (BSHD)
    // isPast mask: kIndices < pastSeq — determines which buffer to read from
    auto pastSeqSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, pastSeqConst);
    auto isPast1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
        kIndices, pastSeqSplat);
    auto isPastExp = builder.create<mlir::triton::ExpandDimsOp>(loc, isPast1D, 1);  // [BN, 1]
    auto isPast2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BnHdType, isPastExp);

    // Past K mask: isPast AND kMask2D (valid past positions with headDim masking)
    auto pastKMask = builder.create<mlir::arith::AndIOp>(loc, isPast2D, kMask2D);

    // Load from past_key buffer (kPtr) — existing pointers are already computed as kPtrs
    auto pastKLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
        kPtrs, pastKMask, mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    auto pastKLoaded = castTo(builder, loc, pastKLoadedRaw, f32Type);

    // Current K: compute adjusted indices (kIndicesClamped - pastSeq) and BSHD offsets
    // Use clamped K indices to prevent OOB pointer computation for masked-out lanes
    auto adjustedK = builder.create<mlir::arith::SubIOp>(loc, kIndicesClamped, pastSeqSplat);
    // Clamp adjusted indices to [0, seqKVCur - 1] for safe pointer computation
    auto zeroConst = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
    auto zeroSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, zeroConst);
    auto adjustedKLow = builder.create<mlir::arith::MaxSIOp>(loc, adjustedK, zeroSplat);
    auto seqKVCurMinus1Const = builder.create<mlir::arith::ConstantIntOp>(loc, std::max(0, seqKVCur - 1), 32);
    auto seqKVCurMinus1Splat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, seqKVCurMinus1Const);
    auto adjustedKClamped = builder.create<mlir::arith::MinSIOp>(loc, adjustedKLow, seqKVCurMinus1Splat);
    auto adjustedKExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, adjustedKClamped, 1);  // [BN, 1]
    auto curKvRowStrideSplat = builder.create<mlir::triton::SplatOp>(loc,
        mlir::RankedTensorType::get({blockN, 1}, i32Type), curKvRowStride);
    auto curKRowOffsets = builder.create<mlir::arith::MulIOp>(loc, adjustedKExpanded, curKvRowStrideSplat);
    auto curKRowBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BnHdType, curKRowOffsets);
    auto curKOffsets2D = builder.create<mlir::arith::AddIOp>(loc, curKRowBroadcast, hdBroadcastK);
    auto curKvBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnHdType, curKvBase);
    auto curKFinalOffsets = builder.create<mlir::arith::AddIOp>(loc, curKvBaseSplat, curKOffsets2D);

    auto curKPtrType = mlir::cast<mlir::triton::PointerType>(curKPtr.getType());
    auto curKPtrTensorType = mlir::RankedTensorType::get({blockN, headDimPadded}, curKPtrType);
    auto curKSplat = builder.create<mlir::triton::SplatOp>(loc, curKPtrTensorType, curKPtr);
    auto curKPtrs = builder.create<mlir::triton::AddPtrOp>(loc, curKPtrTensorType, curKSplat, curKFinalOffsets);

    // Current K mask: kIndices >= pastSeq AND kIndices < seqK (AND headDim mask)
    // Use isCurrent = !isPast (kIndices >= pastSeq)
    auto isCur1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge,
        kIndices, pastSeqSplat);
    auto isCurExp = builder.create<mlir::triton::ExpandDimsOp>(loc, isCur1D, 1);
    auto isCur2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BnHdType, isCurExp);
    auto curKMask = builder.create<mlir::arith::AndIOp>(loc, isCur2D, kMask2D);

    auto curKLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
        curKPtrs, curKMask, mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    auto curKLoaded = castTo(builder, loc, curKLoadedRaw, f32Type);

    // Merge: addition works because exactly one is zero per element (masked load returns 0)
    kLoaded = builder.create<mlir::arith::AddFOp>(loc, pastKLoaded, curKLoaded);
  } else {
    // Single-buffer K loading (original path)
    auto kLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
        kPtrs, kMask2D, mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    kLoaded = castTo(builder, loc, kLoadedRaw, f32Type);
  }

  // QK^T = dot(q_scaled [BM, HD], k^T [HD, BN]) -> [BM, BN]
  auto transposeOrder = builder.getDenseI32ArrayAttr({1, 0});
  auto kTransposed = builder.create<mlir::triton::TransOp>(loc, kLoaded, transposeOrder);

  auto f32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, f32Type);
  auto i32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i32Type);
  auto qkZeroInit = splatConstantF32(builder, loc, f32BmBnType, 0.0f);
  // QK^T precision controlled by tritonTf32Enabled flag. Default IEEE for accuracy.
  auto qkPrecision = sd::Environment::getInstance().tritonTf32Enabled()
                         ? mlir::triton::InputPrecision::TF32
                         : mlir::triton::InputPrecision::IEEE;
  auto qk = builder.create<mlir::triton::DotOp>(
      loc, f32BmBnType, qScaled, kTransposed, qkZeroInit,
      qkPrecision, /*maxNumImpreciseAcc=*/0);

  // Apply key mask: set qk to -inf where kIndices >= seqK
  auto negInfSplat = splatConstantF32(builder, loc, f32BmBnType, -3.4028235e+38f);
  auto kMask1DExp = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 0);  // [1, BN]
  auto i1BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i1Type);
  auto kMaskBmBn = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, kMask1DExp);
  auto qkMasked = builder.create<mlir::arith::SelectOp>(loc, kMaskBmBn, qk, negInfSplat);

  // Apply attention bias/mask if provided
  // Bias shape: [B, H, seqQ, seqK] (rank 4 per-head) or [B, seqQ, seqK] (rank 3)
  // Load bias tile [BM, BN] and add to QK scores — this applies the attention mask
  // (valid positions have bias=0.0, masked/padding positions have bias=-inf)
  mlir::Value qkWithBias = qkMasked;
  if (biasPtr) {
    int biasRank = static_cast<int>(biasShape.size());
    // Determine bias strides based on rank:
    // Rank 4: [B, H, seqQ, seqK] → offset = b*H*seqQ*seqK + h*seqQ*seqK + q*seqK + k
    // Rank 3: [B, seqQ, seqK]    → offset = b*seqQ*seqK + q*seqK + k (no head dim)
    // Rank 2: [B, seqK]          → offset = b*seqK + k (broadcasts across all Q and heads)
    int biasNumHeads = (biasRank >= 4) ? static_cast<int>(biasShape[1]) : 0;
    int biasSeqQ, biasSeqK;
    if (biasRank >= 4) {
      biasSeqQ = static_cast<int>(biasShape[2]);
      biasSeqK = static_cast<int>(biasShape[3]);
    } else if (biasRank == 3) {
      biasSeqQ = static_cast<int>(biasShape[1]);
      biasSeqK = static_cast<int>(biasShape[2]);
    } else {
      // Rank 2: [B, seqK] — broadcast across Q positions and heads
      biasSeqQ = 1;
      biasSeqK = static_cast<int>(biasShape[biasRank - 1]);
    }

    auto biasSeqKConst = builder.create<mlir::arith::ConstantIntOp>(loc, biasSeqK, 32);

    // Compute scalar base offset for this (batch, head)
    // headSliceSize = biasSeqQ * biasSeqK
    auto biasSeqQConst = builder.create<mlir::arith::ConstantIntOp>(loc, biasSeqQ, 32);
    auto headSliceSize = builder.create<mlir::arith::MulIOp>(loc, biasSeqQConst, biasSeqKConst);

    mlir::Value biasBaseScalar;
    if (biasRank >= 4 && biasNumHeads > 1) {
      // 4D per-head: offset = batch * (H * seqQ * seqK) + head * (seqQ * seqK)
      auto biasNumHeadsConst = builder.create<mlir::arith::ConstantIntOp>(loc, biasNumHeads, 32);
      auto batchSliceSize = builder.create<mlir::arith::MulIOp>(loc, biasNumHeadsConst, headSliceSize);
      auto batchOffset = builder.create<mlir::arith::MulIOp>(loc, batchIdx, batchSliceSize);
      auto headOffset = builder.create<mlir::arith::MulIOp>(loc, headIdx, headSliceSize);
      biasBaseScalar = builder.create<mlir::arith::AddIOp>(loc, batchOffset, headOffset);
    } else {
      // 3D or 4D with H=1: offset = batch * (seqQ * seqK)
      biasBaseScalar = builder.create<mlir::arith::MulIOp>(loc, batchIdx, headSliceSize);
    }

    // Q row offsets within bias: qIndicesClamped * biasSeqK  → [BM, 1] → broadcast to [BM, BN]
    // Use clamped indices for pointer safety (same rationale as Q/K clamping)
    auto qBiasRowExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, qIndicesClamped, 1);  // [BM, 1]
    auto biasSeqKSplat = builder.create<mlir::triton::SplatOp>(loc,
        mlir::RankedTensorType::get({blockM, 1}, i32Type), biasSeqKConst);
    auto biasRowOffsets = builder.create<mlir::arith::MulIOp>(loc, qBiasRowExpanded, biasSeqKSplat);
    auto biasRowBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, biasRowOffsets);

    // K column offsets: kIndicesClamped → [1, BN] → broadcast to [BM, BN]
    // Use clamped indices for pointer safety
    auto kBiasColExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndicesClamped, 0);  // [1, BN]
    auto kBiasColBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, kBiasColExpanded);

    // Final: base + qRow*seqK + kCol
    auto biasBaseOffsets = builder.create<mlir::arith::AddIOp>(loc, biasRowBroadcast, kBiasColBroadcast);
    auto biasBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmBnType, biasBaseScalar);
    auto biasFinalOffsets = builder.create<mlir::arith::AddIOp>(loc, biasBaseSplat, biasBaseOffsets);

    // Create bias pointer tensor and load
    auto biasPtrType = mlir::cast<mlir::triton::PointerType>(biasPtr.getType());
    auto biasPtrTensorType = mlir::RankedTensorType::get({blockM, blockN}, biasPtrType);
    auto biasSplat = builder.create<mlir::triton::SplatOp>(loc, biasPtrTensorType, biasPtr);
    auto biasPtrs = builder.create<mlir::triton::AddPtrOp>(loc, biasPtrTensorType, biasSplat, biasFinalOffsets);

    // Bias mask: same as kMaskBmBn (valid Q and K positions)
    auto biasLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
        biasPtrs, kMaskBmBn, mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    auto biasLoaded = castTo(builder, loc, biasLoadedRaw, f32Type);

    // Add bias to QK scores: qk_biased = qk_masked + bias
    qkWithBias = builder.create<mlir::arith::AddFOp>(loc, qkMasked, biasLoaded);
  }

  // Online softmax update:
  // m_new = max(m_i, row_max(qk))
  // correction = exp(m_i - m_new)
  // p = exp(qk - splat(m_new))
  // l_i = l_i * correction + row_sum(p)
  // acc = acc * splat(correction) + dot(p, V)

  // row_max(qk) -> reduce along axis 1
  mlir::Value qkFinalVal = qkWithBias;
  auto rowMaxOp = builder.create<mlir::triton::ReduceOp>(loc,
      mlir::ValueRange{qkFinalVal}, /*axis=*/1);
  {
    auto& region = rowMaxOp.getCombineOp();
    auto* block = builder.createBlock(&region, {}, {f32Type, f32Type}, {loc, loc});
    builder.setInsertionPointToEnd(block);
    auto maxed = builder.create<mlir::arith::MaximumFOp>(loc, block->getArgument(0), block->getArgument(1));
    builder.create<mlir::triton::ReduceReturnOp>(loc, mlir::ValueRange{maxed.getResult()});
  }
  builder.setInsertionPointAfter(rowMaxOp);
  auto rowMax = rowMaxOp->getResult(0);  // [BM]

  // m_new = max(m_i, rowMax)
  auto mNew = builder.create<mlir::arith::MaximumFOp>(loc, mIter, rowMax);

  // correction = exp(m_i - m_new)
  auto mDiff = builder.create<mlir::arith::SubFOp>(loc, mIter, mNew);
  auto correction = builder.create<mlir::math::ExpOp>(loc, mDiff);

  // p = exp(qk - splat(m_new)) -> [BM, BN]
  auto mNewSplat = builder.create<mlir::triton::ExpandDimsOp>(loc, mNew, 1);  // [BM, 1]
  auto mNewBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, f32BmBnType, mNewSplat);
  auto qkShifted = builder.create<mlir::arith::SubFOp>(loc, qkWithBias, mNewBroadcast);
  auto p = builder.create<mlir::math::ExpOp>(loc, qkShifted);

  // row_sum(p) -> reduce along axis 1
  auto rowSumOp = builder.create<mlir::triton::ReduceOp>(loc,
      mlir::ValueRange{p.getResult()}, /*axis=*/1);
  {
    auto& region = rowSumOp.getCombineOp();
    auto* block = builder.createBlock(&region, {}, {f32Type, f32Type}, {loc, loc});
    builder.setInsertionPointToEnd(block);
    auto summed = builder.create<mlir::arith::AddFOp>(loc, block->getArgument(0), block->getArgument(1));
    builder.create<mlir::triton::ReduceReturnOp>(loc, mlir::ValueRange{summed.getResult()});
  }
  builder.setInsertionPointAfter(rowSumOp);
  auto rowSum = rowSumOp->getResult(0);  // [BM]

  // l_new = l_i * correction + rowSum
  auto lScaled = builder.create<mlir::arith::MulFOp>(loc, lIter, correction);
  auto lNew = builder.create<mlir::arith::AddFOp>(loc, lScaled, rowSum);

  // Load V tile [BN, headDim]
  mlir::Value vLoaded;
  if (dualBuffer && curVPtr) {
    // Dual-buffer V loading: same split as K (past from vPtr BHSD, current from curVPtr BSHD)
    auto pastSeqSplatV = builder.create<mlir::triton::SplatOp>(loc, i32BnType, pastSeqConst);
    auto isPastV1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
        kIndices, pastSeqSplatV);
    auto isPastVExp = builder.create<mlir::triton::ExpandDimsOp>(loc, isPastV1D, 1);
    auto isPastV2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BnHdType, isPastVExp);
    auto pastVMask = builder.create<mlir::arith::AndIOp>(loc, isPastV2D, kMask2D);

    // Past V: same offsets as past K (both share BHSD layout and base)
    auto vPtrTensorType = mlir::RankedTensorType::get({blockN, headDimPadded}, vPtrTypeAttn);
    auto vSplat = builder.create<mlir::triton::SplatOp>(loc, vPtrTensorType, vPtr);
    auto vPtrs = builder.create<mlir::triton::AddPtrOp>(loc, vPtrTensorType, vSplat, kFinalOffsets);
    auto pastVLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
        vPtrs, pastVMask, mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    auto pastVLoaded = castTo(builder, loc, pastVLoadedRaw, f32Type);

    // Current V: BSHD offsets (same computation as current K)
    // Use clamped K indices for pointer safety (same as current K path)
    auto adjustedKV = builder.create<mlir::arith::SubIOp>(loc, kIndicesClamped, pastSeqSplatV);
    auto adjustedKVZero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
    auto adjustedKVZeroSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, adjustedKVZero);
    auto adjustedKVLow = builder.create<mlir::arith::MaxSIOp>(loc, adjustedKV, adjustedKVZeroSplat);
    auto seqKVCurM1ForV = builder.create<mlir::arith::ConstantIntOp>(loc, std::max(0, seqKVCur - 1), 32);
    auto seqKVCurM1ForVSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, seqKVCurM1ForV);
    auto adjustedKVClamped = builder.create<mlir::arith::MinSIOp>(loc, adjustedKVLow, seqKVCurM1ForVSplat);
    auto adjustedKVExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, adjustedKVClamped, 1);
    auto curVRowStrideSplat = builder.create<mlir::triton::SplatOp>(loc,
        mlir::RankedTensorType::get({blockN, 1}, i32Type), curKvRowStride);
    auto curVRowOffsets = builder.create<mlir::arith::MulIOp>(loc, adjustedKVExpanded, curVRowStrideSplat);
    auto curVRowBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BnHdType, curVRowOffsets);
    auto curVOffsets2D = builder.create<mlir::arith::AddIOp>(loc, curVRowBroadcast, hdBroadcastK);
    auto curVBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnHdType, curKvBase);
    auto curVFinalOffsets = builder.create<mlir::arith::AddIOp>(loc, curVBaseSplat, curVOffsets2D);

    auto curVPtrType = mlir::cast<mlir::triton::PointerType>(curVPtr.getType());
    auto curVPtrTensorType = mlir::RankedTensorType::get({blockN, headDimPadded}, curVPtrType);
    auto curVSplat = builder.create<mlir::triton::SplatOp>(loc, curVPtrTensorType, curVPtr);
    auto curVPtrs = builder.create<mlir::triton::AddPtrOp>(loc, curVPtrTensorType, curVSplat, curVFinalOffsets);

    auto isCurV1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge,
        kIndices, pastSeqSplatV);
    auto isCurVExp = builder.create<mlir::triton::ExpandDimsOp>(loc, isCurV1D, 1);
    auto isCurV2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BnHdType, isCurVExp);
    auto curVMask = builder.create<mlir::arith::AndIOp>(loc, isCurV2D, kMask2D);

    auto curVLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
        curVPtrs, curVMask, mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    auto curVLoaded = castTo(builder, loc, curVLoadedRaw, f32Type);

    vLoaded = builder.create<mlir::arith::AddFOp>(loc, pastVLoaded, curVLoaded);
  } else {
    // Single-buffer V loading (original path)
    auto vPtrTensorType = mlir::RankedTensorType::get({blockN, headDimPadded}, vPtrTypeAttn);
    auto vSplat = builder.create<mlir::triton::SplatOp>(loc, vPtrTensorType, vPtr);
    auto vPtrs = builder.create<mlir::triton::AddPtrOp>(loc, vPtrTensorType, vSplat, kFinalOffsets);
    auto vLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
        vPtrs, kMask2D, mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    vLoaded = castTo(builder, loc, vLoadedRaw, f32Type);
  }

  // acc_new = acc * splat(correction) + dot(p, V)
  // correction is [BM], need to broadcast to [BM, HD]
  auto correctionExp = builder.create<mlir::triton::ExpandDimsOp>(loc, correction, 1);  // [BM, 1]
  auto correctionBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, f32BmHdType, correctionExp);
  auto accScaled = builder.create<mlir::arith::MulFOp>(loc, accIter, correctionBroadcast);

  // dot(p[BM,BN], V[BN,HD]) -> [BM, HD]
  // PV precision controlled by tritonTf32Enabled flag. Default IEEE for accuracy.
  auto pvPrecision = sd::Environment::getInstance().tritonTf32Enabled()
                         ? mlir::triton::InputPrecision::TF32
                         : mlir::triton::InputPrecision::IEEE;
  auto pv = builder.create<mlir::triton::DotOp>(
      loc, f32BmHdType, p, vLoaded, accScaled,
      pvPrecision, /*maxNumImpreciseAcc=*/0);

  // Yield for next iteration
  mlir::Value pvVal = pv, mNewVal = mNew, lNewVal = lNew;
  mlir::Value yieldVals[] = {pvVal, mNewVal, lNewVal};
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange(yieldVals));

  // After the KV loop
  builder.setInsertionPointAfter(forOp);
  auto finalAcc = forOp.getResult(0);   // [BM, HD]
  auto finalL = forOp.getResult(2);     // [BM]

  // Normalize: result = acc / splat(l_i)
  auto lExp = builder.create<mlir::triton::ExpandDimsOp>(loc, finalL, 1);  // [BM, 1]
  auto lBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, f32BmHdType, lExp);
  auto normalized = builder.create<mlir::arith::DivFOp>(loc, finalAcc, lBroadcast);

  // Store output [BM, headDim]
  // Out base is same as Q base (same layout)
  auto outBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmHdType, qBase);
  auto outFinalOffsets = builder.create<mlir::arith::AddIOp>(loc, outBaseSplat, qOffsets2D);

  auto outPtrTensorTypeAttn = mlir::RankedTensorType::get({blockM, headDimPadded}, outPtrTypeAttn);
  auto outSplatPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorTypeAttn, outPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorTypeAttn, outSplatPtr, outFinalOffsets);

  // Cast normalized f32 result to output element type
  mlir::Value outStoreVal = castTo(builder, loc, normalized, outPtrTypeAttn.getPointeeType());
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, outStoreVal, qMask2D,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  DSP_DIAG(JIT, "TritonIRBuilder: emitted fused attention kernel batch=%d qHeads=%d kvHeads=%d seqQ=%d seqK=%d "
            "headDim=%d scale=%f BM=%d BN=%d kvGroupSize=%d hasBias=%d dualBuffer=%d",
            batchSize, numQHeads, numKvHeads, seqQ, seqK, headDim, scale, blockM, blockN, kvGroupSize,
            biasPtr ? 1 : 0, dualBuffer ? 1 : 0);
}

// ─── Present KV write (for compound attention ops) ──────────────────────────
//
// Writes current key/value (BSHD, 3D [B,seqKV,H*D]) to present_key/value
// output buffer (BHSD, 4D [B,H,totalSeq,D]) at position pastSeq.
// Writes only the seqKV new positions [pastSeq, pastSeq+seqKV) into the destination;
// positions [0, pastSeq) are assumed already populated in the pre-allocated buffer
// from prior decode steps. KV cache scatter into the past KV input buffer is now a
// standard in-graph scatter_upd op, so this kernel no longer depends on any
// C++-side post-pass to propagate new positions.
//
// Grid: pid0 = batch * numKvHeads + kvHeadIdx (same decomposition as attention kernel)
//        pid1 = tile index over seqKV positions
// Each block copies BLOCK_S positions × headDim elements.

void TritonIRBuilder::emitPresentKvWrite(mlir::OpBuilder& builder, mlir::Location loc,
                                          mlir::Value curPtr, mlir::Value presentPtr,
                                          int batchSize, int numQHeads, int numKvHeads,
                                          int pastSeq, int seqKV, int totalSeq, int headDim) {
  DSP_DIAG(JIT, "emitPresentKvWrite: batch=%d qHeads=%d kvHeads=%d pastSeq=%d seqKV=%d totalSeq=%d headDim=%d",
           batchSize, numQHeads, numKvHeads, pastSeq, seqKV, totalSeq, headDim);
  // This function is called WITHIN the attention kernel (same tt.func).
  // Grid is attention's: pid0 = b * numQHeads + qHeadIdx, pid1 = seqQ tile.
  // We decompose pid0 to get batch and qHeadIdx, then map qHeadIdx → kvHeadIdx.
  // Only threads where pid1 == 0 execute the write (avoid redundant writes).
  // For GQA: only write when qHeadIdx % kvGroupSize == 0 (first Q head per KV group).
  auto i32Type = builder.getI32Type();
  auto i1Type = builder.getI1Type();

  int headDimPadded = headDim;
  if (headDimPadded > 0 && (headDimPadded & (headDimPadded - 1)) != 0) {
    int p = 1;
    while (p < headDimPadded) p <<= 1;
    headDimPadded = p;
  }
  bool needsHdMask = (headDimPadded != headDim);

  // Use the attention kernel's program IDs
  auto pid0 = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);
  auto pid1 = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::Y);

  // Only execute on pid1 == 0 to avoid redundant writes across Q tiles
  auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto isPid1Zero = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, pid1, zero);

  // Wrap the entire write in an if (pid1 == 0) block
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, isPid1Zero, /*withElseRegion=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  // Decompose pid0: batch = pid0 / numQHeads, qHeadIdx = pid0 % numQHeads
  auto numQHeadsConst = builder.create<mlir::arith::ConstantIntOp>(loc, numQHeads, 32);
  auto batchIdx = builder.create<mlir::arith::DivSIOp>(loc, pid0, numQHeadsConst);
  auto qHeadIdx = builder.create<mlir::arith::RemSIOp>(loc, pid0, numQHeadsConst);

  // For GQA: kvHeadIdx = qHeadIdx / kvGroupSize. Only write when qHeadIdx % kvGroupSize == 0.
  int kvGroupSize = (numKvHeads > 0) ? (numQHeads / numKvHeads) : 1;
  if (kvGroupSize < 1) kvGroupSize = 1;

  mlir::Value kvHeadIdx;
  if (kvGroupSize > 1) {
    auto kvGroupConst = builder.create<mlir::arith::ConstantIntOp>(loc, kvGroupSize, 32);
    kvHeadIdx = builder.create<mlir::arith::DivSIOp>(loc, qHeadIdx, kvGroupConst);
    // Only first Q head in group writes
    auto remainder = builder.create<mlir::arith::RemSIOp>(loc, qHeadIdx, kvGroupConst);
    auto isFirstInGroup = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, remainder, zero);
    auto innerIf = builder.create<mlir::scf::IfOp>(loc, isFirstInGroup, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&innerIf.getThenRegion().front());
  } else {
    kvHeadIdx = qHeadIdx;
  }

  auto headDimConst = builder.create<mlir::arith::ConstantIntOp>(loc, headDim, 32);
  auto numKvHeadsConst = builder.create<mlir::arith::ConstantIntOp>(loc, numKvHeads, 32);
  auto seqKVConst = builder.create<mlir::arith::ConstantIntOp>(loc, seqKV, 32);
  auto totalSeqConst = builder.create<mlir::arith::ConstantIntOp>(loc, totalSeq, 32);
  auto pastSeqConst = builder.create<mlir::arith::ConstantIntOp>(loc, pastSeq, 32);

  // For decode (seqKV=1), just write a single row of headDim elements per (batch, kvHead)
  // Range over headDim (columns)
  auto i32HdType = mlir::RankedTensorType::get({headDimPadded}, i32Type);
  auto rangeHd = builder.create<mlir::triton::MakeRangeOp>(loc, i32HdType, 0, headDimPadded);
  mlir::Value rangeHdForPtr = rangeHd;
  if (needsHdMask) {
    auto headDimMinus1 = builder.create<mlir::arith::ConstantIntOp>(loc, std::max(0, headDim - 1), 32);
    auto headDimMinus1Splat = builder.create<mlir::triton::SplatOp>(loc, i32HdType, headDimMinus1);
    rangeHdForPtr = builder.create<mlir::arith::MinSIOp>(loc, rangeHd, headDimMinus1Splat);
  }

  // headDim mask
  mlir::Value hdMask;
  if (needsHdMask) {
    auto headDimSplatHd = builder.create<mlir::triton::SplatOp>(loc, i32HdType, headDimConst);
    hdMask = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
        rangeHd, headDimSplatHd);
  }

  // Loop over seqKV positions (usually seqKV=1 for decode, so this is just one iteration)
  auto seqKVVal = builder.create<mlir::arith::ConstantIntOp>(loc, seqKV, 32);
  auto oneConst = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);

  auto forOp = builder.create<mlir::scf::ForOp>(loc, zero, seqKVVal, oneConst);
  builder.setInsertionPointToStart(forOp.getBody());
  auto sIdx = forOp.getInductionVar();  // current position in seqKV

  // Source: curPtr [B, seqKV, numKvHeads*headDim] BSHD (3D)
  // offset = b * seqKV * numKvHeads * headDim + sIdx * numKvHeads * headDim + kvH * headDim + rangeHd
  auto nhTimesHd = builder.create<mlir::arith::MulIOp>(loc, numKvHeadsConst, headDimConst);
  auto srcStride0 = builder.create<mlir::arith::MulIOp>(loc, seqKVConst, nhTimesHd);
  auto srcBatchOff = builder.create<mlir::arith::MulIOp>(loc, batchIdx, srcStride0);
  auto srcSeqOff = builder.create<mlir::arith::MulIOp>(loc, sIdx, nhTimesHd);
  auto srcHeadOff = builder.create<mlir::arith::MulIOp>(loc, kvHeadIdx, headDimConst);
  auto srcBase = builder.create<mlir::arith::AddIOp>(loc, srcBatchOff,
      builder.create<mlir::arith::AddIOp>(loc, srcSeqOff, srcHeadOff));

  auto srcBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32HdType, srcBase);
  auto srcOffsets = builder.create<mlir::arith::AddIOp>(loc, srcBaseSplat, rangeHdForPtr);

  auto curPtrType = mlir::cast<mlir::triton::PointerType>(curPtr.getType());
  auto srcPtrTensorType = mlir::RankedTensorType::get({headDimPadded}, curPtrType);
  auto srcSplat = builder.create<mlir::triton::SplatOp>(loc, srcPtrTensorType, curPtr);
  auto srcPtrs = builder.create<mlir::triton::AddPtrOp>(loc, srcPtrTensorType, srcSplat, srcOffsets);

  auto i1HdType = mlir::RankedTensorType::get({headDimPadded}, i1Type);
  mlir::Value loadMask;
  if (needsHdMask) {
    loadMask = hdMask;
  } else {
    // All true mask
    auto trueConst = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    loadMask = builder.create<mlir::triton::SplatOp>(loc, i1HdType, trueConst);
  }

  auto loaded = builder.create<mlir::triton::LoadOp>(loc,
      srcPtrs, loadMask, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Destination: presentPtr [B, numKvHeads, totalSeq, headDim] BHSD
  // offset = b * numKvHeads * totalSeq * headDim + kvH * totalSeq * headDim + (pastSeq + sIdx) * headDim + rangeHd
  auto totalHd = builder.create<mlir::arith::MulIOp>(loc, totalSeqConst, headDimConst);
  auto dstStride0 = builder.create<mlir::arith::MulIOp>(loc, numKvHeadsConst, totalHd);
  auto dstBatchOff = builder.create<mlir::arith::MulIOp>(loc, batchIdx, dstStride0);
  auto dstHeadOff = builder.create<mlir::arith::MulIOp>(loc, kvHeadIdx, totalHd);
  auto dstSeqIdx = builder.create<mlir::arith::AddIOp>(loc, pastSeqConst, sIdx);
  auto dstSeqOff = builder.create<mlir::arith::MulIOp>(loc, dstSeqIdx, headDimConst);
  auto dstBase = builder.create<mlir::arith::AddIOp>(loc, dstBatchOff,
      builder.create<mlir::arith::AddIOp>(loc, dstHeadOff, dstSeqOff));

  auto dstBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32HdType, dstBase);
  auto dstOffsets = builder.create<mlir::arith::AddIOp>(loc, dstBaseSplat, rangeHdForPtr);

  auto presentPtrType = mlir::cast<mlir::triton::PointerType>(presentPtr.getType());
  auto dstPtrTensorType = mlir::RankedTensorType::get({headDimPadded}, presentPtrType);
  auto dstSplat = builder.create<mlir::triton::SplatOp>(loc, dstPtrTensorType, presentPtr);
  auto dstPtrs = builder.create<mlir::triton::AddPtrOp>(loc, dstPtrTensorType, dstSplat, dstOffsets);

  // Cast if types differ
  mlir::Value storeVal = loaded;
  auto srcElemType = curPtrType.getPointeeType();
  auto dstElemType = presentPtrType.getPointeeType();
  if (srcElemType != dstElemType) {
    storeVal = castTo(builder, loc, loaded, dstElemType);
  }

  builder.create<mlir::triton::StoreOp>(loc, dstPtrs, storeVal, loadMask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  // End for loop
  builder.setInsertionPointAfter(forOp);

  // End GQA guard if needed
  if (kvGroupSize > 1) {
    // Builder is inside innerIf, move after it
    auto* innerIfBlock = builder.getBlock()->getParentOp();
    builder.setInsertionPointAfter(innerIfBlock);
  }

  // End pid1==0 guard
  builder.setInsertionPointAfter(ifOp);

  DSP_DIAG(JIT, "TritonIRBuilder: emitted present KV write batch=%d qHeads=%d kvHeads=%d pastSeq=%d seqKV=%d totalSeq=%d headDim=%d",
            batchSize, numQHeads, numKvHeads, pastSeq, seqKV, totalSeq, headDim);
}

// ─── Flash Attention Backward kernel ────────────────────────────────────────
//
// Implements Flash Attention 2 backward (Tri Dao et al.).
// Softmax recomputation: instead of storing the full attention matrix (O(N^2)),
// only the log-sum-exp L (computed in the forward pass) is stored (O(N)).
// P = exp(S - L) reconstructs the softmax probabilities exactly.
//
// Algorithm per Q-tile (outer) × K/V-tile (inner):
//   D = rowsum(dO * O)                          // precompute, [BM]
//   S = Q_block @ K_block^T * scale             // [BM, BN]
//   P = exp(S - L_block)                        // recompute softmax
//   dV_block += P^T @ dO_block                  // [BN, HD]
//   dP = dO_block @ V_block^T                   // [BM, BN]
//   dS = P * (dP - D_block)                     // softmax grad [BM, BN]
//   dQ_block += dS @ K_block * scale            // [BM, HD]
//   dK_block += dS^T @ Q_block * scale          // [BN, HD]
//
// Grid: (batch * numQHeads, ceil(seqQ / BLOCK_M)) — same as forward.
// All tensors BHSD layout: [batch, heads, seq, headDim].
// dQ/dK/dV are accumulated atomically because each Q-tile writes to dK/dV
// at possibly overlapping K-positions when the outer loop is over Q-tiles.
// However, in the standard blocked outer-Q / inner-K loop structure, each
// (block-i, block-j) pair writes exactly once to dK[block-j] and dV[block-j],
// so no atomics are needed — straight store accumulation with scf.for yield.
//
void TritonIRBuilder::emitFusedAttentionBackwardKernel(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::Value dOPtr, mlir::Value qPtr,
    mlir::Value kPtr, mlir::Value vPtr,
    mlir::Value oPtr, mlir::Value lsePtr,
    mlir::Value dQPtr, mlir::Value dKPtr, mlir::Value dVPtr,
    int batchSize, int numQHeads,
    int seqQ, int seqK,
    int headDim, float scale,
    int blockM, int blockN) {
  DSP_DIAG(JIT, "emitFusedAttentionBackwardKernel: batch=%d qHeads=%d seqQ=%d seqK=%d "
           "headDim=%d scale=%f BM=%d BN=%d",
           batchSize, numQHeads, seqQ, seqK, headDim, scale, blockM, blockN);

  auto f32Type = builder.getF32Type();
  auto i32Type = builder.getI32Type();
  auto i1Type  = builder.getI1Type();

  // Round headDim up to power-of-2 for Triton (MakeRangeOp requires it)
  int headDimPadded = headDim;
  if (headDimPadded > 0 && (headDimPadded & (headDimPadded - 1)) != 0) {
    int p = 1;
    while (p < headDimPadded) p <<= 1;
    headDimPadded = p;
  }
  bool needsHdMask = (headDimPadded != headDim);

  // ── Program IDs ──────────────────────────────────────────────────────────
  // pid0 = batch * numQHeads + qHeadIdx
  // pid1 = Q-tile index
  auto pid0 = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);
  auto pid1 = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::Y);

  auto numQHeadsConst = builder.create<mlir::arith::ConstantIntOp>(loc, numQHeads, 32);
  auto headIdx  = builder.create<mlir::arith::RemSIOp>(loc, pid0, numQHeadsConst);
  auto batchIdx = builder.create<mlir::arith::DivSIOp>(loc, pid0, numQHeadsConst);

  // Q-tile offset along seqQ
  auto blockMConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockM, 32);
  auto qTileStart  = builder.create<mlir::arith::MulIOp>(loc, pid1, blockMConst);

  // ── Tensor type aliases ───────────────────────────────────────────────────
  auto i32BmType   = mlir::RankedTensorType::get({blockM}, i32Type);
  auto i32BnType   = mlir::RankedTensorType::get({blockN}, i32Type);
  auto i32HdType   = mlir::RankedTensorType::get({headDimPadded}, i32Type);
  auto f32BmType   = mlir::RankedTensorType::get({blockM}, f32Type);
  auto f32BmHdType = mlir::RankedTensorType::get({blockM, headDimPadded}, f32Type);
  auto f32BnHdType = mlir::RankedTensorType::get({blockN, headDimPadded}, f32Type);
  auto f32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, f32Type);
  auto i1BmHdType  = mlir::RankedTensorType::get({blockM, headDimPadded}, i1Type);
  auto i1BnHdType  = mlir::RankedTensorType::get({blockN, headDimPadded}, i1Type);
  auto i1BmBnType  = mlir::RankedTensorType::get({blockM, blockN}, i1Type);

  auto seqQConst   = builder.create<mlir::arith::ConstantIntOp>(loc, seqQ, 32);
  auto seqKConst   = builder.create<mlir::arith::ConstantIntOp>(loc, seqK, 32);
  auto headDimConst = builder.create<mlir::arith::ConstantIntOp>(loc, headDim, 32);

  // ── BHSD strides ─────────────────────────────────────────────────────────
  // base(batch, head, seq=0) = batch * numQHeads * seq * HD + head * seq * HD
  // rowStride = headDim  (next row = next sequence position)
  auto stride1 = builder.create<mlir::arith::MulIOp>(loc, seqQConst, headDimConst);  // seq * HD
  auto stride0 = builder.create<mlir::arith::MulIOp>(loc, numQHeadsConst, stride1);  // NH * seq * HD

  // Q/O/dQ/dO base: same BHSD layout, same seqQ dimension
  auto qHeadOff  = builder.create<mlir::arith::MulIOp>(loc, headIdx,  stride1);
  auto qBatchOff = builder.create<mlir::arith::MulIOp>(loc, batchIdx, stride0);
  auto qBase     = builder.create<mlir::arith::AddIOp>(loc, qBatchOff, qHeadOff);
  // Row stride for Q/O/dO = headDim (contiguous rows within a head's [seqQ, HD] slice)
  auto rowStrideQ = headDimConst;

  // K/V/dK/dV base: same NH but over seqK dimension
  auto strideK1  = builder.create<mlir::arith::MulIOp>(loc, seqKConst, headDimConst);
  auto strideK0  = builder.create<mlir::arith::MulIOp>(loc, numQHeadsConst, strideK1);
  auto kvHeadOff = builder.create<mlir::arith::MulIOp>(loc, headIdx,  strideK1);
  auto kvBatchOff= builder.create<mlir::arith::MulIOp>(loc, batchIdx, strideK0);
  auto kvBase    = builder.create<mlir::arith::AddIOp>(loc, kvBatchOff, kvHeadOff);

  // LSE base: [batch, numQHeads, seqQ] — log-sum-exp per query position
  auto lseStride1 = seqQConst;                                                          // seqQ
  auto lseStride0 = builder.create<mlir::arith::MulIOp>(loc, numQHeadsConst, lseStride1);  // NH * seqQ
  auto lseHeadOff = builder.create<mlir::arith::MulIOp>(loc, headIdx,  lseStride1);
  auto lseBatchOff= builder.create<mlir::arith::MulIOp>(loc, batchIdx, lseStride0);
  auto lseBase    = builder.create<mlir::arith::AddIOp>(loc, lseBatchOff, lseHeadOff);

  // ── Q-tile indices [blockM] ───────────────────────────────────────────────
  auto rangeM   = builder.create<mlir::triton::MakeRangeOp>(loc, i32BmType, 0, blockM);
  auto rangeN   = builder.create<mlir::triton::MakeRangeOp>(loc, i32BnType, 0, blockN);
  auto rangeHd  = builder.create<mlir::triton::MakeRangeOp>(loc, i32HdType, 0, headDimPadded);
  mlir::Value rangeHdForPtr = rangeHd;
  if (needsHdMask) {
    auto headDimMinus1 = builder.create<mlir::arith::ConstantIntOp>(loc, std::max(0, headDim - 1), 32);
    auto headDimMinus1Splat = builder.create<mlir::triton::SplatOp>(loc, i32HdType, headDimMinus1);
    rangeHdForPtr = builder.create<mlir::arith::MinSIOp>(loc, rangeHd, headDimMinus1Splat);
  }

  auto splatQStart = builder.create<mlir::triton::SplatOp>(loc, i32BmType, qTileStart);
  auto qIndices    = builder.create<mlir::arith::AddIOp>(loc, splatQStart, rangeM);
  auto seqQMinus1 = builder.create<mlir::arith::ConstantIntOp>(loc, std::max(0, seqQ - 1), 32);
  auto seqQMinus1Splat = builder.create<mlir::triton::SplatOp>(loc, i32BmType, seqQMinus1);
  auto qIndicesClamped = builder.create<mlir::arith::MinSIOp>(loc, qIndices, seqQMinus1Splat);

  // qMask1D: qIndices < seqQ
  auto seqQSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmType, seqQConst);
  auto qMask1D   = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                                         qIndices, seqQSplat);

  // ── Derive MLIR pointer types from the actual argument values ─────────────
  auto qPtrType  = mlir::cast<mlir::triton::PointerType>(qPtr.getType());
  auto kPtrType  = mlir::cast<mlir::triton::PointerType>(kPtr.getType());
  auto vPtrType  = mlir::cast<mlir::triton::PointerType>(vPtr.getType());
  auto oPtrType  = mlir::cast<mlir::triton::PointerType>(oPtr.getType());
  auto doPtrType = mlir::cast<mlir::triton::PointerType>(dOPtr.getType());
  auto lsePtrType= mlir::cast<mlir::triton::PointerType>(lsePtr.getType());
  auto dqPtrType = mlir::cast<mlir::triton::PointerType>(dQPtr.getType());
  auto dkPtrType = mlir::cast<mlir::triton::PointerType>(dKPtr.getType());
  auto dvPtrType = mlir::cast<mlir::triton::PointerType>(dVPtr.getType());

  // ── Helper: build 2D [BM/BN, HD] offsets from a base scalar ─────────────
  // offsets2D = qIndices[:, None] * rowStride + rangeHd[None, :]
  // Then add qBase splat → final flat offsets into the buffer.

  // ─ Q 2D offsets [BM, HD] ─
  auto qMExp     = builder.create<mlir::triton::ExpandDimsOp>(loc, qIndicesClamped, 1);   // [BM, 1]
  auto hdExp0    = builder.create<mlir::triton::ExpandDimsOp>(loc, rangeHdForPtr, 0);    // [1, HD]
  auto rowSQ     = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, 1}, i32Type), rowStrideQ);
  auto qRowOff   = builder.create<mlir::arith::MulIOp>(loc, qMExp, rowSQ);
  auto qRowBcast = builder.create<mlir::triton::BroadcastOp>(loc, mlir::RankedTensorType::get({blockM, headDimPadded}, i32Type), qRowOff);
  auto hdBcast0  = builder.create<mlir::triton::BroadcastOp>(loc, mlir::RankedTensorType::get({blockM, headDimPadded}, i32Type), hdExp0);
  auto qOff2D    = builder.create<mlir::arith::AddIOp>(loc, qRowBcast, hdBcast0);
  auto qBaseSplat= builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, headDimPadded}, i32Type), qBase);
  auto qFinalOff = builder.create<mlir::arith::AddIOp>(loc, qBaseSplat, qOff2D);

  // Q 2D mask: qMask1D broadcast to [BM, HD] (and headDim padding mask if needed)
  auto qMaskExp  = builder.create<mlir::triton::ExpandDimsOp>(loc, qMask1D, 1);
  auto qMask2D_row = builder.create<mlir::triton::BroadcastOp>(loc, i1BmHdType, qMaskExp);
  mlir::Value qMask2D = qMask2D_row;
  if (needsHdMask) {
    auto hdConst  = builder.create<mlir::triton::SplatOp>(loc, i32HdType, headDimConst);
    auto hdMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, rangeHd, hdConst);
    auto hdMaskExp= builder.create<mlir::triton::ExpandDimsOp>(loc, hdMask1D, 0);
    auto hdMask2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BmHdType, hdMaskExp);
    qMask2D = builder.create<mlir::arith::AndIOp>(loc, qMask2D_row, hdMask2D);
  }

  // ── Load Q tile [BM, HD] ─────────────────────────────────────────────────
  auto qPtrTT  = mlir::RankedTensorType::get({blockM, headDimPadded}, qPtrType);
  auto qSplat  = builder.create<mlir::triton::SplatOp>(loc, qPtrTT, qPtr);
  auto qPtrs   = builder.create<mlir::triton::AddPtrOp>(loc, qPtrTT, qSplat, qFinalOff);
  auto qLoadedRaw = builder.create<mlir::triton::LoadOp>(loc, qPtrs, qMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto qLoaded = castTo(builder, loc, qLoadedRaw, f32Type);  // [BM, HD] f32

  // ── Load O tile [BM, HD] ─────────────────────────────────────────────────
  auto oPtrTT  = mlir::RankedTensorType::get({blockM, headDimPadded}, oPtrType);
  auto oSplat  = builder.create<mlir::triton::SplatOp>(loc, oPtrTT, oPtr);
  auto oPtrs   = builder.create<mlir::triton::AddPtrOp>(loc, oPtrTT, oSplat, qFinalOff);
  auto oLoadedRaw = builder.create<mlir::triton::LoadOp>(loc, oPtrs, qMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto oLoaded = castTo(builder, loc, oLoadedRaw, f32Type);  // [BM, HD] f32

  // ── Load dO tile [BM, HD] ─────────────────────────────────────────────────
  auto doPtrTT = mlir::RankedTensorType::get({blockM, headDimPadded}, doPtrType);
  auto doSplat = builder.create<mlir::triton::SplatOp>(loc, doPtrTT, dOPtr);
  auto doPtrs  = builder.create<mlir::triton::AddPtrOp>(loc, doPtrTT, doSplat, qFinalOff);
  auto doLoadedRaw = builder.create<mlir::triton::LoadOp>(loc, doPtrs, qMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto doLoaded = castTo(builder, loc, doLoadedRaw, f32Type);  // [BM, HD] f32

  // ── Precompute D = rowsum(dO * O) [BM] ─────────────────────────────────
  // D[i] = sum_d (dO[i,d] * O[i,d]) — used in the softmax backward
  auto doTimesO = builder.create<mlir::arith::MulFOp>(loc, doLoaded, oLoaded);  // [BM, HD]
  auto dRowSumOp = builder.create<mlir::triton::ReduceOp>(loc,
      mlir::ValueRange{doTimesO.getResult()}, /*axis=*/1);
  {
    auto& region = dRowSumOp.getCombineOp();
    auto* blk = builder.createBlock(&region, {}, {f32Type, f32Type}, {loc, loc});
    builder.setInsertionPointToEnd(blk);
    auto summed = builder.create<mlir::arith::AddFOp>(loc, blk->getArgument(0), blk->getArgument(1));
    builder.create<mlir::triton::ReduceReturnOp>(loc, mlir::ValueRange{summed.getResult()});
  }
  builder.setInsertionPointAfter(dRowSumOp);
  auto D = dRowSumOp->getResult(0);  // [BM] — D[i] = rowsum(dO[i] * O[i])

  // ── Load L (log-sum-exp) for this Q tile [BM] ──────────────────────────
  // L is stored as a 1D vector per (batch, head): shape [batch, numQHeads, seqQ]
  // L[i] = log(sum_j exp(S[i,j])) — from the forward softmax
  auto lse1DPtrTT = mlir::RankedTensorType::get({blockM}, lsePtrType);
  auto lseSplat   = builder.create<mlir::triton::SplatOp>(loc, lse1DPtrTT, lsePtr);
  auto lseBaseVal = lseBase;
  auto splatLseBase = builder.create<mlir::triton::SplatOp>(loc, i32BmType, lseBaseVal);
  auto lseOffsets   = builder.create<mlir::arith::AddIOp>(loc, splatLseBase, qIndicesClamped);
  auto lsePtrs      = builder.create<mlir::triton::AddPtrOp>(loc, lse1DPtrTT, lseSplat, lseOffsets);
  auto lseMask      = qMask1D;
  auto lseLoadedRaw = builder.create<mlir::triton::LoadOp>(loc, lsePtrs, lseMask, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto L = castTo(builder, loc, lseLoadedRaw, f32Type);  // [BM] f32

  // ── Initialize dQ accumulator [BM, HD] to zeros ──────────────────────────
  auto dQAccInit = splatConstantF32(builder, loc, f32BmHdType, 0.0f);

  // ── Inner K/V loop: iterate over K-tiles ─────────────────────────────────
  // For each K-tile j:
  //   Recompute S[BM,BN] = Q[BM,HD] @ K[BN,HD]^T * scale
  //   Recompute P[BM,BN] = exp(S - L[:, None])
  //   dV[BN,HD] += P^T[BN,BM] @ dO[BM,HD]
  //   dP[BM,BN] = dO[BM,HD] @ V[BN,HD]^T
  //   dS[BM,BN] = P * (dP - D[:, None])
  //   dQ[BM,HD] += dS[BM,BN] @ K[BN,HD] * scale
  //   dK[BN,HD] += dS^T[BN,BM] @ Q[BM,HD] * scale

  auto jStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto jEnd   = builder.create<mlir::arith::ConstantIntOp>(loc, seqK, 32);
  auto jStep  = builder.create<mlir::arith::ConstantIntOp>(loc, blockN, 32);

  // dQ accumulator is a loop-carried value (dK, dV are accumulated per K-tile
  // and stored immediately since each K-tile sees all Q-tiles serially here)
  auto kvForOp = builder.create<mlir::scf::ForOp>(
      loc, jStart, jEnd, jStep,
      mlir::ValueRange{dQAccInit});

  builder.setInsertionPointToStart(kvForOp.getBody());
  auto jIdx   = kvForOp.getInductionVar();
  auto dQAcc  = kvForOp.getBody()->getArgument(1);   // [BM, HD] dQ accumulator

  // K-tile indices [BN]
  auto splatJ   = builder.create<mlir::triton::SplatOp>(loc, i32BnType, jIdx);
  auto kIndices = builder.create<mlir::arith::AddIOp>(loc, splatJ, rangeN);
  auto seqKMinus1 = builder.create<mlir::arith::ConstantIntOp>(loc, std::max(0, seqK - 1), 32);
  auto seqKMinus1Splat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, seqKMinus1);
  auto kIndicesClamped = builder.create<mlir::arith::MinSIOp>(loc, kIndices, seqKMinus1Splat);

  // kMask1D: kIndices < seqK
  auto seqKSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, seqKConst);
  auto kMask1D   = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                                         kIndices, seqKSplat);

  // K 2D offsets [BN, HD] into K buffer
  auto kNExp    = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndicesClamped, 1);   // [BN, 1]
  auto hdExpK   = builder.create<mlir::triton::ExpandDimsOp>(loc, rangeHdForPtr, 0);    // [1, HD]
  auto rowSK    = builder.create<mlir::triton::SplatOp>(loc,
                      mlir::RankedTensorType::get({blockN, 1}, i32Type), headDimConst);
  auto kRowOff  = builder.create<mlir::arith::MulIOp>(loc, kNExp, rowSK);
  auto kRowBcast= builder.create<mlir::triton::BroadcastOp>(loc, mlir::RankedTensorType::get({blockN, headDimPadded}, i32Type), kRowOff);
  auto hdBcastK = builder.create<mlir::triton::BroadcastOp>(loc, mlir::RankedTensorType::get({blockN, headDimPadded}, i32Type), hdExpK);
  auto kOff2D   = builder.create<mlir::arith::AddIOp>(loc, kRowBcast, hdBcastK);
  auto kvBaseS  = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockN, headDimPadded}, i32Type), kvBase);
  auto kFinalOff= builder.create<mlir::arith::AddIOp>(loc, kvBaseS, kOff2D);

  // kMask2D [BN, HD]
  auto kMaskExpK = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 1);
  auto kMask2D_row = builder.create<mlir::triton::BroadcastOp>(loc, i1BnHdType, kMaskExpK);
  mlir::Value kMask2D = kMask2D_row;
  if (needsHdMask) {
    auto hdConstK  = builder.create<mlir::triton::SplatOp>(loc, i32HdType, headDimConst);
    auto hdMask1DK = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, rangeHd, hdConstK);
    auto hdMaskExpK= builder.create<mlir::triton::ExpandDimsOp>(loc, hdMask1DK, 0);
    auto hdMask2DK = builder.create<mlir::triton::BroadcastOp>(loc, i1BnHdType, hdMaskExpK);
    kMask2D = builder.create<mlir::arith::AndIOp>(loc, kMask2D_row, hdMask2DK);
  }

  // Load K tile [BN, HD]
  auto kPtrTT  = mlir::RankedTensorType::get({blockN, headDimPadded}, kPtrType);
  auto kSplat  = builder.create<mlir::triton::SplatOp>(loc, kPtrTT, kPtr);
  auto kPtrs   = builder.create<mlir::triton::AddPtrOp>(loc, kPtrTT, kSplat, kFinalOff);
  auto kLoadedRaw = builder.create<mlir::triton::LoadOp>(loc, kPtrs, kMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto kLoaded = castTo(builder, loc, kLoadedRaw, f32Type);  // [BN, HD] f32

  // Load V tile [BN, HD]
  auto vPtrTT  = mlir::RankedTensorType::get({blockN, headDimPadded}, vPtrType);
  auto vSplat  = builder.create<mlir::triton::SplatOp>(loc, vPtrTT, vPtr);
  auto vPtrs   = builder.create<mlir::triton::AddPtrOp>(loc, vPtrTT, vSplat, kFinalOff);
  auto vLoadedRaw = builder.create<mlir::triton::LoadOp>(loc, vPtrs, kMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto vLoaded = castTo(builder, loc, vLoadedRaw, f32Type);  // [BN, HD] f32

  // ── Step 1: Recompute S = Q @ K^T * scale → [BM, BN] ────────────────────
  // kTransposed [HD, BN] = transpose(K [BN, HD])
  auto transposeOrder = builder.getDenseI32ArrayAttr({1, 0});
  auto kTransposed  = builder.create<mlir::triton::TransOp>(loc, kLoaded, transposeOrder);

  auto qkZero   = splatConstantF32(builder, loc, f32BmBnType, 0.0f);
  auto dotPrec  = sd::Environment::getInstance().tritonTf32Enabled()
                      ? mlir::triton::InputPrecision::TF32
                      : mlir::triton::InputPrecision::IEEE;
  // S = Q [BM,HD] @ K^T [HD,BN] * scale
  auto scaleSplatBmHd = splatConstantF32(builder, loc, f32BmHdType, scale);
  auto qScaled  = builder.create<mlir::arith::MulFOp>(loc, qLoaded, scaleSplatBmHd);  // [BM, HD]
  auto S = builder.create<mlir::triton::DotOp>(
      loc, f32BmBnType, qScaled, kTransposed, qkZero, dotPrec, 0);  // [BM, BN]

  // Apply K validity mask: S = -inf where kIndices >= seqK
  auto negInfBmBn = splatConstantF32(builder, loc, f32BmBnType, -3.4028235e+38f);
  auto kMask1DExpBmBn = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 0);
  auto kMaskBmBn = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, kMask1DExpBmBn);
  auto SMasked   = builder.create<mlir::arith::SelectOp>(loc, kMaskBmBn, S, negInfBmBn);

  // ── Step 2: Recompute P = exp(S - L) → [BM, BN] ─────────────────────────
  // L is [BM], broadcast to [BM, BN]
  auto LExpanded  = builder.create<mlir::triton::ExpandDimsOp>(loc, L, 1);         // [BM, 1]
  auto LBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, f32BmBnType, LExpanded);
  auto SMinusL    = builder.create<mlir::arith::SubFOp>(loc, SMasked, LBroadcast);
  auto P = builder.create<mlir::math::ExpOp>(loc, SMinusL);  // [BM, BN]

  // Zero out P where K positions are invalid (exp(-inf) should be 0 but be explicit)
  auto zerosBmBn = splatConstantF32(builder, loc, f32BmBnType, 0.0f);
  auto PValid    = builder.create<mlir::arith::SelectOp>(loc, kMaskBmBn, P, zerosBmBn);

  // ── Step 3: dV_block += P^T @ dO → [BN, HD] ──────────────────────────────
  // P^T [BN, BM] = transpose(P [BM, BN])
  // dV_contribution = P^T [BN, BM] @ dO [BM, HD] → [BN, HD]
  // Read existing dV at this K-block to accumulate
  auto dVZeroInit = splatConstantF32(builder, loc, f32BnHdType, 0.0f);

  auto dVPtrTT = mlir::RankedTensorType::get({blockN, headDimPadded}, dvPtrType);
  auto dVSplatP= builder.create<mlir::triton::SplatOp>(loc, dVPtrTT, dVPtr);
  auto dVPtrs  = builder.create<mlir::triton::AddPtrOp>(loc, dVPtrTT, dVSplatP, kFinalOff);
  // Load current dV accumulator from output buffer (atomic accumulation via load-add-store)
  auto dVCurrentRaw = builder.create<mlir::triton::LoadOp>(loc, dVPtrs, kMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto dVCurrent = castTo(builder, loc, dVCurrentRaw, f32Type);  // [BN, HD] f32

  auto PTransposed = builder.create<mlir::triton::TransOp>(loc, PValid, transposeOrder);  // [BN, BM]
  auto dVContrib   = builder.create<mlir::triton::DotOp>(
      loc, f32BnHdType, PTransposed, doLoaded, dVCurrent, dotPrec, 0);  // [BN, HD]
  // Store accumulated dV back
  mlir::Value dVStore = castTo(builder, loc, dVContrib, dvPtrType.getPointeeType());
  builder.create<mlir::triton::StoreOp>(loc, dVPtrs, dVStore, kMask2D,
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);

  // ── Step 4: dP = dO @ V^T → [BM, BN] ───────────────────────────────────
  // V^T [HD, BN] = transpose(V [BN, HD])
  auto vTransposed = builder.create<mlir::triton::TransOp>(loc, vLoaded, transposeOrder);   // [HD, BN]
  auto dPZero      = splatConstantF32(builder, loc, f32BmBnType, 0.0f);
  auto dP = builder.create<mlir::triton::DotOp>(
      loc, f32BmBnType, doLoaded, vTransposed, dPZero, dotPrec, 0);  // [BM, BN]

  // ── Step 5: dS = P * (dP - D) → [BM, BN] ───────────────────────────────
  // D is [BM], broadcast to [BM, BN]
  auto DExpanded  = builder.create<mlir::triton::ExpandDimsOp>(loc, D, 1);         // [BM, 1]
  auto DBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, f32BmBnType, DExpanded);
  auto dPMinusD   = builder.create<mlir::arith::SubFOp>(loc, dP, DBroadcast);       // dP - D
  auto dS         = builder.create<mlir::arith::MulFOp>(loc, PValid, dPMinusD);     // P*(dP-D) [BM,BN]

  // ── Step 6: dQ_block += dS @ K * scale → [BM, HD] ──────────────────────
  // dS [BM, BN] @ K [BN, HD] → [BM, HD], then multiply by scale
  auto dQContrib = builder.create<mlir::triton::DotOp>(
      loc, f32BmHdType, dS, kLoaded, dQAcc, dotPrec, 0);  // [BM, HD] accumulated into dQAcc
  auto scaleSplatBmHd2 = splatConstantF32(builder, loc, f32BmHdType, scale);
  auto dQContribScaled = builder.create<mlir::arith::MulFOp>(loc, dQContrib, scaleSplatBmHd2);

  // ── Step 7: dK_block += dS^T @ Q * scale → [BN, HD] ────────────────────
  // dS^T [BN, BM] = transpose(dS [BM, BN])
  // dK_contribution = dS^T [BN,BM] @ Q [BM,HD] * scale → [BN, HD]
  // Read existing dK at this K-block
  auto dKPtrTT = mlir::RankedTensorType::get({blockN, headDimPadded}, dkPtrType);
  auto dKSplatP= builder.create<mlir::triton::SplatOp>(loc, dKPtrTT, dKPtr);
  auto dKPtrs  = builder.create<mlir::triton::AddPtrOp>(loc, dKPtrTT, dKSplatP, kFinalOff);
  auto dKCurrentRaw = builder.create<mlir::triton::LoadOp>(loc, dKPtrs, kMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto dKCurrent = castTo(builder, loc, dKCurrentRaw, f32Type);  // [BN, HD] f32

  auto dSTransposed  = builder.create<mlir::triton::TransOp>(loc, dS, transposeOrder);   // [BN, BM]
  auto dKContrib     = builder.create<mlir::triton::DotOp>(
      loc, f32BnHdType, dSTransposed, qLoaded, dKCurrent, dotPrec, 0);  // [BN, HD]
  auto scaleSplatBnHd = splatConstantF32(builder, loc, f32BnHdType, scale);
  auto dKContribScaled = builder.create<mlir::arith::MulFOp>(loc, dKContrib, scaleSplatBnHd);
  // Store accumulated dK back
  mlir::Value dKStore = castTo(builder, loc, dKContribScaled, dkPtrType.getPointeeType());
  builder.create<mlir::triton::StoreOp>(loc, dKPtrs, dKStore, kMask2D,
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);

  // Yield updated dQ accumulator
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{dQContribScaled});

  // ── After K/V loop: store dQ ────────────────────────────────────────────
  builder.setInsertionPointAfter(kvForOp);
  auto finalDQ = kvForOp.getResult(0);  // [BM, HD]

  auto dQPtrTT = mlir::RankedTensorType::get({blockM, headDimPadded}, dqPtrType);
  auto dQSplat = builder.create<mlir::triton::SplatOp>(loc, dQPtrTT, dQPtr);
  auto dQPtrs  = builder.create<mlir::triton::AddPtrOp>(loc, dQPtrTT, dQSplat, qFinalOff);
  mlir::Value dQStore = castTo(builder, loc, finalDQ, dqPtrType.getPointeeType());
  builder.create<mlir::triton::StoreOp>(loc, dQPtrs, dQStore, qMask2D,
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);

  DSP_DIAG(JIT, "TritonIRBuilder: emitted fused attention backward kernel "
           "batch=%d qHeads=%d seqQ=%d seqK=%d headDim=%d scale=%f BM=%d BN=%d",
           batchSize, numQHeads, seqQ, seqK, headDim, scale, blockM, blockN);
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
