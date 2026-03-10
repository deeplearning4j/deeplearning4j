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

void TritonIRBuilder::emitMatmulKernel(mlir::OpBuilder& builder, mlir::Location loc,
                                        mlir::Value aPtr, mlir::Value bPtr, mlir::Value cPtr,
                                        int M, int N, int K,
                                        int blockM, int blockN, int blockK) {
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

  // Determine InputPrecision for DotOp based on input types
  auto dotPrecision = mlir::triton::InputPrecision::TF32;  // default for f32 inputs
  bool inputIsF32 = mlir::isa<mlir::Float32Type>(aElemType);
  if (!inputIsF32) {
    // f16, bf16, int8 use IEEE — TF32 only applies to f32 inputs
    dotPrecision = mlir::triton::InputPrecision::IEEE;
  }

  DSP_DIAG(JIT, "TritonIRBuilder::emitMatmulKernel: A elem=%s, B elem=%s, C elem=%s, precision=%s",
            inputIsF32 ? "f32" : "non-f32", inputIsF32 ? "f32" : "non-f32",
            mlir::isa<mlir::Float32Type>(cElemType) ? "f32" : "non-f32",
            inputIsF32 ? "TF32" : "IEEE");

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
  // tt.dot requires A and B to have same element bit width.
  // Accumulator is always f32. Tensor cores handle f16/bf16→f32 natively.
  auto dotResult = builder.create<mlir::triton::DotOp>(
      loc, accType, aLoaded, bLoaded, accIter,
      dotPrecision, /*maxNumImpreciseAcc=*/0);

  // Yield accumulator for next K-iteration
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{dotResult});

  // After the K-loop — store result C tile
  builder.setInsertionPointAfter(forOp);
  auto finalAcc = forOp.getResult(0);  // f32 accumulator

  // Cast f32 accumulator to output type if needed
  mlir::Value storeVal = finalAcc;
  if (cElemType != f32Type) {
    auto cTileType = mlir::RankedTensorType::get({blockM, blockN}, cElemType);
    if (mlir::isa<mlir::FloatType>(cElemType)) {
      storeVal = builder.create<mlir::arith::TruncFOp>(loc, cTileType, finalAcc);
    } else if (mlir::isa<mlir::IntegerType>(cElemType)) {
      storeVal = builder.create<mlir::arith::FPToSIOp>(loc, cTileType, finalAcc);
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
                                                const std::vector<LongType>& biasShape) {
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
    // BHSD: [batch, numKvHeads, seqK, headDim] — use kvHeadIdx for GQA
    auto kvStride0 = builder.create<mlir::arith::MulIOp>(loc, numKvHeadsConst,
        builder.create<mlir::arith::MulIOp>(loc, seqKConst, headDimConst));
    auto kvStride1 = builder.create<mlir::arith::MulIOp>(loc, seqKConst, headDimConst);
    kvBase = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::MulIOp>(loc, batchIdx, kvStride0),
        builder.create<mlir::arith::MulIOp>(loc, kvHeadIdx, kvStride1));
    kvRowStride = headDimConst;
  }

  // Load Q tile [BLOCK_M, headDim]
  // Q pointer offsets: qBase + qIndices[:, None] * headDim + rangeHd[None, :]
  auto qMExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, qIndices, 1);  // [BM, 1]
  auto hdExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, rangeHd, 0);   // [1, HD]

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

  // Load K tile [BLOCK_N, headDim]
  auto kNExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 1);  // [BN, 1]
  auto hdExpandedK = builder.create<mlir::triton::ExpandDimsOp>(loc, rangeHd, 0);  // [1, HD]

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

  mlir::Value kPtrsVal = kPtrs;
  auto kLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
      kPtrsVal, kMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  // Cast K to f32 for computation
  auto kLoaded = castTo(builder, loc, kLoadedRaw, f32Type);

  // QK^T = dot(q_scaled [BM, HD], k^T [HD, BN]) -> [BM, BN]
  auto transposeOrder = builder.getDenseI32ArrayAttr({1, 0});
  auto kTransposed = builder.create<mlir::triton::TransOp>(loc, kLoaded, transposeOrder);

  auto f32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, f32Type);
  auto i32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i32Type);
  auto qkZeroInit = splatConstantF32(builder, loc, f32BmBnType, 0.0f);
  auto qk = builder.create<mlir::triton::DotOp>(
      loc, f32BmBnType, qScaled, kTransposed, qkZeroInit,
      mlir::triton::InputPrecision::TF32, /*maxNumImpreciseAcc=*/0);

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
    int biasNumHeads = (biasRank >= 4) ? static_cast<int>(biasShape[1]) : 0;
    int biasSeqQ = (biasRank >= 4) ? static_cast<int>(biasShape[2]) :
                   (biasRank >= 3) ? static_cast<int>(biasShape[1]) : seqQ;
    int biasSeqK = (biasRank >= 4) ? static_cast<int>(biasShape[3]) :
                   (biasRank >= 3) ? static_cast<int>(biasShape[2]) : seqK;

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

    // Q row offsets within bias: qIndices * biasSeqK  → [BM, 1] → broadcast to [BM, BN]
    auto qBiasRowExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, qIndices, 1);  // [BM, 1]
    auto biasSeqKSplat = builder.create<mlir::triton::SplatOp>(loc,
        mlir::RankedTensorType::get({blockM, 1}, i32Type), biasSeqKConst);
    auto biasRowOffsets = builder.create<mlir::arith::MulIOp>(loc, qBiasRowExpanded, biasSeqKSplat);
    auto biasRowBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, biasRowOffsets);

    // K column offsets: kIndices → [1, BN] → broadcast to [BM, BN]
    auto kBiasColExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 0);  // [1, BN]
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
  auto vPtrTensorType = mlir::RankedTensorType::get({blockN, headDimPadded}, vPtrTypeAttn);
  auto vSplat = builder.create<mlir::triton::SplatOp>(loc, vPtrTensorType, vPtr);
  auto vPtrs = builder.create<mlir::triton::AddPtrOp>(loc, vPtrTensorType, vSplat, kFinalOffsets);

  mlir::Value vPtrsVal = vPtrs;
  auto vLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
      vPtrsVal, kMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  // Cast V to f32 for computation
  auto vLoaded = castTo(builder, loc, vLoadedRaw, f32Type);

  // acc_new = acc * splat(correction) + dot(p, V)
  // correction is [BM], need to broadcast to [BM, HD]
  auto correctionExp = builder.create<mlir::triton::ExpandDimsOp>(loc, correction, 1);  // [BM, 1]
  auto correctionBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, f32BmHdType, correctionExp);
  auto accScaled = builder.create<mlir::arith::MulFOp>(loc, accIter, correctionBroadcast);

  // dot(p[BM,BN], V[BN,HD]) -> [BM, HD]
  auto pv = builder.create<mlir::triton::DotOp>(
      loc, f32BmHdType, p, vLoaded, accScaled,
      mlir::triton::InputPrecision::TF32, /*maxNumImpreciseAcc=*/0);

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
            "headDim=%d scale=%f BM=%d BN=%d kvGroupSize=%d hasBias=%d",
            batchSize, numQHeads, numKvHeads, seqQ, seqK, headDim, scale, blockM, blockN, kvGroupSize,
            biasPtr ? 1 : 0);
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
