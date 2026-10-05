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
// Math operations that MLIR's MathToSPIRV conversion cannot lower for a Vulkan (Shader) target.
//
// GLSL.std.450, the extended instruction set MathToSPIRV targets for Vulkan, defines Exp, Log, Pow, Sin, Cos, Tanh,
// Cosh and Atan for 16- and 32-bit floats only, while the emitter catalogue admits DOUBLE wherever the device has
// shaderFloat64: such a kernel failed SPIR-V verification. MathToSPIRV registers no GLSL pattern for math.erf (its
// only one, CLErf, needs the OpenCL Kernel capability and is rejected for a Shader target), none for math.erfc, and
// none for math.trunc, so those fail GPUToSPIRV's full conversion for every float type. This pass rewrites them
// into arithmetic that SPIR-V does define (add, multiply, divide, fused multiply-add, comparisons, selects,
// floor/ceil/round and, for the powers of two, integer bit manipulation), before MathToSPIRV runs:
//
//   f64 exp, log, powf, sin, cos, tanh, cosh, atan   fdlibm algorithms (e_exp.c, e_log.c, k_sin.c, k_cos.c, s_atan.c,
//                                                     s_expm1.c, s_tanh.c, e_cosh.c)
//   f64 erf, erfc                                     fdlibm s_erf.c
//   f32 erf, erfc (f16 computes in f32)               fdlibm s_erff.c, with its own f32 exp
//   f16/f32/f64 trunc                                 select(x < 0, ceil(x), floor(x))
//
// Accuracy, measured against 113-bit references over the whole range of each function: exp, log, atan, erf within
// 1 ulp; erfc, expm1 within 2; sin, cos within 2.6 (up to three rounded steps in the pi/2 reduction); tanh, cosh
// within 3; powf within 1.5 ulp for |y| <= 100 and within about |y| / 100 ulp beyond (10 ulp at |y| = 1000), because
// the logarithm, its product with y and the exponential all carry two parts (a plain exp(y log|x|) is |y log|x||
// 2^-53 off) and the logarithm is good to 1.4e-18 absolute; the f32 erf and erfc within 2 ulp on a device that divides
// with IEEE accuracy.
//
// Contracts that follow from the instruction set:
//  * Fma is GLSL.std.450 Fma, whose precision Vulkan states as "inherited from FMul followed by FAdd", so it is not
//    required to be fused. Every f64-capable GPU executes a fused one, and two things rely on it: the pi/2 reduction
//    of sin and cos (within 2.6 ulp for |x| up to 2^46 with a fused Fma, only up to about 2^22 with an unfused one,
//    where the products n pi/2 are no longer exact) and the exact error terms of the double-double products in
//    powf (unfused, powf is back to the plain |y log|x|| 2^-53 accuracy). Beyond 2^46, and for +-inf, sin and cos
//    return NaN rather than a value with no correct digits (a Payne-Hanek reduction would be needed). The other
//    functions are insensitive to the difference.
//  * The powers of two and the exponent of log are built from the IEEE bit pattern, which needs Int64 for f64. A
//    device with Float64 but without Int64 computes the same quantities with exact f64 arithmetic (a ladder over
//    the binary digits of the exponent) and the same results; the module's spirv.target_env says which applies.
//  * Infinities and NaNs follow the device's float behavior (Vulkan only guarantees them with
//    SignedZeroInfNanPreserve); the expansions produce the C99 values for every special argument.
//
#include <graph/vulkan/VulkanOpLowerings.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN && defined(HAVE_MLIR) && HAVE_MLIR

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/ArrayRef.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace sd {
namespace graph {

namespace {

// Scalar arithmetic of one float type (f16, f32 or f64) at one insertion point. int64Available tells whether the device
// has Int64, which the f64 bit manipulation needs; f32 bit manipulation uses i32, which every device has.
class MathEmitter {
 public:
  MathEmitter(mlir::OpBuilder& builder, mlir::Location location, mlir::FloatType floatType, bool hasInt64)
      : b(builder), loc(location), type(floatType), int64Available(hasInt64) {}

  // FloatType::getWidth() is not const in this MLIR; Type::isF64() is.
  bool isDouble() const { return type.isF64(); }
  bool hasBits() const { return !isDouble() || int64Available; }

  mlir::Value c(double v) { return b.create<mlir::arith::ConstantOp>(loc, type, b.getFloatAttr(type, v)); }
  mlir::Value add(mlir::Value x, mlir::Value y) { return b.create<mlir::arith::AddFOp>(loc, x, y); }
  mlir::Value sub(mlir::Value x, mlir::Value y) { return b.create<mlir::arith::SubFOp>(loc, x, y); }
  mlir::Value mul(mlir::Value x, mlir::Value y) { return b.create<mlir::arith::MulFOp>(loc, x, y); }
  mlir::Value div(mlir::Value x, mlir::Value y) { return b.create<mlir::arith::DivFOp>(loc, x, y); }
  mlir::Value neg(mlir::Value x) { return b.create<mlir::arith::NegFOp>(loc, x); }
  mlir::Value fma(mlir::Value x, mlir::Value y, mlir::Value z) { return b.create<mlir::math::FmaOp>(loc, x, y, z); }
  mlir::Value abs(mlir::Value x) { return b.create<mlir::math::AbsFOp>(loc, x); }
  mlir::Value floor(mlir::Value x) { return b.create<mlir::math::FloorOp>(loc, x); }
  mlir::Value ceil(mlir::Value x) { return b.create<mlir::math::CeilOp>(loc, x); }
  mlir::Value roundEven(mlir::Value x) { return b.create<mlir::math::RoundEvenOp>(loc, x); }

  mlir::Value cmp(mlir::arith::CmpFPredicate predicate, mlir::Value x, mlir::Value y) {
    return b.create<mlir::arith::CmpFOp>(loc, predicate, x, y);
  }
  mlir::Value lt(mlir::Value x, mlir::Value y) { return cmp(mlir::arith::CmpFPredicate::OLT, x, y); }
  mlir::Value gt(mlir::Value x, mlir::Value y) { return cmp(mlir::arith::CmpFPredicate::OGT, x, y); }
  mlir::Value ge(mlir::Value x, mlir::Value y) { return cmp(mlir::arith::CmpFPredicate::OGE, x, y); }
  mlir::Value eq(mlir::Value x, mlir::Value y) { return cmp(mlir::arith::CmpFPredicate::OEQ, x, y); }
  mlir::Value ne(mlir::Value x, mlir::Value y) { return cmp(mlir::arith::CmpFPredicate::UNE, x, y); }
  mlir::Value isNan(mlir::Value x) { return cmp(mlir::arith::CmpFPredicate::UNO, x, x); }
  mlir::Value both(mlir::Value p, mlir::Value q) { return b.create<mlir::arith::AndIOp>(loc, p, q); }
  mlir::Value either(mlir::Value p, mlir::Value q) { return b.create<mlir::arith::OrIOp>(loc, p, q); }
  mlir::Value select(mlir::Value condition, mlir::Value t, mlir::Value f) {
    return b.create<mlir::arith::SelectOp>(loc, condition, t, f);
  }

  // ((c[0] x + c[1]) x + c[2]) ... + c[n-1]
  mlir::Value horner(mlir::Value x, llvm::ArrayRef<double> highestFirst) {
    mlir::Value acc;
    for (double coefficient : highestFirst) acc = acc ? fma(acc, x, c(coefficient)) : c(coefficient);
    return acc;
  }
  // c[0] + x (c[1] + x (c[2] + ...))
  mlir::Value hornerLowestFirst(mlir::Value x, llvm::ArrayRef<double> lowestFirst) {
    mlir::Value acc;
    for (size_t i = lowestFirst.size(); i-- > 0;) acc = acc ? fma(acc, x, c(lowestFirst[i])) : c(lowestFirst[i]);
    return acc;
  }

  // Integers as wide as the float: the IEEE bit pattern. Only called when hasBits().
  mlir::Type bitsType() { return b.getIntegerType(type.getWidth()); }
  mlir::Value ci(int64_t v) { return b.create<mlir::arith::ConstantOp>(loc, bitsType(), b.getIntegerAttr(bitsType(), v)); }
  mlir::Value bits(mlir::Value x) { return b.create<mlir::arith::BitcastOp>(loc, bitsType(), x); }
  mlir::Value fromBits(mlir::Value i) { return b.create<mlir::arith::BitcastOp>(loc, type, i); }
  mlir::Value toInt(mlir::Value x) { return b.create<mlir::arith::FPToSIOp>(loc, bitsType(), x); }
  mlir::Value toFloat(mlir::Value i) { return b.create<mlir::arith::SIToFPOp>(loc, type, i); }
  mlir::Value iadd(mlir::Value x, mlir::Value y) { return b.create<mlir::arith::AddIOp>(loc, x, y); }
  mlir::Value isub(mlir::Value x, mlir::Value y) { return b.create<mlir::arith::SubIOp>(loc, x, y); }
  mlir::Value iand(mlir::Value x, mlir::Value y) { return b.create<mlir::arith::AndIOp>(loc, x, y); }
  mlir::Value ior(mlir::Value x, mlir::Value y) { return b.create<mlir::arith::OrIOp>(loc, x, y); }
  mlir::Value ishl(mlir::Value x, mlir::Value y) { return b.create<mlir::arith::ShLIOp>(loc, x, y); }
  mlir::Value ishr(mlir::Value x, mlir::Value y) { return b.create<mlir::arith::ShRSIOp>(loc, x, y); }

  // 2^k for an integral k with |k| <= 1023 (f64) or |k| <= 126 (f32): the exponent field, or without Int64 the exact
  // product of 2^(+-2^j) over the binary digits of |k| (every partial product is a power of two within range).
  mlir::Value pow2(mlir::Value k) {
    if (hasBits()) {
      const int64_t bias = isDouble() ? 1023 : 127;
      const int64_t mantissaBits = isDouble() ? 52 : 23;
      return fromBits(ishl(iadd(toInt(k), ci(bias)), ci(mantissaBits)));
    }
    mlir::Value remaining = abs(k);
    mlir::Value scale = c(1.0);
    for (int j = 9; j >= 0; --j) {
      mlir::Value digit = ge(remaining, c(static_cast<double>(1 << j)));
      remaining = select(digit, sub(remaining, c(static_cast<double>(1 << j))), remaining);
      scale = mul(scale, select(digit, c(std::ldexp(1.0, 1 << j)), c(1.0)));
    }
    return select(lt(k, c(0.0)), div(c(1.0), scale), scale);
  }

  // v = mantissa 2^exponent, mantissa in [1, 2), for a positive normal f64 v. Without Int64 the exponent is found by
  // binary lifting over f64 compares: scale down by 2^(2^j) while v >= 2^(2^j), then up while v 2^(2^j) < 2.
  void splitExponent(mlir::Value v, mlir::Value& mantissa, mlir::Value& exponent) {
    if (hasBits()) {
      mlir::Value word = bits(v);
      mlir::Value biased = iand(ishr(word, ci(52)), ci(0x7ff));
      exponent = toFloat(isub(biased, ci(1023)));
      mantissa = fromBits(ior(iand(word, ci(0x000FFFFFFFFFFFFFLL)), ci(0x3FF0000000000000LL)));
      return;
    }
    exponent = c(0.0);
    for (int j = 9; j >= 0; --j) {
      mlir::Value big = ge(v, c(std::ldexp(1.0, 1 << j)));
      v = select(big, mul(v, c(std::ldexp(1.0, -(1 << j)))), v);
      exponent = add(exponent, select(big, c(static_cast<double>(1 << j)), c(0.0)));
    }
    for (int j = 9; j >= 0; --j) {
      mlir::Value lifted = mul(v, c(std::ldexp(1.0, 1 << j)));
      mlir::Value small = lt(lifted, c(2.0));
      v = select(small, lifted, v);
      exponent = sub(exponent, select(small, c(static_cast<double>(1 << j)), c(0.0)));
    }
    mantissa = v;
  }

  mlir::OpBuilder& b;
  mlir::Location loc;
  mlir::FloatType type;
  bool int64Available;
};

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------------------------------------------
// exp
// ---------------------------------------------------------------------------------------------------------------

constexpr double kLn2Hi = 6.93147180369123816490e-01;  // 32 significant bits: k * kLn2Hi is exact
constexpr double kLn2Lo = 1.90821492927058770002e-10;
constexpr double kInvLn2 = 1.44269504088896338700e+00;
constexpr double kExpOverflow = 7.09782712893383973096e+02;
constexpr double kExpUnderflow = -7.45133219101941108420e+02;
// f32: ln2 split into a 15-bit part (k * kLn2Hi32 is exact for |k| < 738) and the rest
constexpr double kLn2Hi32 = 22713.0 / 32768.0;
constexpr double kLn2Lo32 = 1.428606765330187e-06;
constexpr double kExpOverflow32 = 8.872283905206835e+01;
constexpr double kExpUnderflow32 = -1.0397207708399179e+02;

// Taylor series of exp(r) on |r| <= ln2 / 2, highest degree first: to r^13 for f64 (remainder below 5e-18 relative)
// and to r^7 for f32 (below 6e-9, under a tenth of an f32 ulp)
constexpr double kTaylor13[] = {1.0 / 6227020800.0, 1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0,
                                1.0 / 362880.0,     1.0 / 40320.0,     1.0 / 5040.0,     1.0 / 720.0,
                                1.0 / 120.0,        1.0 / 24.0,        1.0 / 6.0,        0.5,
                                1.0,                1.0};
constexpr double kTaylor7[] = {1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0};

// exp(hi + lo) for hi + lo in the finite range, lo a null Value when there is no second part: x = k ln2 + r with
// |r| <= ln2 / 2 taken from the sum (hi exact times k ln2Hi exact, then lo, then k ln2Lo), exp(r) by its Taylor
// series, times 2^k applied in two halves so that each power of two is a normal number.
mlir::Value expParts(MathEmitter& m, mlir::Value hi, mlir::Value lo) {
  const bool wide = m.isDouble();
  mlir::Value total = lo ? m.add(hi, lo) : hi;
  mlir::Value k = m.roundEven(m.mul(total, m.c(kInvLn2)));
  mlir::Value r = m.fma(m.neg(k), m.c(wide ? kLn2Hi : kLn2Hi32), hi);
  if (lo) r = m.add(r, lo);
  r = m.fma(m.neg(k), m.c(wide ? kLn2Lo : kLn2Lo32), r);
  mlir::Value p = wide ? m.horner(r, kTaylor13) : m.horner(r, kTaylor7);
  mlir::Value kHalf = m.floor(m.mul(k, m.c(0.5)));
  mlir::Value result = m.mul(m.mul(p, m.pow2(kHalf)), m.pow2(m.sub(k, kHalf)));
  result = m.select(m.gt(total, m.c(wide ? kExpOverflow : kExpOverflow32)), m.c(kInf), result);
  result = m.select(m.lt(total, m.c(wide ? kExpUnderflow : kExpUnderflow32)), m.c(0.0), result);
  return m.select(m.isNan(total), total, result);
}

mlir::Value expF64(MathEmitter& m, mlir::Value x) { return expParts(m, x, mlir::Value()); }

// ---------------------------------------------------------------------------------------------------------------
// log
// ---------------------------------------------------------------------------------------------------------------

// x = 2^e (1 + f) with 1 + f in [sqrt(2)/2, sqrt(2)), s = f / (2 + f) and the e_log.c polynomial R(s^2) =
// Lg1 s^2 + ... + Lg7 s^14, for a positive finite x (subnormals scaled by 2^54 first). Other arguments give values
// the callers replace.
struct LogTerms {
  mlir::Value e, f, s, polynomial;
};

LogTerms logTerms(MathEmitter& m, mlir::Value x) {
  mlir::Value subnormal = m.lt(x, m.c(2.2250738585072014e-308));
  mlir::Value scaled = m.select(subnormal, m.mul(x, m.c(1.80143985094819840000e+16)), x);  // x 2^54
  mlir::Value mantissa, exponent;
  m.splitExponent(scaled, mantissa, exponent);
  mlir::Value large = m.gt(mantissa, m.c(1.41421356237309504880));
  mantissa = m.select(large, m.mul(mantissa, m.c(0.5)), mantissa);
  LogTerms t;
  t.e = m.add(exponent, m.select(large, m.c(1.0), m.c(0.0)));
  t.e = m.sub(t.e, m.select(subnormal, m.c(54.0), m.c(0.0)));
  t.f = m.sub(mantissa, m.c(1.0));
  t.s = m.div(t.f, m.add(m.c(2.0), t.f));
  mlir::Value z = m.mul(t.s, t.s);
  mlir::Value w = m.mul(z, z);
  mlir::Value t1 = m.mul(w, m.horner(w, {1.531383769920937332e-01, 2.222219843214978396e-01,
                                         3.999999999940941908e-01}));
  mlir::Value t2 = m.mul(z, m.horner(w, {1.479819860511658591e-01, 1.818357216161805012e-01,
                                         2.857142874366239149e-01, 6.666666666666735130e-01}));
  t.polynomial = m.add(t2, t1);
  return t;
}

// fdlibm's log(1 + f) = f - (hfsq - s (hfsq + R)) with hfsq = f^2 / 2 (e_log.c), plus e ln2
mlir::Value logF64(MathEmitter& m, mlir::Value x) {
  LogTerms t = logTerms(m, x);
  mlir::Value hfsq = m.mul(m.c(0.5), m.mul(t.f, t.f));
  mlir::Value result =
      m.sub(m.mul(t.e, m.c(kLn2Hi)),
            m.sub(m.sub(hfsq, m.add(m.mul(t.s, m.add(hfsq, t.polynomial)), m.mul(t.e, m.c(kLn2Lo)))), t.f));
  result = m.select(m.eq(x, m.c(0.0)), m.c(-kInf), result);
  result = m.select(m.lt(x, m.c(0.0)), m.c(kNaN), result);
  result = m.select(m.eq(x, m.c(kInf)), x, result);
  return m.select(m.isNan(x), x, result);
}

// a + b = sum + error exactly
void twoSum(MathEmitter& m, mlir::Value a, mlir::Value b, mlir::Value& sum, mlir::Value& error) {
  sum = m.add(a, b);
  mlir::Value virtualB = m.sub(sum, a);
  error = m.add(m.sub(a, m.sub(sum, virtualB)), m.sub(b, virtualB));
}

// log(x) as hi + lo to about 3e-18 relative, for a positive finite x = 2^e (1 + f): log(x) = e ln2 + f - hfsq +
// s (hfsq + R(s^2)) with hfsq = f^2 / 2 and s = f / (2 + f) (e_log.c). e ln2Hi (exact), f and hfsq are summed exactly,
// and the rounding of every other term is carried instead of dropped: s as sHi + sLo (the quotient and its residual), s^2
// as zHi + zLo for the argument of R (R(zHi) + R'(zHi) zLo), hfsq + R as qHi + qLo, and s qHi as a product and its
// fused-multiply-add error. The plain evaluation leaves s (hfsq + R) rounded 3-4 times, 1e-17 relative to log(x).
void logParts(MathEmitter& m, mlir::Value x, mlir::Value& hi, mlir::Value& lo) {
  LogTerms t = logTerms(m, x);
  mlir::Value dHi, dLo;
  twoSum(m, m.c(2.0), t.f, dHi, dLo);  // 2 + f exactly
  mlir::Value sHi = m.div(t.f, dHi);
  mlir::Value sLo = m.div(m.sub(m.fma(m.neg(sHi), dHi, t.f), m.mul(sHi, dLo)), dHi);  // (f - sHi (dHi + dLo)) / dHi
  mlir::Value zHi = m.mul(sHi, sHi);
  mlir::Value zLo = m.add(m.fma(sHi, sHi, m.neg(zHi)), m.mul(m.c(2.0), m.mul(sHi, sLo)));
  mlir::Value w = m.mul(zHi, zHi);
  mlir::Value t1 = m.mul(w, m.horner(w, {1.531383769920937332e-01, 2.222219843214978396e-01,
                                         3.999999999940941908e-01}));
  mlir::Value t2 = m.mul(zHi, m.horner(w, {1.479819860511658591e-01, 1.818357216161805012e-01,
                                           2.857142874366239149e-01, 6.666666666666735130e-01}));
  // dR/dz = Lg1 + 2 Lg2 z (the next term, 3 Lg3 z^2, is 1e-3 of the first)
  mlir::Value slope = m.fma(m.c(2.0 * 3.999999999940941908e-01), zHi, m.c(6.666666666666735130e-01));
  mlir::Value polynomial = m.add(m.add(t2, t1), m.mul(slope, zLo));
  mlir::Value square = m.mul(t.f, t.f);
  mlir::Value squareError = m.fma(t.f, t.f, m.neg(square));  // f f = square + squareError exactly
  mlir::Value hfsqHi = m.mul(m.c(0.5), square);
  mlir::Value hfsqLo = m.mul(m.c(0.5), squareError);
  mlir::Value qHi, qError;
  twoSum(m, hfsqHi, polynomial, qHi, qError);  // hfsq + R = qHi + qLo
  mlir::Value qLo = m.add(qError, hfsqLo);
  mlir::Value product = m.mul(sHi, qHi);
  mlir::Value productError = m.fma(sHi, qHi, m.neg(product));
  mlir::Value base = m.mul(t.e, m.c(kLn2Hi));
  mlir::Value tHi, tError, uHi, uError, vHi, vError;
  twoSum(m, base, t.f, tHi, tError);
  twoSum(m, tHi, m.neg(hfsqHi), uHi, uError);
  twoSum(m, uHi, product, vHi, vError);
  mlir::Value small = m.add(m.add(m.add(tError, uError), vError),
                            m.add(m.add(productError, m.add(m.mul(sLo, qHi), m.mul(sHi, qLo))),
                                  m.sub(m.mul(t.e, m.c(kLn2Lo)), hfsqLo)));
  twoSum(m, vHi, small, hi, lo);
}

// ---------------------------------------------------------------------------------------------------------------
// pow
// ---------------------------------------------------------------------------------------------------------------

// pow(x, y) = exp(y log|x|) with the product y log|x| and the exponential carried to double-double (the relative
// error of a plain exp(y log|x|) is |y log|x|| 2^-53, hundreds of ulp for large exponents). The signs and special
// values are C99's: negative for a negative x (including -0) and an odd integral y, NaN for a finite negative x and a
// non-integral y, pow(x, 0) = pow(1, y) = pow(-1, +-inf) = 1.
mlir::Value powF64(MathEmitter& m, mlir::Value x, mlir::Value y) {
  mlir::Value infinity = m.c(kInf);
  mlir::Value ax = m.abs(x);
  mlir::Value logHi, logLo;
  logParts(m, ax, logHi, logLo);
  logHi = m.select(m.eq(ax, m.c(0.0)), m.c(-kInf), logHi);
  logHi = m.select(m.eq(ax, infinity), infinity, logHi);
  logHi = m.select(m.isNan(x), x, logHi);
  mlir::Value productHi = m.mul(y, logHi);
  mlir::Value productLo = m.add(m.fma(y, logHi, m.neg(productHi)), m.mul(y, logLo));
  productLo = m.select(m.lt(m.abs(productHi), infinity), productLo, m.c(0.0));
  mlir::Value magnitude = expParts(m, productHi, productLo);

  mlir::Value integral = m.eq(m.floor(y), y);
  mlir::Value notIntegral = m.ne(m.floor(y), y);
  mlir::Value half = m.mul(y, m.c(0.5));
  mlir::Value odd = m.both(integral, m.ne(m.floor(half), half));
  mlir::Value zero = m.c(0.0);
  mlir::Value negativeZero = m.both(m.eq(x, zero), m.lt(m.div(m.c(1.0), x), zero));
  mlir::Value negativeBase = m.either(m.lt(x, zero), negativeZero);
  mlir::Value result = m.select(m.both(negativeBase, odd), m.neg(magnitude), magnitude);
  mlir::Value domainError = m.both(m.both(m.lt(x, zero), m.lt(ax, infinity)), notIntegral);
  result = m.select(domainError, m.c(kNaN), result);
  result = m.select(m.eq(y, zero), m.c(1.0), result);
  result = m.select(m.eq(x, m.c(1.0)), m.c(1.0), result);
  return m.select(m.both(m.eq(x, m.c(-1.0)), m.eq(m.abs(y), infinity)), m.c(1.0), result);
}

// ---------------------------------------------------------------------------------------------------------------
// expm1, tanh, cosh
// ---------------------------------------------------------------------------------------------------------------

// expm1 without cancellation near 0: the Taylor series to x^18 below |x| = 0.7, exp(x) - 1 above (where the
// subtraction loses at most a bit).
mlir::Value expm1F64(MathEmitter& m, mlir::Value x) {
  mlir::Value series = m.mul(
      x, m.horner(x, {1.0 / 6402373705728000.0, 1.0 / 355687428096000.0, 1.0 / 20922789888000.0,
                      1.0 / 1307674368000.0, 1.0 / 87178291200.0, 1.0 / 6227020800.0, 1.0 / 479001600.0,
                      1.0 / 39916800.0, 1.0 / 3628800.0, 1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0,
                      1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0}));
  mlir::Value direct = m.sub(expF64(m, x), m.c(1.0));
  return m.select(m.lt(m.abs(x), m.c(0.7)), series, direct);
}

// fdlibm s_tanh.c: 1 - 2 / (expm1(2|x|) + 2) for |x| >= 1, -expm1(-2|x|) / (expm1(-2|x|) + 2) below, 1 past 22.
// One expm1 serves both: its argument is -2|x| below 1 and 2|x| from there. The sign is restored last; tanh(+-0) =
// +-0.
mlir::Value tanhF64(MathEmitter& m, mlir::Value x) {
  mlir::Value ax = m.abs(x);
  mlir::Value belowOne = m.lt(ax, m.c(1.0));
  mlir::Value t = expm1F64(m, m.select(belowOne, m.mul(m.c(-2.0), ax), m.mul(m.c(2.0), ax)));
  mlir::Value denominator = m.add(t, m.c(2.0));
  mlir::Value small = m.div(m.neg(t), denominator);
  mlir::Value large = m.sub(m.c(1.0), m.div(m.c(2.0), denominator));
  mlir::Value result = m.select(belowOne, small, large);
  result = m.select(m.gt(ax, m.c(22.0)), m.c(1.0), result);
  mlir::Value signedResult = m.select(m.lt(x, m.c(0.0)), m.neg(result), result);
  return m.select(m.eq(x, m.c(0.0)), x, signedResult);
}

// cosh: (e + 1/e) / 2 with e = exp|x|; near overflow (|x| >= 709) as (exp(|x|/2) / 2) exp(|x|/2), finite to 710.47.
// One exponential serves both: of |x| below 709, of |x| / 2 from there.
mlir::Value coshF64(MathEmitter& m, mlir::Value x) {
  mlir::Value ax = m.abs(x);
  mlir::Value moderateRange = m.lt(ax, m.c(709.0));
  mlir::Value e = expF64(m, m.select(moderateRange, ax, m.mul(m.c(0.5), ax)));
  mlir::Value moderate = m.mul(m.c(0.5), m.add(e, m.div(m.c(1.0), e)));
  mlir::Value large = m.mul(m.mul(m.c(0.5), e), e);
  return m.select(moderateRange, moderate, large);
}

// ---------------------------------------------------------------------------------------------------------------
// sin, cos
// ---------------------------------------------------------------------------------------------------------------

// Past this the pi/2 reduction below loses its quotient (|n| estimated from x 2/pi in f64 is off by more than 0.01)
constexpr double kTrigLimit = 70368744177664.0;  // 2^46

// x = n pi/2 + r with |r| <= pi/4: n rounded, pi/2 in four Cody-Waite parts (three of 33 bits, then the tail), each
// product subtracted with one rounding (fused multiply-add). quadrant = n mod 4 in {0, 1, 2, 3}, computed in f64
// (exact for |n| < 2^53) so that no integer conversion is needed.
void reducePiOver2(MathEmitter& m, mlir::Value x, mlir::Value& r, mlir::Value& quadrant) {
  mlir::Value n = m.roundEven(m.mul(x, m.c(6.36619772367581382433e-01)));
  mlir::Value negN = m.neg(n);
  r = m.fma(negN, m.c(1.57079632673412561417e+00), x);
  r = m.fma(negN, m.c(6.07710050630396597660e-11), r);
  r = m.fma(negN, m.c(2.02226624871116645580e-21), r);
  r = m.fma(negN, m.c(8.47842766036889956997e-32), r);
  quadrant = m.sub(n, m.mul(m.c(4.0), m.floor(m.mul(n, m.c(0.25)))));
}

// fdlibm k_sin.c / k_cos.c on |r| <= pi/4
mlir::Value kernelSin(MathEmitter& m, mlir::Value r) {
  mlir::Value z = m.mul(r, r);
  mlir::Value tail = m.horner(z, {1.58969099521155010221e-10, -2.50507602534068634195e-08,
                                  2.75573137070700676789e-06, -1.98412698298579493134e-04,
                                  8.33333333332248946124e-03});
  mlir::Value v = m.mul(z, r);
  return m.add(r, m.mul(v, m.add(m.c(-1.66666666666666324348e-01), m.mul(z, tail))));
}

mlir::Value kernelCos(MathEmitter& m, mlir::Value r) {
  mlir::Value z = m.mul(r, r);
  mlir::Value polynomial = m.mul(z, m.horner(z, {-1.13596475577881948265e-11, 2.08757232129817482790e-09,
                                                 -2.75573143513906633035e-07, 2.48015872894767294178e-05,
                                                 -1.38888888888741095749e-03, 4.16666666666666019037e-02}));
  mlir::Value hz = m.mul(m.c(0.5), z);
  mlir::Value w = m.sub(m.c(1.0), hz);
  // (1 - w) - hz recovers the rounding error of w
  return m.add(w, m.add(m.sub(m.sub(m.c(1.0), w), hz), m.mul(z, polynomial)));
}

// sin and cos of |x| <= 2^46, NaN beyond (and for +-inf); sin(+-0) = +-0
mlir::Value sinCosF64(MathEmitter& m, mlir::Value x, bool cosine) {
  mlir::Value r, quadrant;
  reducePiOver2(m, x, r, quadrant);
  mlir::Value s = kernelSin(m, r);
  mlir::Value cc = kernelCos(m, r);
  // sin: s, c, -s, -c by quadrant; cos: c, -s, -c, s
  mlir::Value q0 = m.eq(quadrant, m.c(0.0)), q1 = m.eq(quadrant, m.c(1.0)), q2 = m.eq(quadrant, m.c(2.0));
  mlir::Value result;
  if (cosine) {
    result = m.select(q0, cc, m.select(q1, m.neg(s), m.select(q2, m.neg(cc), s)));
  } else {
    result = m.select(q0, s, m.select(q1, cc, m.select(q2, m.neg(s), m.neg(cc))));
    result = m.select(m.eq(x, m.c(0.0)), x, result);
  }
  return m.select(m.gt(m.abs(x), m.c(kTrigLimit)), m.c(kNaN), result);
}

// ---------------------------------------------------------------------------------------------------------------
// atan
// ---------------------------------------------------------------------------------------------------------------

// fdlibm s_atan.c: |x| reduced to t on one of five intervals, atan(|x|) = atanhi[id] + atan(t) (with atanlo[id]
// carrying the rest of the constant), the sign restored at the end; atan(+-0) = +-0.
mlir::Value atanF64(MathEmitter& m, mlir::Value x) {
  mlir::Value ax = m.abs(x);
  mlir::Value below7_16 = m.lt(ax, m.c(0.4375));
  mlir::Value below11_16 = m.lt(ax, m.c(0.6875));
  mlir::Value below19_16 = m.lt(ax, m.c(1.1875));
  mlir::Value below39_16 = m.lt(ax, m.c(2.4375));
  auto pick = [&](double tiny, double first, double second, double third, double huge) {
    return m.select(below7_16, m.c(tiny),
                    m.select(below11_16, m.c(first),
                             m.select(below19_16, m.c(second), m.select(below39_16, m.c(third), m.c(huge)))));
  };
  // t = numerator / denominator per interval: ax; (2ax - 1)/(2 + ax); (ax - 1)/(ax + 1); (ax - 1.5)/(1 + 1.5ax); -1/ax
  mlir::Value numerator = m.select(
      below7_16, ax,
      m.select(below11_16, m.sub(m.mul(m.c(2.0), ax), m.c(1.0)),
               m.select(below19_16, m.sub(ax, m.c(1.0)),
                        m.select(below39_16, m.sub(ax, m.c(1.5)), m.c(-1.0)))));
  mlir::Value denominator = m.select(
      below7_16, m.c(1.0),
      m.select(below11_16, m.add(m.c(2.0), ax),
               m.select(below19_16, m.add(ax, m.c(1.0)),
                        m.select(below39_16, m.add(m.c(1.0), m.mul(m.c(1.5), ax)), ax))));
  mlir::Value t = m.div(numerator, denominator);
  mlir::Value hi = pick(0.0, 4.63647609000806093515e-01, 7.85398163397448278999e-01, 9.82793723247329054082e-01,
                        1.57079632679489655800e+00);
  mlir::Value lo = pick(0.0, 2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17,
                        6.12323399573676603587e-17);

  mlir::Value z = m.mul(t, t);
  mlir::Value w = m.mul(z, z);
  mlir::Value s1 = m.mul(z, m.horner(w, {1.62858201153657823623e-02, 4.97687799461593236017e-02,
                                         6.66107313738753120669e-02, 9.09088713343650656196e-02,
                                         1.42857142725034663711e-01, 3.33333333333329318027e-01}));
  mlir::Value s2 = m.mul(w, m.horner(w, {-3.65315727442169155270e-02, -5.83357013379057348645e-02,
                                         -7.69187620504482999495e-02, -1.11111104054623557880e-01,
                                         -1.99999999998764832476e-01}));
  mlir::Value correction = m.mul(t, m.add(s1, s2));
  mlir::Value small = m.sub(t, correction);
  mlir::Value reduced = m.sub(hi, m.sub(m.sub(correction, lo), t));
  mlir::Value result = m.select(below7_16, small, reduced);
  mlir::Value signedResult = m.select(m.lt(x, m.c(0.0)), m.neg(result), result);
  return m.select(m.eq(x, m.c(0.0)), x, signedResult);
}

// ---------------------------------------------------------------------------------------------------------------
// erf, erfc (fdlibm s_erf.c for f64, s_erff.c for f32)
// ---------------------------------------------------------------------------------------------------------------

struct ErfTable {
  double erx;          // erf(1) rounded to a few bits: erf(x) = erx + P(|x| - 1) / Q(|x| - 1) on [0.84375, 1.25]
  double oneMinusErx;  // exact
  double pp[5];        // erf(x) = x + x P(x^2) / Q(x^2) on |x| < 0.84375
  double qq[6];
  double pa[7];
  double qa[7];
  double ra[8];        // erfc(x) = exp(-x^2 - 0.5625 + R(1/x^2) / S(1/x^2)) / x on [1.25, 1/0.35]
  double sa[9];
  double rb[8];        // and on [1/0.35, 28]
  double sb[9];
  double splitErf;     // where erf switches from the ra/sa to the rb/sb fit
  double splitErfc;    // and erfc
  double gridScale;    // z = |x| truncated to a multiple of 1/gridScale: z^2 and z^2 + 0.5625 are exact
  double tailLimit;    // |x| from which the tail is zero (erfc underflows)
};

constexpr ErfTable kErf64 = {
    8.45062911510467529297e-01,
    1.0 - 8.45062911510467529297e-01,
    {1.28379167095512558561e-01, -3.25042107247001499370e-01, -2.84817495755985104766e-02,
     -5.77027029648944159157e-03, -2.37630166566501626084e-05},
    {1.0, 3.97917223959155352819e-01, 6.50222499887672944485e-02, 5.08130628187576562776e-03,
     1.32494738004321644526e-04, -3.96022827877536812320e-06},
    {-2.36211856075265944077e-03, 4.14856118683748331666e-01, -3.72207876035701323847e-01,
     3.18346619901161753674e-01, -1.10894694282396677476e-01, 3.54783043256182359371e-02,
     -2.16637559486879084300e-03},
    {1.0, 1.06420880400844228286e-01, 5.40397917702171048937e-01, 7.18286544141962662868e-02,
     1.26171219808761642112e-01, 1.36370839120290507362e-02, 1.19844998467991074170e-02},
    {-9.86494403484714822705e-03, -6.93858572707181764372e-01, -1.05586262253232909814e+01,
     -6.23753324503260060396e+01, -1.62396669462573470355e+02, -1.84605092906711035994e+02,
     -8.12874355063065934246e+01, -9.81432934416914548592e+00},
    {1.0, 1.96512716674392571292e+01, 1.37657754143519042600e+02, 4.34565877475229228821e+02,
     6.45387271733267880336e+02, 4.29008140027567833386e+02, 1.08635005541779435134e+02,
     6.57024977031928170135e+00, -6.04244152148580987438e-02},
    {-9.86494292470009928597e-03, -7.99283237680523006574e-01, -1.77579549177547519889e+01,
     -1.60636384855821916062e+02, -6.37566443368389627722e+02, -1.02509513161107724954e+03,
     -4.83519191608651397019e+02, 0.0},
    {1.0, 3.03380607434824582924e+01, 3.25792512996573918826e+02, 1.53672958608443695994e+03,
     3.19985821950859553908e+03, 2.55305040643316442583e+03, 4.74528541206955367215e+02,
     -2.24409524465858183362e+01, 0.0},
    2.8571434020996094,  // 0x4006DB6E00000000
    2.8571414947509766,  // 0x4006DB6D00000000
    1048576.0,           // 2^20
    28.0};

constexpr ErfTable kErf32 = {
    8.45062911510467529297e-01,  // 0x3f58560b, the same value as the f64 erx
    1.0 - 8.45062911510467529297e-01,
    {1.2837916613e-01, -3.2504209876e-01, -2.8481749818e-02, -5.7702702470e-03, -2.3763017452e-05},
    {1.0, 3.9791721106e-01, 6.5022252500e-02, 5.0813062117e-03, 1.3249473704e-04, -3.9602282413e-06},
    {-2.3621185683e-03, 4.1485610604e-01, -3.7220788002e-01, 3.1834661961e-01, -1.1089469492e-01,
     3.5478305072e-02, -2.1663755178e-03},
    {1.0, 1.0642088205e-01, 5.4039794207e-01, 7.1828655899e-02, 1.2617121637e-01, 1.3637083583e-02,
     1.1984500103e-02},
    {-9.8649440333e-03, -6.9385856390e-01, -1.0558626175e+01, -6.2375331879e+01, -1.6239666748e+02,
     -1.8460508728e+02, -8.1287437439e+01, -9.8143291473e+00},
    {1.0, 1.9651271820e+01, 1.3765776062e+02, 4.3456588745e+02, 6.4538726807e+02, 4.2900814819e+02,
     1.0863500214e+02, 6.5702495575e+00, -6.0424413532e-02},
    {-9.8649431020e-03, -7.9928326607e-01, -1.7757955551e+01, -1.6063638306e+02, -6.3756646729e+02,
     -1.0250950928e+03, -4.8351919556e+02, 0.0},
    {1.0, 3.0338060379e+01, 3.2579251099e+02, 1.5367296143e+03, 3.1998581543e+03, 2.5530502930e+03,
     4.7452853394e+02, -2.2440952301e+01, 0.0},
    2.857142686843872,  // 0x4036db6d
    2.857142686843872,
    256.0,  // z has at most 12 bits below 16, z^2 at most 24
    12.0};  // erfc(12) = 1.4e-63, far below the smallest f32

// P(x^2) / Q(x^2) for |x| < 0.84375
mlir::Value erfSmallRatio(MathEmitter& m, const ErfTable& t, mlir::Value z) {
  return m.div(m.hornerLowestFirst(z, t.pp), m.hornerLowestFirst(z, t.qq));
}

// P(s) / Q(s), s = |x| - 1, for 0.84375 <= |x| < 1.25
mlir::Value erfMidRatio(MathEmitter& m, const ErfTable& t, mlir::Value ax) {
  mlir::Value s = m.sub(ax, m.c(1.0));
  return m.div(m.hornerLowestFirst(s, t.pa), m.hornerLowestFirst(s, t.qa));
}

// erfc(ax) for 1.25 <= ax < tailLimit (0 from there on, and for +inf): exp(-z^2 - 0.5625 + (z - ax)(z + ax) + R/S) /
// ax with z = ax truncated to a multiple of 1/gridScale. -z^2 - 0.5625 is exact, so the large part of the exponent has
// no rounding error; the rest, (z - ax)(z + ax) + R/S, is one rounding away from exact. One exponential of the
// sum, split in two parts, replaces fdlibm's product of two.
mlir::Value erfTail(MathEmitter& m, const ErfTable& t, mlir::Value ax, double split) {
  mlir::Value s = m.div(m.c(1.0), m.mul(ax, ax));
  mlir::Value useA = m.lt(ax, m.c(split));
  mlir::Value numerator, denominator;
  // coefficient-wise choice between the [1.25, 1/0.35] and the [1/0.35, 28] fit (zero-padded to the same degree)
  for (int i = 7; i >= 0; --i) {
    mlir::Value coefficient = m.select(useA, m.c(t.ra[i]), m.c(t.rb[i]));
    numerator = numerator ? m.fma(numerator, s, coefficient) : coefficient;
  }
  for (int i = 8; i >= 0; --i) {
    mlir::Value coefficient = m.select(useA, m.c(t.sa[i]), m.c(t.sb[i]));
    denominator = denominator ? m.fma(denominator, s, coefficient) : coefficient;
  }
  mlir::Value z = m.mul(m.floor(m.mul(ax, m.c(t.gridScale))), m.c(1.0 / t.gridScale));
  mlir::Value hi = m.sub(m.neg(m.mul(z, z)), m.c(0.5625));
  mlir::Value lo = m.add(m.mul(m.sub(z, ax), m.add(z, ax)), m.div(numerator, denominator));
  mlir::Value tail = m.div(expParts(m, hi, lo), ax);
  return m.select(m.ge(ax, m.c(t.tailLimit)), m.c(0.0), tail);
}

mlir::Value erfValue(MathEmitter& m, const ErfTable& t, mlir::Value x) {
  mlir::Value ax = m.abs(x);
  mlir::Value y = erfSmallRatio(m, t, m.mul(x, x));
  // x + x y, as (8x + x 8y) / 8 so that a subnormal x is rounded once
  mlir::Value small = m.mul(m.c(0.125), m.fma(x, m.mul(m.c(8.0), y), m.mul(m.c(8.0), x)));
  mlir::Value mid = m.add(m.c(t.erx), erfMidRatio(m, t, ax));
  mlir::Value tail = erfTail(m, t, ax, t.splitErf);
  mlir::Value big = m.select(m.ge(ax, m.c(6.0)), m.c(1.0), m.sub(m.c(1.0), tail));
  mlir::Value midOrBig = m.select(m.lt(ax, m.c(1.25)), mid, big);
  mlir::Value signedValue = m.select(m.lt(x, m.c(0.0)), m.neg(midOrBig), midOrBig);
  mlir::Value result = m.select(m.lt(ax, m.c(0.84375)), small, signedValue);
  return m.select(m.isNan(x), x, result);
}

mlir::Value erfcValue(MathEmitter& m, const ErfTable& t, mlir::Value x) {
  mlir::Value ax = m.abs(x);
  mlir::Value y = erfSmallRatio(m, t, m.mul(x, x));
  mlir::Value belowQuarter = m.lt(x, m.c(0.25));
  // 1 - erf(x) below 1/4, 1/2 - (x y + (x - 1/2)) from there to 0.84375 (the subtraction x - 1/2 is exact)
  mlir::Value small = m.select(belowQuarter, m.sub(m.c(1.0), m.add(x, m.mul(x, y))),
                               m.sub(m.c(0.5), m.add(m.mul(x, y), m.sub(x, m.c(0.5)))));
  mlir::Value mid = erfMidRatio(m, t, ax);
  mlir::Value midValue = m.select(m.ge(x, m.c(0.0)), m.sub(m.c(t.oneMinusErx), mid),
                                  m.add(m.c(1.0), m.add(m.c(t.erx), mid)));
  mlir::Value tail = erfTail(m, t, ax, t.splitErfc);
  mlir::Value tailValue = m.select(m.gt(x, m.c(0.0)), tail, m.sub(m.c(2.0), tail));
  mlir::Value result = m.select(m.lt(ax, m.c(0.84375)), small,
                                m.select(m.lt(ax, m.c(1.25)), midValue, tailValue));
  return m.select(m.isNan(x), x, result);
}

// ---------------------------------------------------------------------------------------------------------------
// trunc
// ---------------------------------------------------------------------------------------------------------------

// the integer part toward zero: floor for x >= 0 (and -0, NaN, +-inf pass through), ceil below
mlir::Value truncValue(MathEmitter& m, mlir::Value x) { return m.select(m.lt(x, m.c(0.0)), m.ceil(x), m.floor(x)); }

// ---------------------------------------------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------------------------------------------

// Whether this pass owns the operation: its result type is one MathToSPIRV cannot lower the operation for.
bool isTarget(mlir::Operation* op) {
  if (op->getNumResults() != 1) return false;
  auto type = llvm::dyn_cast<mlir::FloatType>(op->getResult(0).getType());
  if (!type) return false;
  const unsigned width = type.getWidth();
  const bool standard = type.isF16() || type.isF32() || type.isF64();
  if (!standard) return false;
  if (llvm::isa<mlir::math::TruncOp, mlir::math::ErfOp, mlir::math::ErfcOp>(op)) return true;
  if (width != 64) return false;
  return llvm::isa<mlir::math::ExpOp, mlir::math::LogOp, mlir::math::PowFOp, mlir::math::SinOp,
                   mlir::math::CosOp, mlir::math::TanhOp, mlir::math::CoshOp, mlir::math::AtanOp>(op);
}

struct VulkanF64MathExpansionPass
    : public mlir::PassWrapper<VulkanF64MathExpansionPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VulkanF64MathExpansionPass)

  llvm::StringRef getArgument() const override { return "vulkan-f64-math-expansion"; }
  llvm::StringRef getDescription() const override {
    return "Rewrite the math operations GLSL.std.450 cannot express for a Vulkan target (f64 exp/log/pow/sin/cos/"
           "tanh/cosh/atan, erf/erfc and trunc) into arithmetic SPIR-V defines";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::math::MathDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::spirv::TargetEnv targetEnv(mlir::spirv::lookupTargetEnvOrDefault(module.getOperation()));
    const bool hasFloat64 = targetEnv.allows(mlir::spirv::Capability::Float64);
    const bool hasInt64 = targetEnv.allows(mlir::spirv::Capability::Int64);

    std::vector<mlir::Operation*> targets;
    module.walk([&](mlir::Operation* op) {
      if (isTarget(op)) targets.push_back(op);
    });

    for (mlir::Operation* op : targets) {
      auto type = llvm::cast<mlir::FloatType>(op->getResult(0).getType());
      if (type.isF64() && !hasFloat64) {
        op->emitOpError("f64 math requires the SPIR-V Float64 capability of the target environment");
        signalPassFailure();
        return;
      }
      mlir::OpBuilder builder(op);
      mlir::Value x = op->getOperand(0);
      mlir::Value replacement;
      if (llvm::isa<mlir::math::TruncOp>(op)) {
        MathEmitter m(builder, op->getLoc(), type, hasInt64);
        replacement = truncValue(m, x);
      } else if (llvm::isa<mlir::math::ErfOp, mlir::math::ErfcOp>(op)) {
        const bool complementary = llvm::isa<mlir::math::ErfcOp>(op);
        if (type.isF64()) {
          MathEmitter m(builder, op->getLoc(), type, hasInt64);
          replacement = complementary ? erfcValue(m, kErf64, x) : erfValue(m, kErf64, x);
        } else {
          // f32 directly, f16 through f32
          mlir::FloatType wide = builder.getF32Type();
          MathEmitter m(builder, op->getLoc(), wide, hasInt64);
          mlir::Value widened = type.isF32() ? x : mlir::Value(builder.create<mlir::arith::ExtFOp>(op->getLoc(), wide, x));
          mlir::Value value = complementary ? erfcValue(m, kErf32, widened) : erfValue(m, kErf32, widened);
          replacement = type.isF32() ? value : mlir::Value(builder.create<mlir::arith::TruncFOp>(op->getLoc(), type, value));
        }
      } else {
        MathEmitter m(builder, op->getLoc(), type, hasInt64);
        if (llvm::isa<mlir::math::ExpOp>(op)) replacement = expF64(m, x);
        else if (llvm::isa<mlir::math::LogOp>(op)) replacement = logF64(m, x);
        else if (llvm::isa<mlir::math::PowFOp>(op)) replacement = powF64(m, x, op->getOperand(1));
        else if (llvm::isa<mlir::math::SinOp>(op)) replacement = sinCosF64(m, x, false);
        else if (llvm::isa<mlir::math::CosOp>(op)) replacement = sinCosF64(m, x, true);
        else if (llvm::isa<mlir::math::TanhOp>(op)) replacement = tanhF64(m, x);
        else if (llvm::isa<mlir::math::CoshOp>(op)) replacement = coshF64(m, x);
        else replacement = atanF64(m, x);
      }
      op->getResult(0).replaceAllUsesWith(replacement);
      op->erase();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createVulkanF64MathExpansionPass() {
  return std::make_unique<VulkanF64MathExpansionPass>();
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN && HAVE_MLIR
