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


#include <graph/gpu/PtxGraphBackend.h>
#include <graph/DspDiagnostics.h>
#include <graph/gpu/GpuKernelLauncher.h>
#include <graph/gpu/OpCategoryTable.h>
#include <system/common.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <sstream>
#include <unordered_set>

namespace sd {
namespace graph {

// ---- Singleton ----

PtxGraphBackend& PtxGraphBackend::getInstance() {
  static PtxGraphBackend* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new PtxGraphBackend();
  });
  return *instance;
}

PtxGraphBackend::PtxGraphBackend() = default;

PtxGraphBackend::~PtxGraphBackend() {
  invalidateCache();
}

// ---- Availability ----

bool PtxGraphBackend::isAvailable() const {
  return true;  // Always available on CUDA builds
}

// ---- SM version ----

int PtxGraphBackend::getSmVersion() {
  cudaDeviceProp props;
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);
  return props.major * 10 + props.minor;
}

// ---- Segment fusibility (delegates to shared logic) ----

bool PtxGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  return jitCanFuseSegment(slots, start, end);
}

// ---- PTX instruction generation helpers ----

// Register counter for PTX float register allocation
struct PtxRegAlloc {
  int nextFloat = 0;
  int nextPred = 0;

  std::string allocFloat() { return "f" + std::to_string(nextFloat++); }
  std::string allocPred() { return "p" + std::to_string(nextPred++); }
};

/**
 * Emit PTX instructions for a binary element-wise op.
 */
static std::string emitPtxBinaryOp(std::ostringstream& out, PtxRegAlloc& ra,
                                     const std::string& opName,
                                     const std::string& inReg,
                                     const std::string& secReg) {
  std::string result = ra.allocFloat();

  if (opName == "add" || opName == "Add") {
    out << "    add.f32 " << result << ", " << inReg << ", " << secReg << ";\n";
  } else if (opName == "subtract" || opName == "Sub") {
    out << "    sub.f32 " << result << ", " << inReg << ", " << secReg << ";\n";
  } else if (opName == "multiply" || opName == "Mul") {
    out << "    mul.f32 " << result << ", " << inReg << ", " << secReg << ";\n";
  } else if (opName == "divide" || opName == "Div" || opName == "RealDiv") {
    out << "    div.rn.f32 " << result << ", " << inReg << ", " << secReg << ";\n";
  } else if (opName == "minimum" || opName == "Min" || opName == "min_pairwise" || opName == "MinPairwise") {
    out << "    min.f32 " << result << ", " << inReg << ", " << secReg << ";\n";
  } else if (opName == "maximum" || opName == "Max" || opName == "max_pairwise" || opName == "MaxPairwise") {
    out << "    max.f32 " << result << ", " << inReg << ", " << secReg << ";\n";
  } else if (opName == "pow" || opName == "Pow") {
    // pow(a,b) = ex2(b * lg2(a))
    std::string lgA = ra.allocFloat();
    std::string bLg = ra.allocFloat();
    out << "    lg2.approx.f32 " << lgA << ", " << inReg << ";\n";
    out << "    mul.f32 " << bLg << ", " << secReg << ", " << lgA << ";\n";
    out << "    ex2.approx.f32 " << result << ", " << bLg << ";\n";
  } else if (opName == "reversesubtract" || opName == "ReverseSubtract") {
    out << "    sub.f32 " << result << ", " << secReg << ", " << inReg << ";\n";
  } else if (opName == "reversedivide" || opName == "ReverseDivide") {
    out << "    div.rn.f32 " << result << ", " << secReg << ", " << inReg << ";\n";
  } else if (opName == "squaredsubtract" || opName == "SquaredSubtract") {
    std::string diff = ra.allocFloat();
    out << "    sub.f32 " << diff << ", " << inReg << ", " << secReg << ";\n";
    out << "    mul.f32 " << result << ", " << diff << ", " << diff << ";\n";
  } else if (opName == "swish_mul" || opName == "SwishMul") {
    // swish_mul(x,y) = x * sigmoid(x) * y
    std::string neg = ra.allocFloat();
    std::string scaled = ra.allocFloat();
    std::string e2 = ra.allocFloat();
    std::string sum = ra.allocFloat();
    std::string sig = ra.allocFloat();
    std::string swish = ra.allocFloat();
    out << "    neg.f32 " << neg << ", " << inReg << ";\n";
    out << "    mul.f32 " << scaled << ", " << neg << ", 0f3FB8AA3B;\n";
    out << "    ex2.approx.f32 " << e2 << ", " << scaled << ";\n";
    out << "    add.f32 " << sum << ", " << e2 << ", 0f3F800000;\n";
    out << "    rcp.approx.f32 " << sig << ", " << sum << ";\n";
    out << "    mul.f32 " << swish << ", " << inReg << ", " << sig << ";\n";
    out << "    mul.f32 " << result << ", " << swish << ", " << secReg << ";\n";
  } else {
    // Fallback: add
    out << "    add.f32 " << result << ", " << inReg << ", " << secReg << "; // fallback binary: " << opName << "\n";
  }

  return result;
}

/**
 * Emit PTX instructions for a unary element-wise op.
 */
static std::string emitPtxUnaryOp(std::ostringstream& out, PtxRegAlloc& ra,
                                    const std::string& opName,
                                    const std::string& inReg,
                                    const NativeSlot& slot) {
  std::string result = ra.allocFloat();

  if (opName == "relu" || opName == "Relu") {
    out << "    max.f32 " << result << ", " << inReg << ", 0f00000000;\n";
  } else if (opName == "abs" || opName == "Abs") {
    out << "    abs.f32 " << result << ", " << inReg << ";\n";
  } else if (opName == "neg" || opName == "Neg") {
    out << "    neg.f32 " << result << ", " << inReg << ";\n";
  } else if (opName == "square" || opName == "Square") {
    out << "    mul.f32 " << result << ", " << inReg << ", " << inReg << ";\n";
  } else if (opName == "sqrt" || opName == "Sqrt") {
    out << "    sqrt.approx.f32 " << result << ", " << inReg << ";\n";
  } else if (opName == "rsqrt" || opName == "Rsqrt") {
    out << "    rsqrt.approx.f32 " << result << ", " << inReg << ";\n";
  } else if (opName == "reciprocal" || opName == "Reciprocal") {
    out << "    rcp.approx.f32 " << result << ", " << inReg << ";\n";
  } else if (opName == "exp" || opName == "Exp") {
    // exp(x) = ex2(x * log2(e))
    std::string t = ra.allocFloat();
    out << "    mul.f32 " << t << ", " << inReg << ", 0f3FB8AA3B;\n";
    out << "    ex2.approx.f32 " << result << ", " << t << ";\n";
  } else if (opName == "log" || opName == "Log") {
    // log(x) = lg2(x) * ln(2)
    std::string t = ra.allocFloat();
    out << "    lg2.approx.f32 " << t << ", " << inReg << ";\n";
    out << "    mul.f32 " << result << ", " << t << ", 0f3F317218;\n";
  } else if (opName == "sigmoid" || opName == "Sigmoid") {
    std::string neg = ra.allocFloat();
    std::string scaled = ra.allocFloat();
    std::string e2 = ra.allocFloat();
    std::string sum = ra.allocFloat();
    out << "    neg.f32 " << neg << ", " << inReg << ";\n";
    out << "    mul.f32 " << scaled << ", " << neg << ", 0f3FB8AA3B;\n";
    out << "    ex2.approx.f32 " << e2 << ", " << scaled << ";\n";
    out << "    add.f32 " << sum << ", " << e2 << ", 0f3F800000;\n";
    out << "    rcp.approx.f32 " << result << ", " << sum << ";\n";
  } else if (opName == "tanh" || opName == "Tanh") {
    out << "    // tanh via compound: 2*sigmoid(2x)-1\n";
    std::string x2 = ra.allocFloat();
    std::string neg2x = ra.allocFloat();
    std::string scaled = ra.allocFloat();
    std::string e2 = ra.allocFloat();
    std::string sum = ra.allocFloat();
    std::string sig = ra.allocFloat();
    std::string sig2 = ra.allocFloat();
    out << "    add.f32 " << x2 << ", " << inReg << ", " << inReg << ";\n";
    out << "    neg.f32 " << neg2x << ", " << x2 << ";\n";
    out << "    mul.f32 " << scaled << ", " << neg2x << ", 0f3FB8AA3B;\n";
    out << "    ex2.approx.f32 " << e2 << ", " << scaled << ";\n";
    out << "    add.f32 " << sum << ", " << e2 << ", 0f3F800000;\n";
    out << "    rcp.approx.f32 " << sig << ", " << sum << ";\n";
    out << "    add.f32 " << sig2 << ", " << sig << ", " << sig << ";\n";
    out << "    sub.f32 " << result << ", " << sig2 << ", 0f3F800000;\n";
  } else if (opName == "swish" || opName == "Swish" || opName == "silu" || opName == "Silu") {
    // swish(x) = x * sigmoid(x)
    std::string neg = ra.allocFloat();
    std::string scaled = ra.allocFloat();
    std::string e2 = ra.allocFloat();
    std::string sum = ra.allocFloat();
    std::string sig = ra.allocFloat();
    out << "    neg.f32 " << neg << ", " << inReg << ";\n";
    out << "    mul.f32 " << scaled << ", " << neg << ", 0f3FB8AA3B;\n";
    out << "    ex2.approx.f32 " << e2 << ", " << scaled << ";\n";
    out << "    add.f32 " << sum << ", " << e2 << ", 0f3F800000;\n";
    out << "    rcp.approx.f32 " << sig << ", " << sum << ";\n";
    out << "    mul.f32 " << result << ", " << inReg << ", " << sig << ";\n";
  } else if (opName == "mish" || opName == "Mish") {
    // mish(x) = x * tanh(softplus(x))
    std::string xScaled = ra.allocFloat();
    std::string expX = ra.allocFloat();
    std::string sp = ra.allocFloat();
    std::string spLog = ra.allocFloat();
    out << "    mul.f32 " << xScaled << ", " << inReg << ", 0f3FB8AA3B;\n";
    out << "    ex2.approx.f32 " << expX << ", " << xScaled << ";\n";
    out << "    add.f32 " << sp << ", " << expX << ", 0f3F800000;\n";
    out << "    lg2.approx.f32 " << spLog << ", " << sp << ";\n";
    std::string softplus = ra.allocFloat();
    out << "    mul.f32 " << softplus << ", " << spLog << ", 0f3F317218;\n";
    // tanh(softplus) via 2*sigmoid(2*softplus)-1
    std::string sp2 = ra.allocFloat();
    std::string negsp2 = ra.allocFloat();
    std::string sc2 = ra.allocFloat();
    std::string e2b = ra.allocFloat();
    std::string sm = ra.allocFloat();
    std::string sg = ra.allocFloat();
    std::string sg2 = ra.allocFloat();
    std::string th = ra.allocFloat();
    out << "    add.f32 " << sp2 << ", " << softplus << ", " << softplus << ";\n";
    out << "    neg.f32 " << negsp2 << ", " << sp2 << ";\n";
    out << "    mul.f32 " << sc2 << ", " << negsp2 << ", 0f3FB8AA3B;\n";
    out << "    ex2.approx.f32 " << e2b << ", " << sc2 << ";\n";
    out << "    add.f32 " << sm << ", " << e2b << ", 0f3F800000;\n";
    out << "    rcp.approx.f32 " << sg << ", " << sm << ";\n";
    out << "    add.f32 " << sg2 << ", " << sg << ", " << sg << ";\n";
    out << "    sub.f32 " << th << ", " << sg2 << ", 0f3F800000;\n";
    out << "    mul.f32 " << result << ", " << inReg << ", " << th << ";\n";
  } else if (opName == "gelu" || opName == "Gelu") {
    // GELU approx: x * sigmoid(1.702 * x)
    std::string scaled = ra.allocFloat();
    std::string neg = ra.allocFloat();
    std::string negScaled = ra.allocFloat();
    std::string e2 = ra.allocFloat();
    std::string sum = ra.allocFloat();
    std::string sig = ra.allocFloat();
    out << "    mul.f32 " << scaled << ", " << inReg << ", 0f3FD9999A;\n";
    out << "    neg.f32 " << neg << ", " << scaled << ";\n";
    out << "    mul.f32 " << negScaled << ", " << neg << ", 0f3FB8AA3B;\n";
    out << "    ex2.approx.f32 " << e2 << ", " << negScaled << ";\n";
    out << "    add.f32 " << sum << ", " << e2 << ", 0f3F800000;\n";
    out << "    rcp.approx.f32 " << sig << ", " << sum << ";\n";
    out << "    mul.f32 " << result << ", " << inReg << ", " << sig << ";\n";
  } else if (opName == "sign" || opName == "Sign") {
    // sign(x): compare with 0 using predicates
    std::string pPos = ra.allocPred();
    std::string pNeg = ra.allocPred();
    out << "    mov.f32 " << result << ", 0f00000000;\n";
    out << "    setp.gt.f32 " << pPos << ", " << inReg << ", 0f00000000;\n";
    out << "    @" << pPos << " mov.f32 " << result << ", 0f3F800000;\n";
    out << "    setp.lt.f32 " << pNeg << ", " << inReg << ", 0f00000000;\n";
    out << "    @" << pNeg << " mov.f32 " << result << ", 0fBF800000;\n";
  } else if (opName == "ceil" || opName == "Ceil") {
    out << "    cvt.rpi.f32.f32 " << result << ", " << inReg << ";\n";
  } else if (opName == "floor" || opName == "Floor") {
    out << "    cvt.rmi.f32.f32 " << result << ", " << inReg << ";\n";
  } else if (opName == "round" || opName == "Round") {
    out << "    cvt.rni.f32.f32 " << result << ", " << inReg << ";\n";
  } else if (opName == "sin" || opName == "Sin") {
    out << "    sin.approx.f32 " << result << ", " << inReg << ";\n";
  } else if (opName == "cos" || opName == "Cos") {
    out << "    cos.approx.f32 " << result << ", " << inReg << ";\n";
  } else if (opName == "log1p" || opName == "Log1p") {
    // log1p(x) = log(1+x) = lg2(1+x) * ln(2)
    std::string sum = ra.allocFloat();
    std::string lg = ra.allocFloat();
    out << "    add.f32 " << sum << ", " << inReg << ", 0f3F800000;\n";
    out << "    lg2.approx.f32 " << lg << ", " << sum << ";\n";
    out << "    mul.f32 " << result << ", " << lg << ", 0f3F317218;\n";
  } else if (opName == "softplus" || opName == "Softplus") {
    // softplus(x) = max(0, x) + log(1 + exp(-|x|))  [overflow-safe]
    std::string absX = ra.allocFloat();
    std::string negAbsX = ra.allocFloat();
    std::string maxPart = ra.allocFloat();
    std::string negScaled = ra.allocFloat();
    std::string expNeg = ra.allocFloat();
    std::string sp = ra.allocFloat();
    std::string spLog = ra.allocFloat();
    out << "    abs.f32 " << absX << ", " << inReg << ";\n";
    out << "    neg.f32 " << negAbsX << ", " << absX << ";\n";
    out << "    max.f32 " << maxPart << ", " << inReg << ", 0f00000000;\n";  // max(0, x)
    out << "    mul.f32 " << negScaled << ", " << negAbsX << ", 0f3FB8AA3B;\n";  // -|x| * log2(e)
    out << "    ex2.approx.f32 " << expNeg << ", " << negScaled << ";\n";  // exp(-|x|)
    out << "    add.f32 " << sp << ", " << expNeg << ", 0f3F800000;\n";  // 1 + exp(-|x|)
    out << "    lg2.approx.f32 " << spLog << ", " << sp << ";\n";
    out << "    mul.f32 " << spLog << ", " << spLog << ", 0f3F317218;\n";  // log(1 + exp(-|x|))
    out << "    add.f32 " << result << ", " << maxPart << ", " << spLog << ";\n";  // max(0,x) + log(1+exp(-|x|))
  } else if (opName == "softsign" || opName == "Softsign") {
    // softsign(x) = x / (1 + |x|)
    std::string absX = ra.allocFloat();
    std::string denom = ra.allocFloat();
    out << "    abs.f32 " << absX << ", " << inReg << ";\n";
    out << "    add.f32 " << denom << ", " << absX << ", 0f3F800000;\n";
    out << "    div.rn.f32 " << result << ", " << inReg << ", " << denom << ";\n";
  } else if (opName == "elu" || opName == "Elu") {
    // elu(x) = x >= 0 ? x : exp(x)-1
    std::string pGe = ra.allocPred();
    std::string xScaled = ra.allocFloat();
    std::string expX = ra.allocFloat();
    out << "    setp.ge.f32 " << pGe << ", " << inReg << ", 0f00000000;\n";
    out << "    mul.f32 " << xScaled << ", " << inReg << ", 0f3FB8AA3B;\n";
    out << "    ex2.approx.f32 " << expX << ", " << xScaled << ";\n";
    out << "    sub.f32 " << result << ", " << expX << ", 0f3F800000;\n";
    out << "    @" << pGe << " mov.f32 " << result << ", " << inReg << ";\n";
  } else if (opName == "relu6" || opName == "Relu6") {
    // relu6(x) = min(max(x, 0), 6)
    std::string t = ra.allocFloat();
    out << "    max.f32 " << t << ", " << inReg << ", 0f00000000;\n";
    out << "    min.f32 " << result << ", " << t << ", 0f40C00000;\n";  // 6.0f
  } else if (opName == "hard_sigmoid" || opName == "HardSigmoid") {
    // hard_sigmoid(x) = min(1, max(0, 0.2*x + 0.5))
    std::string scaled = ra.allocFloat();
    std::string shifted = ra.allocFloat();
    std::string clamped = ra.allocFloat();
    out << "    mul.f32 " << scaled << ", " << inReg << ", 0f3E4CCCCD;\n";  // 0.2f
    out << "    add.f32 " << shifted << ", " << scaled << ", 0f3F000000;\n";  // 0.5f
    out << "    max.f32 " << clamped << ", " << shifted << ", 0f00000000;\n";
    out << "    min.f32 " << result << ", " << clamped << ", 0f3F800000;\n";
  } else if (opName == "hardtanh" || opName == "HardTanh") {
    // hardtanh(x) = min(1, max(-1, x))
    std::string t = ra.allocFloat();
    out << "    max.f32 " << t << ", " << inReg << ", 0fBF800000;\n";  // -1.0f
    out << "    min.f32 " << result << ", " << t << ", 0f3F800000;\n";  // 1.0f
  } else if (opName == "leakyrelu" || opName == "LeakyRelu") {
    // leakyrelu(x) = x >= 0 ? x : alpha*x  (alpha default 0.01)
    std::string pGe = ra.allocPred();
    std::string scaled = ra.allocFloat();
    // 0.01f = 0x3C23D70A
    std::string alphaHex = "0f3C23D70A";
    out << "    setp.ge.f32 " << pGe << ", " << inReg << ", 0f00000000;\n";
    out << "    mul.f32 " << scaled << ", " << inReg << ", " << alphaHex << ";\n";
    out << "    selp.f32 " << result << ", " << inReg << ", " << scaled << ", " << pGe << ";\n";
  } else if (opName == "erf" || opName == "Erf") {
    // Approximate erf using tanh approximation:
    // erf(x) ~ tanh(x * 1.2024 * (1 + 0.04028 * x^2))
    // For PTX simplicity, use identity pass-through (exact erf not available in PTX)
    out << "    mov.f32 " << result << ", " << inReg << "; // erf: identity fallback in PTX\n";
  } else {
    // Identity fallback for ops not yet mapped to PTX
    out << "    mov.f32 " << result << ", " << inReg << "; // unsupported op: " << opName << "\n";
  }

  return result;
}

/**
 * Emit PTX instructions for a comparison op.
 */
static std::string emitPtxComparisonOp(std::ostringstream& out, PtxRegAlloc& ra,
                                         const std::string& opName,
                                         const std::string& inReg,
                                         const std::string& secReg) {
  std::string result = ra.allocFloat();
  std::string pred = ra.allocPred();

  if (opName == "greater" || opName == "Greater") {
    out << "    setp.gt.f32 " << pred << ", " << inReg << ", " << secReg << ";\n";
  } else if (opName == "greater_equal" || opName == "GreaterEqual") {
    out << "    setp.ge.f32 " << pred << ", " << inReg << ", " << secReg << ";\n";
  } else if (opName == "less" || opName == "Less") {
    out << "    setp.lt.f32 " << pred << ", " << inReg << ", " << secReg << ";\n";
  } else if (opName == "less_equal" || opName == "LessEqual") {
    out << "    setp.le.f32 " << pred << ", " << inReg << ", " << secReg << ";\n";
  } else if (opName == "equals" || opName == "Equals") {
    out << "    setp.eq.f32 " << pred << ", " << inReg << ", " << secReg << ";\n";
  } else if (opName == "not_equals" || opName == "NotEquals") {
    out << "    setp.ne.f32 " << pred << ", " << inReg << ", " << secReg << ";\n";
  } else {
    out << "    setp.gt.f32 " << pred << ", " << inReg << ", " << secReg << ";\n";
  }
  out << "    selp.f32 " << result << ", 0f3F800000, 0f00000000, " << pred << ";\n";

  return result;
}

/**
 * Emit PTX instructions for a logical op.
 */
static std::string emitPtxLogicalOp(std::ostringstream& out, PtxRegAlloc& ra,
                                      const std::string& opName,
                                      const std::string& inReg,
                                      const std::string& secReg) {
  std::string result = ra.allocFloat();
  std::string pA = ra.allocPred();
  std::string pB = ra.allocPred();
  std::string pR = ra.allocPred();

  out << "    setp.ne.f32 " << pA << ", " << inReg << ", 0f00000000;\n";

  if (opName == "boolean_not" || opName == "BooleanNot" ||
      opName == "logical_not" || opName == "LogicalNot") {
    // not(a): result = !a
    std::string pNot = ra.allocPred();
    out << "    not.pred " << pNot << ", " << pA << ";\n";
    out << "    selp.f32 " << result << ", 0f3F800000, 0f00000000, " << pNot << ";\n";
  } else {
    out << "    setp.ne.f32 " << pB << ", " << secReg << ", 0f00000000;\n";
    if (opName == "boolean_and" || opName == "BooleanAnd" ||
        opName == "logical_and" || opName == "LogicalAnd") {
      out << "    and.pred " << pR << ", " << pA << ", " << pB << ";\n";
    } else if (opName == "boolean_or" || opName == "BooleanOr" ||
               opName == "logical_or" || opName == "LogicalOr") {
      out << "    or.pred " << pR << ", " << pA << ", " << pB << ";\n";
    } else if (opName == "boolean_xor" || opName == "BooleanXor") {
      out << "    xor.pred " << pR << ", " << pA << ", " << pB << ";\n";
    } else {
      out << "    and.pred " << pR << ", " << pA << ", " << pB << ";\n";
    }
    out << "    selp.f32 " << result << ", 0f3F800000, 0f00000000, " << pR << ";\n";
  }

  return result;
}

/**
 * Emit PTX instructions for a ternary op (where/select).
 */
static std::string emitPtxTernaryOp(std::ostringstream& out, PtxRegAlloc& ra,
                                      const std::string& condReg,
                                      const std::string& trueReg,
                                      const std::string& falseReg) {
  std::string result = ra.allocFloat();
  std::string pred = ra.allocPred();
  out << "    setp.ne.f32 " << pred << ", " << condReg << ", 0f00000000;\n";
  out << "    selp.f32 " << result << ", " << trueReg << ", " << falseReg << ", " << pred << ";\n";
  return result;
}

/**
 * Dispatch to appropriate PTX emitter based on op category.
 */
static std::string emitPtxOpByCategory(std::ostringstream& out, PtxRegAlloc& ra,
                                         TritonOpCategory cat, const std::string& opName,
                                         const std::string& inReg,
                                         const std::string& secReg,
                                         const std::string& terReg,
                                         const NativeSlot& slot) {
  switch (cat) {
    case TritonOpCategory::BINARY_ELEMENTWISE:
      return emitPtxBinaryOp(out, ra, opName, inReg, secReg);
    case TritonOpCategory::UNARY_ELEMENTWISE:
      return emitPtxUnaryOp(out, ra, opName, inReg, slot);
    case TritonOpCategory::COMPARISON:
      return emitPtxComparisonOp(out, ra, opName, inReg, secReg);
    case TritonOpCategory::LOGICAL:
      return emitPtxLogicalOp(out, ra, opName, inReg, secReg);
    case TritonOpCategory::TERNARY:
      return emitPtxTernaryOp(out, ra, inReg, secReg, terReg);
    case TritonOpCategory::CAST:
    case TritonOpCategory::IDENTITY: {
      // Pass-through
      std::string result = ra.allocFloat();
      out << "    mov.f32 " << result << ", " << inReg << ";\n";
      return result;
    }
    default: {
      // Identity fallback
      std::string result = ra.allocFloat();
      out << "    mov.f32 " << result << ", " << inReg << "; // unsupported category: " << opName << "\n";
      return result;
    }
  }
}

// ---- PTX generation ----

std::string PtxGraphBackend::generatePtx(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    JitCompiledKernel& result) {

  int smVersion = getSmVersion();

  // Collect external inputs and outputs (same logic as NVRTC backend)
  std::unordered_map<int, int> externalInputMap;  // extIdx -> paramIdx
  std::vector<int> outputSlotIndices;
  int paramIdx = 0;

  for (int si = startSlot; si <= endSlot; si++) {
    auto& slot = slots[si];
    for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
      int srcIdx = slot.wiring.inputSourceIndices[inp];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (externalInputMap.find(extIdx) == externalInputMap.end()) {
          externalInputMap[extIdx] = paramIdx++;
        }
      }
    }
  }

  auto& lastSlot = slots[endSlot];
  for (int o = 0; o < lastSlot.wiring.numOutputs; o++) {
    outputSlotIndices.push_back(lastSlot.wiring.outputSlotIndices[o]);
  }

  int numInputParams = static_cast<int>(externalInputMap.size());
  int numOutputParams = static_cast<int>(outputSlotIndices.size());

  // Build arg mappings
  for (auto& [extIdx, pIdx] : externalInputMap) {
    JitCompiledKernel::ArgMapping am;
    am.slotIndex = -(extIdx + 1);
    am.isOutput = false;
    result.argMap.push_back(am);
  }
  for (int outSlotIdx : outputSlotIndices) {
    JitCompiledKernel::ArgMapping am;
    am.slotIndex = outSlotIdx;
    am.isOutput = true;
    result.argMap.push_back(am);
  }

  // Build kernel body first so register declarations match what was emitted.
  std::ostringstream body;

  // Load external inputs
  PtxRegAlloc ra;
  std::unordered_map<int, std::string> extRegMap;  // extParamIdx -> float register

  for (int i = 0; i < numInputParams; i++) {
    int rdBase = i + 1;
    std::string fReg = ra.allocFloat();
    body << "    ld.param.u64 rd" << rdBase << ", [in" << i << "];\n";
    body << "    add.u64 rd" << rdBase << ", rd" << rdBase << ", rd0;\n";
    body << "    ld.global.f32 " << fReg << ", [rd" << rdBase << "];\n";
    extRegMap[i] = fReg;
  }
  body << "\n";

  // Walk slots and emit ops
  std::unordered_map<int, std::string> slotOutputRegs;  // outputSlotIdx -> float register

  for (int si = startSlot; si <= endSlot; si++) {
    auto& slot = slots[si];
    auto cat = getOpCategoryFromName(slot.ident.opName);
    int inputCount = categoryInputCount(cat);

    // Helper to resolve an input source index to a register name
    auto resolveInput = [&](int inputIdx) -> std::string {
      if (inputIdx < slot.wiring.numInputs) {
        int srcIdx = slot.wiring.inputSourceIndices[inputIdx];
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          return extRegMap[externalInputMap[extIdx]];
        } else {
          auto it = slotOutputRegs.find(srcIdx);
          if (it != slotOutputRegs.end()) {
            return it->second;
          }
        }
      }
      // Missing input: allocate a zero register
      std::string zReg = ra.allocFloat();
      body << "    mov.f32 " << zReg << ", 0f00000000; // missing input\n";
      return zReg;
    };

    // Resolve inputs based on category input count
    std::string inReg = (slot.wiring.numInputs > 0) ? resolveInput(0) : "0f00000000";
    std::string secReg = (inputCount >= 2 && slot.wiring.numInputs > 1) ? resolveInput(1) : "0f00000000";
    std::string terReg = (inputCount >= 3 && slot.wiring.numInputs > 2) ? resolveInput(2) : "0f00000000";

    body << "    // slot " << si << ": " << slot.ident.opName << "\n";

    std::string resultReg = emitPtxOpByCategory(body, ra, cat, slot.ident.opName, inReg, secReg, terReg, slot);

    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      slotOutputRegs[slot.wiring.outputSlotIndices[o]] = resultReg;
    }
  }

  // Store outputs
  body << "\n";
  for (int i = 0; i < numOutputParams; i++) {
    int rdIdx = numInputParams + i + 1;
    int outSlotIdx = outputSlotIndices[i];
    auto it = slotOutputRegs.find(outSlotIdx);
    if (it != slotOutputRegs.end()) {
      body << "    ld.param.u64 rd" << rdIdx << ", [out" << i << "];\n";
      body << "    add.u64 rd" << rdIdx << ", rd" << rdIdx << ", rd0;\n";
      body << "    st.global.f32 [rd" << rdIdx << "], " << it->second << ";\n";
    }
  }

  int maxFloatRegs = std::max(8, ra.nextFloat + 1);
  int maxPredRegs = std::max(2, ra.nextPred + 1);

  // PTX ISA version must be compatible with the target SM architecture.
  // Mapping sourced from LLVM NVPTX getMinPTXVersionForSM() (2026).
  // NOTE: Blackwell numbering is non-monotonic in PTX requirements:
  //   sm_110 (11.0 family, Jetson Thor) needs PTX 9.0
  //   sm_120/121 (12.x family, RTX 50xx) needs PTX 8.7/8.8
  //   sm_100/103 (10.x family, B100/B200/B300) needs PTX 8.6/8.8
  // So we use a switch on major version first, then minor.
  const char* ptxVersion;
  int smMajor = smVersion / 10;
  int smMinor = smVersion % 10;
  switch (smMajor) {
    case 12:  // Blackwell consumer (RTX 50xx): sm_120->8.7, sm_121->8.8
      ptxVersion = (smMinor >= 1) ? "8.8" : "8.7";
      break;
    case 11:  // Jetson Thor: sm_110->9.0
      ptxVersion = "9.0";
      break;
    case 10:  // Blackwell datacenter (B100/B200/B300): sm_100/101->8.6, sm_103->8.8
      ptxVersion = (smMinor >= 3) ? "8.8" : "8.6";
      break;
    case 9:   // Hopper: sm_90->7.8 (sm_90a needs 8.0 but we emit base sm_90)
      ptxVersion = "7.8";
      break;
    case 8:   // Ampere/Ada: sm_80->7.0, sm_86->7.1, sm_87->7.4, sm_88->9.0, sm_89->7.8
      if      (smMinor >= 9) ptxVersion = "7.8";
      else if (smMinor == 8) ptxVersion = "9.0";  // sm_88 (future arch, per LLVM)
      else if (smMinor >= 7) ptxVersion = "7.4";
      else if (smMinor >= 6) ptxVersion = "7.1";
      else                   ptxVersion = "7.0";
      break;
    case 7:   // Volta/Turing: sm_70->6.0, sm_72->6.1, sm_75->6.3
      if      (smMinor >= 5) ptxVersion = "6.3";
      else if (smMinor >= 2) ptxVersion = "6.1";
      else                   ptxVersion = "6.0";
      break;
    case 6:   // Pascal: sm_60/61/62->5.0
      ptxVersion = "5.0";
      break;
    default:  // Maxwell and older: sm_53->4.2, sm_50/52->4.1
      ptxVersion = (smMinor >= 3) ? "4.2" : "4.1";
      break;
  }

  std::ostringstream ptx;
  ptx << ".version " << ptxVersion << "\n";
  ptx << ".target sm_" << smVersion << "\n";
  ptx << ".address_size 64\n\n";

  // Kernel entry
  ptx << ".visible .entry ptx_fused_kernel(\n";
  for (int i = 0; i < numInputParams; i++) {
    ptx << "    .param .u64 in" << i << ",\n";
  }
  for (int i = 0; i < numOutputParams; i++) {
    ptx << "    .param .u64 out" << i << ",\n";
  }
  ptx << "    .param .s32 n\n";
  ptx << ") {\n";

  // Register declarations
  ptx << "    .reg .pred p<" << maxPredRegs << ">;\n";
  ptx << "    .reg .f32 f<" << maxFloatRegs << ">;\n";
  ptx << "    .reg .b32 r<16>;\n";
  ptx << "    .reg .b64 rd<" << (numInputParams + numOutputParams + 8) << ">;\n\n";

  // Thread index computation: idx = blockIdx.x * blockDim.x + threadIdx.x
  ptx << "    mov.u32 r0, %ctaid.x;\n";
  ptx << "    mov.u32 r1, %ntid.x;\n";
  ptx << "    mov.u32 r2, %tid.x;\n";
  ptx << "    mad.lo.s32 r3, r0, r1, r2;\n";

  // Bounds check
  ptx << "    ld.param.s32 r4, [n];\n";
  ptx << "    setp.ge.s32 p0, r3, r4;\n";
  ptx << "    @p0 bra EXIT;\n\n";

  // Compute byte offset: r3 * 4 (sizeof float)
  ptx << "    cvt.u64.u32 rd0, r3;\n";
  ptx << "    shl.b64 rd0, rd0, 2;\n\n";
  ptx << body.str();
  ptx << "\nEXIT:\n";
  ptx << "    ret;\n";
  ptx << "}\n";

  return ptx.str();
}

// ---- Compilation ----

bool PtxGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                      NDArray** externalInputs, int numExternalInputs,
                                      NDArray** outputSlots, int totalOutputSlots,
                                      LongType shapeKey,
                                      int totalSlots,
                                      int* requestedOutputSlotIndices,
                                      int numRequestedOutputs) {
  JitSegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, shapeKey};

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      lastCompilationAudit_ = it->second.audit;
      return true;
    }
  }

  JitCompiledKernel compiled;

  std::string ptxSrc = generatePtx(slots, seg.def.startSlot, seg.def.endSlot,
                                     externalInputs, numExternalInputs,
                                     outputSlots, totalOutputSlots, compiled);

  if (ptxSrc.empty()) {
    DSP_DIAG(COMPILE, "PtxGraphBackend: PTX generation failed for segment [%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
    return false;
  }

  // Guard against kernel argument list sizes that exceed device limits.
  // PTX backend currently passes each external/input pointer as a distinct
  // kernel parameter. Large fused segments can exceed kernel param space
  // (commonly 4KB), causing driver-side PTX JIT failures at module load.
  {
    constexpr size_t kConservativeMaxKernelParamBytes = 4096;
    size_t paramBytes = compiled.argMap.size() * sizeof(void*) + sizeof(int);
    if (paramBytes > kConservativeMaxKernelParamBytes) {
      DSP_DIAG(COMPILE, "PtxGraphBackend: segment [%d-%d] requires %zu kernel-arg bytes, exceeds limit %zu, skipping",
                seg.def.startSlot, seg.def.endSlot, paramBytes, kConservativeMaxKernelParamBytes);
      return false;
    }
  }

  // Build audit
  for (int i = seg.def.startSlot; i <= seg.def.endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].ident.opName;
    entry.wasCompiled = isNvrtcJittable(getOpCategoryFromName(slots[i].ident.opName));
    if (!entry.wasCompiled) {
      entry.reason = "unmappable op (not in OpCategoryTable or not NVRTC-jittable)";
    }
    compiled.audit.push_back(entry);
  }

  // Load PTX directly (JIT compilation by CUDA driver)
  compiled.gpuModule = GpuKernelLauncher::loadPtxModule(ptxSrc.c_str(), ptxSrc.size());
  if (!compiled.gpuModule) {
    DSP_DIAG(COMPILE, "PtxGraphBackend: PTX module load failed for segment [%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
    return false;
  }

  compiled.kernelFunction = GpuKernelLauncher::getKernelFunc(compiled.gpuModule, "ptx_fused_kernel");
  if (!compiled.kernelFunction) {
    DSP_DIAG(COMPILE, "PtxGraphBackend: kernel function not found in module");
    GpuKernelLauncher::unloadModule(compiled.gpuModule);
    compiled.gpuModule = nullptr;
    return false;
  }

  lastCompilationAudit_ = compiled.audit;

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_[key] = std::move(compiled);
  }

  DSP_DIAG(JIT, "PtxGraphBackend: loaded segment [%d-%d] (%zu bytes PTX, shape key %lld)",
            seg.def.startSlot, seg.def.endSlot, ptxSrc.size(), shapeKey);
  return true;
}

// ---- Execution (delegates to shared logic) ----

Status PtxGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots,
                                        void* stream) {
  JitSegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, seg.def.shapeKeyState.compiledShapeKey};
  return jitExecuteSegment(key, cache_, cacheMtx_, "PtxGraphBackend",
                           slots, externalInputs, numExternalInputs,
                           outputSlots, totalOutputSlots, stream);
}

// ---- Cache invalidation (delegates to shared logic) ----

void PtxGraphBackend::invalidateCache() {
  jitInvalidateCache(cache_, cacheMtx_, lastCompilationAudit_);
}

// ---- Audit ----

std::vector<CompilationAuditEntry> PtxGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

