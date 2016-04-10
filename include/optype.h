#pragma once

//TODO convert this into an enum class
// might break JNI though...

typedef int OpType;

namespace op_type
{
     constexpr OpType Variance = 0;
     constexpr OpType StandardDeviation = 1;
}

