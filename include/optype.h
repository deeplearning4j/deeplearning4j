#pragma once

//TODO convert this into an enum class
// might break JNI though...

typedef int OpType;

#if defined(_MSC_VER) && (_MSC_VER < 1900)
#define constexpr const
#endif

namespace op_type
{
     constexpr OpType Variance = 0;
     constexpr OpType StandardDeviation = 1;
}

