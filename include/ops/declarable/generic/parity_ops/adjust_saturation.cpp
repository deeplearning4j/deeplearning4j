//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_adjust_saturation)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/adjust_saturation.h>

namespace nd4j {
namespace ops {
    CONFIGURABLE_OP_IMPL(adjust_saturation, 1, 1, true, -2, -2) {
        auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);

        REQUIRE_TRUE(input->rankOf() == 3 || input->rankOf() == 4, 0, "AdjustSaturation: op expects either 3D or 4D input, but got %i instead", input->rankOf());

        T delta = 0;
        if (block.numT() > 0)
            delta = T_ARG(0);
        else if (block.width() > 1) {
            auto _d = INPUT_VARIABLE(1);
            if (!_d->isScalar()) {
                auto str = ShapeUtils<T>::shapeAsString(_d);
                REQUIRE_TRUE(_d->isScalar(), 0, "AdjustSaturation: delta should be scalar NDArray, but got %s instead", str.c_str());
            }
        }

        bool isNHWC = false;
        if (block.numI() > 0)
            isNHWC = INT_ARG(0) == 1;

        int numChannels = isNHWC ? input->sizeAt(-1) : input->sizeAt(-3);

        REQUIRE_TRUE(numChannels == 3, 0, "AdjustSaturation: this operation expects image with 3 channels (R, G, B), but got % instead", numChannels);

        helpers::_adjust_saturation(input, output, delta, isNHWC);

        return ND4J_STATUS_OK;
    }
}
}

#endif