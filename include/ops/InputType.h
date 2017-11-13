//
// @author raver119@gmail.com
//

#ifndef ND4J_INPUTTYPE_H
#define ND4J_INPUTTYPE_H

namespace nd4j {
    namespace ops {
        enum InputType {
            InputType_BOOLEAN = 0,
            InputType_NUMERIC = 1,
            InputType_STRINGULAR = 2,
            InputType_NUMERIC_SET = 3,
            InputType_STRINGULAR_SET = 4,
        };
    }
}

#endif