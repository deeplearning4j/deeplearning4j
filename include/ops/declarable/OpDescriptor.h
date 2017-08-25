//
// @author raver119@gmail.com
//

#ifndef LIBND4J_OPDESCRIPTOR_H
#define LIBND4J_OPDESCRIPTOR_H

#include <string>

namespace nd4j {
    namespace ops {

        class OpDescriptor {
        protected:


        public:
            int getNumberOfInputs();
            int getNumberOfOutputs();

            std::string *getName();
        };
    }
}

#endif //LIBND4J_OPDESCRIPTOR_H
