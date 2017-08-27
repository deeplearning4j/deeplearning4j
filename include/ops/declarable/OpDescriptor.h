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
            int _opNum = 0;
            std::string _opName;

            int _numInputs;
            int _numOutputs;

        public:
            // default constructor
            OpDescriptor(int numInputs, int numOutputs, std::string opName) {
                _numInputs = numInputs;
                _numOutputs = numOutputs;
                _opName = opName;
            }

            OpDescriptor(int numInputs, int numOutputs, const char *opName) {
                _numInputs = numInputs;
                _numOutputs = numOutputs;

                std::string tmp(opName);
                _opName = tmp;
            }

            // default destructor
            ~OpDescriptor() {
                //
            }

            int getNumberOfInputs() {
                return _numInputs;
            }

            int getNumberOfOutputs() {
                return _numOutputs;
            }

            std::string *getOpName() {
                return &_opName;
            }

            int getOpNum() {
                return _opNum;
            }
        };
    }
}

#endif //LIBND4J_OPDESCRIPTOR_H
