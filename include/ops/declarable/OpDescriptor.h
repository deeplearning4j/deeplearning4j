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

            bool _divergent;
            bool _allowsInplace;

            int _tArgs = 0;
            int _iArgs = 0;

        public:
            // default constructor
            OpDescriptor(int numInputs, int numOutputs, std::string opName, bool allowsInplace) : OpDescriptor(numInputs, numOutputs, opName.c_str(), allowsInplace) {
                //
            }

            // default constructor
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace) {
                _numInputs = numInputs;
                _numOutputs = numOutputs;

                std::string tmp(opName);
                _opName = tmp;
                _allowsInplace = allowsInplace;
                _divergent = false;
            }

            // constructor for configurable op
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs) : OpDescriptor(numInputs, numOutputs, opName, allowsInplace) {
                _tArgs = tArgs;
                _iArgs = iArgs;
            }

            // constructor for non-configurable divergent op
            OpDescriptor(int numInputs, int numOutputs, std::string opName, bool allowsInplace, bool divergent) : OpDescriptor(numInputs, numOutputs, opName.c_str(), allowsInplace, divergent) {

            }

            // constructor for non-configurable divergent op
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent) : OpDescriptor(numInputs, numOutputs, opName, allowsInplace) {
                _divergent = divergent;
            }

            // constructor for configurable divergent op
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent, int tArgs, int iArgs) : OpDescriptor(numInputs, numOutputs, opName, allowsInplace, tArgs, iArgs) {
                _divergent = divergent;
            }

            // default destructor
            ~OpDescriptor() {
                //
            }

            int getNumberOfTArgs() {
                return _tArgs;
            }

            int getNumberOfIArgs() {
                return _iArgs;
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

            bool isDivergent() {
                return _divergent;
            }

            bool allowsInplace() {
                return _allowsInplace;
            }

            int getOpNum() {
                return _opNum;
            }
        };
    }
}

#endif //LIBND4J_OPDESCRIPTOR_H
