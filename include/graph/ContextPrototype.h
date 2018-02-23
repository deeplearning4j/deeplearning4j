//
//  @author raver119@gmail.com
//

#ifndef ND4J_CONTEXT_PROTOTYPE_H
#define ND4J_CONTEXT_PROTOTYPE_H

#include <vector>


namespace nd4j {
    namespace graph {
        template <typename T>
        class ContextPrototype {
        protected:
            // int ids of the input nodes
            std::vector<std::pair<int, int>> _inputs;
            int _nodeId;
            std::vector<T> _tArgs;
            std::vector<int> _iArgs;            
			
			bool _isInplace;

            // opNum for legacy XYZ ops
            int _opNum = -1;

        public:
            ContextPrototype(int nodeId = 1, bool inPlace = false);
            ~ContextPrototype() = default;

            int getNodeId();
            int nodeId();

            // this method returns true, if inputs are defined
            bool hasVariablesFilled();

            bool isInplace();
            void markInplace(bool reallyInplace);

            void pickInput(int input);
            void pickInput(int input, int index);
            void pickInput(std::pair<int, int>& p);
            void fillInputs(std::initializer_list<int> inputs);
            void fillInputs(std::vector<int>& inputs);
            std::vector<std::pair<int, int>>* inputs();

            std::vector<T>* getTArguments();
            std::vector<int>* getIArguments();

            int numT();
            int numI();

            std::pair<int, int>* input(int idx);

            int opNum();
            void setOpNum(int opNum);

            /**
             * This method returns number of inputs available in this block
             * @return
             */
            unsigned long width();

            // just a clone
            ContextPrototype<T>* clone();

            template <typename N>
            ContextPrototype<N>* asT();
        };
    }
}

#endif //ND4J_CONTEXT_PROTOTYPE_H