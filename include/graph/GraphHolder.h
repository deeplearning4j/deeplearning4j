//
//  @author raver119@gmail.com
//

#include <helpers/logger.h>
#include <pointercast.h>
#include <map>
#include <graph/Graph.h>

namespace nd4j {
    namespace graph {
        class GraphHolder {
        private:
            static GraphHolder *_INSTANCE;
            std::map<Nd4jIndex, Graph<float>*> _graphF;
            std::map<Nd4jIndex, Graph<double>*> _graphD;
            std::map<Nd4jIndex, Graph<float16>*> _graphH;

            GraphHolder() = default;
            ~GraphHolder() = default;
        public:
            static GraphHolder* getInstance();

            template <typename T>
            void registerGraph(Nd4jIndex graphId, Graph<T>* graph);
            
            template <typename T>
            Graph<T>* cloneGraph(Nd4jIndex graphId);

            template <typename T>
            Graph<T>* pullGraph(Nd4jIndex graphId);

            template <typename T>
            void forgetGraph(Nd4jIndex graphId);

            template <typename T>
            void dropGraph(Nd4jIndex graphId);


            void dropGraphAny(Nd4jIndex graphId);

            template <typename T>
            bool hasGraph(Nd4jIndex graphId);
        };
    }
}