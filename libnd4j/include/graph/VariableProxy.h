//
//  @author raver119@gmail.com
//

#include <graph/VariableSpace.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        class VariableProxy: public VariableSpace<T> {
        protected:
            VariableSpace<T>* _backed = nullptr;
            VariableSpace<T>* _current = nullptr;
        public:
            explicit VariableProxy(VariableSpace<T>* reference);
            ~VariableProxy();

            virtual VariableSpace<T>& operator=(const VariableSpace<T>& other);

            virtual int numberOfPlaceholders();
            virtual std::vector<Variable<T>*>* getPlaceholders();
            virtual nd4j::random::RandomBuffer* getRNG();
            virtual void setRNG(nd4j::random::RandomBuffer* rng);

            virtual nd4j::memory::Workspace *workspace();

            virtual bool hasExternalVariable(int it);
            virtual bool hasExternalVariable(std::pair<int,int>& pair);
            virtual bool hasExternalVariable(std::string *symbol);

            virtual bool hasVariable(int id);
            virtual bool hasVariable(int id, int idx);
            virtual bool hasVariable(std::pair<int,int>& pair);
            virtual bool hasVariable(std::string *symbol);

            virtual nd4j::graph::Variable<T> *getVariable(int id);
            virtual nd4j::graph::Variable<T> *getVariable(int id, int idx);
            virtual nd4j::graph::Variable<T> *getVariable(std::pair<int,int>& pair);
            virtual nd4j::graph::Variable<T> *getVariable(std::string *symbol);

            virtual void putVariable(std::pair<int,int>& pair, NDArray<T> *array);
            virtual void putVariable(std::pair<int,int>& pair, Variable<T> *variable);
            virtual void putVariable(int id, Variable<T> *variable);
            virtual void putVariable(int id, NDArray<T> *array);
            virtual void putVariable(int id, int idx, NDArray<T> *array);
            virtual void putVariable(int id, int idx, Variable<T> *array);

            virtual void dropVariable(std::pair<int,int> &pair);
            virtual void dropVariable(int id, int idx);

            virtual void putOutputVariable(Variable<T> *variable);

            virtual void trackList(nd4j::NDArrayList<T>* list);

            // memory-related statistics
            virtual Nd4jLong externalMemory();
            virtual Nd4jLong internalMemory();
            virtual Nd4jLong totalMemory();

            virtual int externalEntries();
            virtual int internalEntries();
            virtual int totalEntries();

            virtual nd4j::graph::VariableSpace<T>* clone();

            virtual nd4j::graph::Stash<T>* getStash();
            virtual void setFlowPath(FlowPath* timers);
            virtual FlowPath* flowPath();
        };
    }
}