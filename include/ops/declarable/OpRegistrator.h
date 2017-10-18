//
// Created by raver119 on 07.10.2017.
//

#ifndef LIBND4J_OPREGISTRATOR_H
#define LIBND4J_OPREGISTRATOR_H

#include <pointercast.h>
#include <vector>
#include <map>
#include <mutex>
#include <ops/declarable/DeclarableOp.h>

namespace nd4j {
    namespace ops {

        class ND4J_EXPORT OpRegistrator {
        private:
            static OpRegistrator* _INSTANCE;
            OpRegistrator() {};
            ~OpRegistrator();

            std::map<Nd4jIndex, std::string> _msvc;

            std::map<Nd4jIndex, nd4j::ops::DeclarableOp<float> *> _declarablesLF;
            std::map<std::string, nd4j::ops::DeclarableOp<float> *> _declarablesF;

            std::map<Nd4jIndex, nd4j::ops::DeclarableOp<double> *> _declarablesLD;
            std::map<std::string, nd4j::ops::DeclarableOp<double> *> _declarablesD;

            std::map<Nd4jIndex, nd4j::ops::DeclarableOp<float16> *> _declarablesLH;
            std::map<std::string, nd4j::ops::DeclarableOp<float16> *> _declarablesH;

            std::mutex _locker;
            std::string _opsList;
            bool isInit = false;
        public:

            static OpRegistrator* getInstance();

            void updateMSVC(Nd4jIndex newHash, std::string& oldName);

            template <typename T>
            std::string local_to_string(T value);
            const char * getAllCustomOperations();

            /**
            * This method registers operation
            *
            * @param op
            */
            bool registerOperationFloat(nd4j::ops::DeclarableOp<float>* op);
            bool registerOperationFloat(const char* name, nd4j::ops::DeclarableOp<float>* op);

            bool registerOperationDouble(const char* name, nd4j::ops::DeclarableOp<double>* op);
            bool registerOperationHalf(const char* name, nd4j::ops::DeclarableOp<float16>* op);
            bool registerOperationHalf(nd4j::ops::DeclarableOp<float16> *op);
            bool registerOperationDouble(nd4j::ops::DeclarableOp<double> *op);
            nd4j::ops::DeclarableOp<float>* getOperationFloat(const char *name);


            /**
            * This method returns registered Op by name
            *
            * @param name
            * @return
            */
            nd4j::ops::DeclarableOp<float> *getOperationFloat(std::string& name);


            nd4j::ops::DeclarableOp<float> *getOperationFloat(Nd4jIndex hash);
            nd4j::ops::DeclarableOp<float16> *getOperationHalf(Nd4jIndex hash);
            nd4j::ops::DeclarableOp<float16>* getOperationHalf(const char *name);
            nd4j::ops::DeclarableOp<float16> *getOperationHalf(std::string& name);
            nd4j::ops::DeclarableOp<double >* getOperationDouble(const char *name);
            nd4j::ops::DeclarableOp<double> *getOperationDouble(Nd4jIndex hash);
            nd4j::ops::DeclarableOp<double> *getOperationDouble(std::string& name);
    };



        template <typename OpName>
        struct __registratorFloat {
            __registratorFloat();
        };

        template <typename OpName>
        struct __registratorHalf {
            __registratorHalf();
        };

        template <typename OpName>
        struct __registratorDouble {
            __registratorDouble();
        };

        template <typename OpName>
        struct __registratorSynonymFloat {
            __registratorSynonymFloat(const char *name, const char *oname);
        };

        template <typename OpName>
        struct __registratorSynonymHalf {
            __registratorSynonymHalf(const char *name, const char *oname);
        };

        template <typename OpName>
        struct __registratorSynonymDouble {
            __registratorSynonymDouble(const char *name, const char *oname);
        };

    }
}

#endif //LIBND4J_OPREGISTRATOR_H
