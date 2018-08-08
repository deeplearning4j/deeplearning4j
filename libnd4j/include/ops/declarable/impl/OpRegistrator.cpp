/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 07.10.2017.
//



#include <ops/declarable/OpRegistrator.h>
#include <sstream>

namespace nd4j {
    namespace ops {

        ///////////////////////////////

        template <typename OpName>
        __registratorFloat<OpName>::__registratorFloat() {
            auto ptr = new OpName();
            OpRegistrator::getInstance()->registerOperationFloat(ptr);
        }

        template <typename OpName>
        __registratorHalf<OpName>::__registratorHalf() {
            auto ptr = new OpName();
            OpRegistrator::getInstance()->registerOperationHalf(ptr);
        }

        template <typename OpName>
        __registratorDouble<OpName>::__registratorDouble() {
            auto ptr = new OpName();
            OpRegistrator::getInstance()->registerOperationDouble(ptr);
        }

        template <typename OpName>
        __registratorInt<OpName>::__registratorInt() {
            auto ptr = new OpName();
            OpRegistrator::getInstance()->registerOperationInt(ptr);
        }

        template <typename OpName>
        __registratorLong<OpName>::__registratorLong() {
            auto ptr = new OpName();
            OpRegistrator::getInstance()->registerOperationLong(ptr);
        }

        template <typename OpName>
        __registratorSynonymFloat<OpName>::__registratorSynonymFloat(const char *name, const char *oname) {
            auto ptr = reinterpret_cast<OpName *>(OpRegistrator::getInstance()->getOperationFloat(oname));
            if (ptr == nullptr) {
                std::string newName(name);
                std::string oldName(oname);

                OpRegistrator::getInstance()->updateMSVC(nd4j::ops::HashHelper::getInstance()->getLongHash(newName), oldName);
                return;
            }
            OpRegistrator::getInstance()->registerOperationFloat(name, ptr);
        }

        template <typename OpName>
        __registratorSynonymHalf<OpName>::__registratorSynonymHalf(const char *name, const char *oname) {
            auto ptr = reinterpret_cast<OpName *>(OpRegistrator::getInstance()->getOperationHalf(oname));
            if (ptr == nullptr) {
                std::string newName(name);
                std::string oldName(oname);

                OpRegistrator::getInstance()->updateMSVC(nd4j::ops::HashHelper::getInstance()->getLongHash(newName), oldName);
                return;
            }
            OpRegistrator::getInstance()->registerOperationHalf(name, ptr);
        }

        template <typename OpName>
        __registratorSynonymDouble<OpName>::__registratorSynonymDouble(const char *name, const char *oname) {
            auto ptr = reinterpret_cast<OpName *>(OpRegistrator::getInstance()->getOperationDouble(oname));
            if (ptr == nullptr) {
                std::string newName(name);
                std::string oldName(oname);

                OpRegistrator::getInstance()->updateMSVC(nd4j::ops::HashHelper::getInstance()->getLongHash(newName), oldName);
                return;
            }
            OpRegistrator::getInstance()->registerOperationDouble(name, ptr);
        }

        template <typename OpName>
        __registratorSynonymInt<OpName>::__registratorSynonymInt(const char *name, const char *oname) {
            auto ptr = reinterpret_cast<OpName *>(OpRegistrator::getInstance()->getOperationInt(oname));
            if (ptr == nullptr) {
                std::string newName(name);
                std::string oldName(oname);

                OpRegistrator::getInstance()->updateMSVC(nd4j::ops::HashHelper::getInstance()->getLongHash(newName), oldName);
                return;
            }
            OpRegistrator::getInstance()->registerOperationInt(name, ptr);
        }

        template <typename OpName>
        __registratorSynonymLong<OpName>::__registratorSynonymLong(const char *name, const char *oname) {
            auto ptr = reinterpret_cast<OpName *>(OpRegistrator::getInstance()->getOperationLong(oname));
            if (ptr == nullptr) {
                std::string newName(name);
                std::string oldName(oname);

                OpRegistrator::getInstance()->updateMSVC(nd4j::ops::HashHelper::getInstance()->getLongHash(newName), oldName);
                return;
            }
            OpRegistrator::getInstance()->registerOperationLong(name, ptr);
        }
        ///////////////////////////////


        OpRegistrator* OpRegistrator::getInstance() {
            if (!_INSTANCE)
                _INSTANCE = new nd4j::ops::OpRegistrator();

            return _INSTANCE;
        }


        void OpRegistrator::updateMSVC(Nd4jLong newHash, std::string& oldName) {
            std::pair<Nd4jLong, std::string> pair(newHash, oldName);
            _msvc.insert(pair);
        }

        template <typename T>
        std::string OpRegistrator::local_to_string(T value) {
            //create an output string stream
            std::ostringstream os ;

            //throw the value into the string stream
            os << value ;

            //convert the string stream into a string and return
            return os.str() ;
        }

        template <>
        std::string OpRegistrator::local_to_string(int value) {
            //create an output string stream
            std::ostringstream os ;

            //throw the value into the string stream
            os << value ;

            //convert the string stream into a string and return
            return os.str() ;
        }

        void OpRegistrator::sigIntHandler(int sig) {
#ifndef _RELEASE
            delete OpRegistrator::getInstance();
#endif
        }

        void OpRegistrator::exitHandler() {
#ifndef _RELEASE
            delete OpRegistrator::getInstance();
#endif
        }

        void OpRegistrator::sigSegVHandler(int sig) {
#ifndef _RELEASE
            delete OpRegistrator::getInstance();
#endif
        }

        OpRegistrator::~OpRegistrator() {
#ifndef _RELEASE
            _msvc.clear();

            for (auto x : _uniqueD)
                delete x;

            for (auto x : _uniqueF)
                delete x;

            for (auto x : _uniqueH)
                delete x;

            for (auto x : _uniqueI)
                delete x;

            for (auto x : _uniqueL)
                delete x;

            _uniqueH.clear();
            _uniqueF.clear();
            _uniqueD.clear();
            _uniqueI.clear();
            _uniqueL.clear();

            _declarablesF.clear();
            _declarablesD.clear();
            _declarablesH.clear();
            _declarablesI.clear();
            _declarablesL.clear();

            _declarablesLD.clear();
            _declarablesLF.clear();
            _declarablesLH.clear();
            _declarablesLI.clear();
            _declarablesLL.clear();
#endif
        }

        const char * OpRegistrator::getAllCustomOperations() {
            _locker.lock();

            if (!isInit) {
                for (std::map<std::string, nd4j::ops::DeclarableOp<float>*>::iterator it=_declarablesF.begin(); it!=_declarablesF.end(); ++it) {
                    std::string op = it->first + ":"
                                     + local_to_string(it->second->getOpDescriptor()->getHash()) + ":"
                                     + local_to_string(it->second->getOpDescriptor()->getNumberOfInputs()) + ":"
                                     + local_to_string(it->second->getOpDescriptor()->getNumberOfOutputs()) + ":"
                                     + local_to_string(it->second->getOpDescriptor()->allowsInplace())  + ":"
                                     + local_to_string(it->second->getOpDescriptor()->getNumberOfTArgs())  + ":"
                                     + local_to_string(it->second->getOpDescriptor()->getNumberOfIArgs())  + ":"
                                     + ";" ;
                    _opsList += op;
                }

                isInit = true;
            }

            _locker.unlock();

            return _opsList.c_str();
        }

        bool OpRegistrator::registerOperationFloat(const char* name, nd4j::ops::DeclarableOp<float>* op) {
            std::string str(name);
            std::pair<std::string, nd4j::ops::DeclarableOp<float>*> pair(str, op);
            _declarablesF.insert(pair);

            auto hash = nd4j::ops::HashHelper::getInstance()->getLongHash(str);
            std::pair<Nd4jLong, nd4j::ops::DeclarableOp<float>*> pair2(hash, op);
            _declarablesLF.insert(pair2);

            //nd4j_printf("Adding op [%s]:[%lld]\n", name, hash);
            return true;
        }

        bool OpRegistrator::registerOperationDouble(const char* name, nd4j::ops::DeclarableOp<double>* op) {
            std::string str(name);
            std::pair<std::string, nd4j::ops::DeclarableOp<double>*> pair(str, op);
            _declarablesD.insert(pair);

            auto hash = nd4j::ops::HashHelper::getInstance()->getLongHash(str);
            std::pair<Nd4jLong, nd4j::ops::DeclarableOp<double>*> pair2(hash, op);
            _declarablesLD.insert(pair2);
            return true;
        }

        bool OpRegistrator::registerOperationHalf(const char* name, nd4j::ops::DeclarableOp<float16>* op) {
            std::string str(name);
            std::pair<std::string, nd4j::ops::DeclarableOp<float16>*> pair(str, op);
            _declarablesH.insert(pair);

            auto hash = nd4j::ops::HashHelper::getInstance()->getLongHash(str);
            std::pair<Nd4jLong, nd4j::ops::DeclarableOp<float16>*> pair2(hash, op);
            _declarablesLH.insert(pair2);
            return true;
        }

        bool OpRegistrator::registerOperationInt(const char* name, nd4j::ops::DeclarableOp<int>* op) {
            std::string str(name);
            std::pair<std::string, nd4j::ops::DeclarableOp<int>*> pair(str, op);
            _declarablesI.insert(pair);

            auto hash = nd4j::ops::HashHelper::getInstance()->getLongHash(str);
            std::pair<Nd4jLong, nd4j::ops::DeclarableOp<int>*> pair2(hash, op);
            _declarablesLI.insert(pair2);
            return true;
        }

        bool OpRegistrator::registerOperationLong(const char* name, nd4j::ops::DeclarableOp<Nd4jLong>* op) {
            std::string str(name);
            std::pair<std::string, nd4j::ops::DeclarableOp<Nd4jLong>*> pair(str, op);
            _declarablesL.insert(pair);

            auto hash = nd4j::ops::HashHelper::getInstance()->getLongHash(str);
            std::pair<Nd4jLong, nd4j::ops::DeclarableOp<Nd4jLong>*> pair2(hash, op);
            _declarablesLL.insert(pair2);
            return true;
        }
        /**
         * This method registers operation
         *
         * @param op
         */
        bool OpRegistrator::registerOperationFloat(nd4j::ops::DeclarableOp<float>* op) {
            _uniqueF.emplace_back(op);
            return registerOperationFloat(op->getOpName()->c_str(), op);
        }

        bool OpRegistrator::registerOperationHalf(nd4j::ops::DeclarableOp<float16> *op) {
            _uniqueH.emplace_back(op);
            return registerOperationHalf(op->getOpName()->c_str(), op);
        }

        bool OpRegistrator::registerOperationDouble(nd4j::ops::DeclarableOp<double> *op) {
            _uniqueD.emplace_back(op);
            return registerOperationDouble(op->getOpName()->c_str(), op);
        }

        bool OpRegistrator::registerOperationInt(nd4j::ops::DeclarableOp<int> *op) {
            _uniqueI.emplace_back(op);
            return registerOperationInt(op->getOpName()->c_str(), op);
        }

        bool OpRegistrator::registerOperationLong(nd4j::ops::DeclarableOp<Nd4jLong> *op) {
            _uniqueL.emplace_back(op);
            return registerOperationLong(op->getOpName()->c_str(), op);
        }

        nd4j::ops::DeclarableOp<float>* OpRegistrator::getOperationFloat(const char *name) {
            std::string str(name);
            return getOperationFloat(str);
        }

        /**
         * This method returns registered Op by name
         *
         * @param name
         * @return
         */
        nd4j::ops::DeclarableOp<float>* OpRegistrator::getOperationFloat(std::string& name) {
            if (!_declarablesF.count(name)) {
                nd4j_debug("Unknown operation requested: [%s]\n", name.c_str());
                return nullptr;
            }

            return _declarablesF.at(name);
        }


        nd4j::ops::DeclarableOp<float> *OpRegistrator::getOperationFloat(Nd4jLong hash) {
            if (!_declarablesLF.count(hash)) {
                if (!_msvc.count(hash)) {
                    nd4j_printf("Unknown F operation requested by hash: [%lld]\n", hash);
                    return nullptr;
                } else {
                    _locker.lock();

                    auto str = _msvc.at(hash);
                    auto op = _declarablesF.at(str);
                    auto oHash = op->getOpDescriptor()->getHash();

                    std::pair<Nd4jLong, nd4j::ops::DeclarableOp<float>*> pair(oHash, op);
                    _declarablesLF.insert(pair);

                    _locker.unlock();
                }
            }

            return _declarablesLF.at(hash);
        }


        nd4j::ops::DeclarableOp<float16> *OpRegistrator::getOperationHalf(Nd4jLong hash) {
            if (!_declarablesLH.count(hash)) {
                if (!_msvc.count(hash)) {
                    nd4j_printf("Unknown H operation requested by hash: [%lld]\n", hash);
                    return nullptr;
                } else {
                    _locker.lock();

                    auto str = _msvc.at(hash);
                    auto op = _declarablesH.at(str);
                    auto oHash = op->getOpDescriptor()->getHash();

                    std::pair<Nd4jLong, nd4j::ops::DeclarableOp<float16>*> pair(oHash, op);
                    _declarablesLH.insert(pair);

                    _locker.unlock();
                }
            }

            return _declarablesLH.at(hash);
        }


        nd4j::ops::DeclarableOp<float16>* OpRegistrator::getOperationHalf(const char *name) {
            std::string str(name);
            return getOperationHalf(str);
        }

        nd4j::ops::DeclarableOp<float16> *OpRegistrator::getOperationHalf(std::string& name) {
            if (!_declarablesH.count(name)) {
                nd4j_debug("Unknown operation requested: [%s]\n", name.c_str());
                return nullptr;
            }

            return _declarablesH.at(name);
        }


        nd4j::ops::DeclarableOp<double >* OpRegistrator::getOperationDouble(const char *name) {
            std::string str(name);
            return getOperationDouble(str);
        }


        nd4j::ops::DeclarableOp<double> *OpRegistrator::getOperationDouble(Nd4jLong hash) {
            if (!_declarablesLD.count(hash)) {
                if (!_msvc.count(hash)) {
                    nd4j_printf("Unknown D operation requested by hash: [%lld]\n", hash);
                    return nullptr;
                } else {
                    _locker.lock();

                    auto str = _msvc.at(hash);
                    auto op = _declarablesD.at(str);
                    auto oHash = op->getOpDescriptor()->getHash();

                    std::pair<Nd4jLong, nd4j::ops::DeclarableOp<double>*> pair(oHash, op);
                    _declarablesLD.insert(pair);

                    _locker.unlock();
                }
            }

            return _declarablesLD.at(hash);
        }

        nd4j::ops::DeclarableOp<double> *OpRegistrator::getOperationDouble(std::string& name) {
            if (!_declarablesD.count(name)) {
                nd4j_debug("Unknown operation requested: [%s]\n", name.c_str());
                return nullptr;
            }

            return _declarablesD.at(name);
        }

        nd4j::ops::DeclarableOp<int >* OpRegistrator::getOperationInt(const char *name) {
            std::string str(name);
            return getOperationInt(str);
        }


        nd4j::ops::DeclarableOp<int> *OpRegistrator::getOperationInt(Nd4jLong hash) {
            if (!_declarablesLI.count(hash)) {
                if (!_msvc.count(hash)) {
                    nd4j_printf("Unknown D operation requested by hash: [%lld]\n", hash);
                    return nullptr;
                } else {
                    _locker.lock();

                    auto str = _msvc.at(hash);
                    auto op = _declarablesI.at(str);
                    auto oHash = op->getOpDescriptor()->getHash();

                    std::pair<Nd4jLong, nd4j::ops::DeclarableOp<int>*> pair(oHash, op);
                    _declarablesLI.insert(pair);

                    _locker.unlock();
                }
            }

            return _declarablesLI.at(hash);
        }

        nd4j::ops::DeclarableOp<int> *OpRegistrator::getOperationInt(std::string& name) {
            if (!_declarablesI.count(name)) {
                nd4j_debug("Unknown operation requested: [%s]\n", name.c_str());
                return nullptr;
            }

            return _declarablesI.at(name);
        }

        nd4j::ops::DeclarableOp<Nd4jLong >* OpRegistrator::getOperationLong(const char *name) {
            std::string str(name);
            return getOperationLong(str);
        }


        nd4j::ops::DeclarableOp<Nd4jLong> *OpRegistrator::getOperationLong(Nd4jLong hash) {
            if (!_declarablesLL.count(hash)) {
                if (!_msvc.count(hash)) {
                    nd4j_printf("Unknown D operation requested by hash: [%lld]\n", hash);
                    return nullptr;
                } else {
                    _locker.lock();

                    auto str = _msvc.at(hash);
                    auto op = _declarablesL.at(str);
                    auto oHash = op->getOpDescriptor()->getHash();

                    std::pair<Nd4jLong, nd4j::ops::DeclarableOp<Nd4jLong>*> pair(oHash, op);
                    _declarablesLL.insert(pair);

                    _locker.unlock();
                }
            }

            return _declarablesLL.at(hash);
        }

        nd4j::ops::DeclarableOp<Nd4jLong> *OpRegistrator::getOperationLong(std::string& name) {
            if (!_declarablesI.count(name)) {
                nd4j_debug("Unknown operation requested: [%s]\n", name.c_str());
                return nullptr;
            }

            return _declarablesL.at(name);
        }

        template <>
        DeclarableOp<float> * OpRegistrator::getOperationT<float>(Nd4jLong hash) {
            return this->getOperationFloat(hash);
        }

        template <>
        DeclarableOp<float16> * OpRegistrator::getOperationT<float16>(Nd4jLong hash) {
            return this->getOperationHalf(hash);
        }

        template <>
        DeclarableOp<double> * OpRegistrator::getOperationT<double>(Nd4jLong hash) {
            return this->getOperationDouble(hash);
        }

        template <>
        DeclarableOp<int> * OpRegistrator::getOperationT<int>(Nd4jLong hash) {
            return this->getOperationInt(hash);
        }

        template <>
        DeclarableOp<Nd4jLong> * OpRegistrator::getOperationT<Nd4jLong>(Nd4jLong hash) {
            return this->getOperationLong(hash);
        }

        int OpRegistrator::numberOfOperations() {
            return (int) _declarablesLF.size();
        }

        template <>
        std::vector<Nd4jLong> OpRegistrator::getAllHashes<float>() {
            std::vector<Nd4jLong> result;

            for (auto &v:_declarablesLF) {
                result.emplace_back(v.first);
            }

            return result;
        }

        template <>
        std::vector<Nd4jLong> OpRegistrator::getAllHashes<double>() {
            std::vector<Nd4jLong> result;

            for (auto &v:_declarablesLD) {
                result.emplace_back(v.first);
            }

            return result;
        }

        template <>
        std::vector<Nd4jLong> OpRegistrator::getAllHashes<float16>() {
            std::vector<Nd4jLong> result;

            for (auto &v:_declarablesLH) {
                result.emplace_back(v.first);
            }

            return result;
        }

        template <>
        std::vector<Nd4jLong> OpRegistrator::getAllHashes<int>() {
            std::vector<Nd4jLong> result;

            for (auto &v:_declarablesLI) {
                result.emplace_back(v.first);
            }

            return result;
        }

        template <>
        std::vector<Nd4jLong> OpRegistrator::getAllHashes<Nd4jLong>() {
            std::vector<Nd4jLong> result;

            for (auto &v:_declarablesLL) {
                result.emplace_back(v.first);
            }

            return result;
        }
        nd4j::ops::OpRegistrator* nd4j::ops::OpRegistrator::_INSTANCE = 0;
    }
}

