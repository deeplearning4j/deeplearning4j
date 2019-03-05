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
// Created by raver on 3/2/2019.
//

#ifndef DEV_TESTS_DECLARABLEBENCHMARK_H
#define DEV_TESTS_DECLARABLEBENCHMARK_H

#include <NDArray.h>
#include <Context.h>
#include <OpBenchmark.h>
#include <declarable/DeclarableOp.h>
#include <declarable/OpRegistrator.h>

namespace nd4j {
    class ND4J_EXPORT DeclarableBenchmark : public OpBenchmark  {
    protected:
    nd4j::ops::DeclarableOp *_op = nullptr;
    nd4j::graph::Context *_context = nullptr;
    public:
    DeclarableBenchmark(nd4j::ops::DeclarableOp &op, std::string name = 0) : OpBenchmark() {
        _op = ops::OpRegistrator::getInstance()->getOperation(op.getOpHash());
        _testName = name;
    }

    void setContext(nd4j::graph::Context *ctx) {
        _context = ctx;
    }

    std::string axis() override {
    return "N/A";
}

std::string orders() override {
return "N/A";
}

std::string strides() override {
return "N/A";
}

std::string inplace() override {
return "N/A";
}

void executeOnce() override {
_op->execute(_context);
}

OpBenchmark *clone() override {
return new DeclarableBenchmark(*_op, _testName);
}

std::string shape() override {
if (_context != nullptr && _context->isFastPath())
return ShapeUtils::shapeAsString(_context->getNDArray(0));
else
return "N/A";
}

std::string dataType() override {
if (_context != nullptr && _context->isFastPath())
return DataTypeUtils::asString(_context->getNDArray(0)->dataType());
else
return "N/A";
}

~DeclarableBenchmark() {
    if (_context != nullptr)
        delete _context;
}
};
}

#endif //DEV_TESTS_DECLARABLEBENCHMARKS_H