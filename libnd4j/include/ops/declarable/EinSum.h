//
// Created by agibsonccc on 9/10/22.
//
/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_EINSUM_H
#define LIBND4J_EINSUM_H
#include <algorithm>
#include <system/op_boilerplate.h>
#include <vector>
#include <array/NDArray.h>
namespace samediff {
namespace einsum {
static std::string validString = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
static std::string operators = ",->.";

bool isValidEinSumChar(const char input);
char getSymbol(int i);
std::vector<char> getUnused(std::string used,int n,std::vector<std::string> ret);
void convertToValidEinSumChars(std::string einsumString,std::string ret);
void alphaCanoncalize(std::string equation,std::string& ret);
void findOutputString(std::string subscripts,std::string& ret);
bool hasOnlyValidEinsumChars(std::string input);
std::tuple<int,int,int> findOutputShape(std::vector<std::string> inputs,std::vector<std::vector<int>> shapes,std::string output);
void convertSubScripts(std::vector<std::string> oldSub,std::map<std::string,std::string> symbolMap,std::string & result);
void convertInterLeavedInput(std::vector<sd::NDArray *> operands,std::tuple<std::string,std::vector<std::string>> result);
void split(std::string input,std::string stringDelimiter,std::vector<std::string> result);
void join(std::string result,int beginIdx,int endIndx,std::vector<std::string> joinInput);
void joinWithDelimiter(std::string delimiter,std::vector<std::string> toJoin,std::string& result);
void parseEinsumInput(std::string inputOperands,
                      std::vector<sd::NDArray *> operands,
                      std::string & inputSubscriptsResult,
                      std::string & outputSubscriptResult,
                      std::vector<sd::NDArray *> & operandsOutput);
}
}

#endif  // LIBND4J_EINSUM_H
