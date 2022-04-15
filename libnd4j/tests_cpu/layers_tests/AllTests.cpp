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

//
// Created by raver119 on 04.08.17.
//
//
#include "testlayers.h"
/*
#include "ConvolutionTests.cpp"
#include "DeclarableOpsTests.cpp"
#include "DenseLayerTests.cpp"
#include "FlatBuffersTests.cpp"
#include "GraphTests.cpp"
#include "HashUtilsTests.cpp"
#include "NDArrayTests.cpp"
#include "SessionLocalTests.cpp"
#include "StashTests.cpp"
#include "TadTests.cpp"
#include "VariableSpaceTests.cpp"
#include "VariableTests.cpp"
#include "WorkspaceTests.cpp"
 */
///////

//#include "CyclicTests.h"
// #include "ProtoBufTests.cpp"

#if defined(HAVE_VEDA)
#include <libgen.h>
#include <linux/limits.h>
#include <unistd.h>

#include <string>
#include <ops/declarable/platform/vednn/veda_helper.h>
void load_device_lib() {
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  const char *path;
  if (count != -1) {
    path = dirname(result);
    sd::Environment::getInstance().setVedaDeviceDir( std::string(path)+"/../../blas/");
  }
}

#endif

int main(int argc, char **argv) {
#if defined(HAVE_VEDA)
  load_device_lib();
#endif
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
