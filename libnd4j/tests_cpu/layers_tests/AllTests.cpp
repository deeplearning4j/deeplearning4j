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
///////



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

using namespace testing;

class ConfigurableEventListener : public TestEventListener
{

 protected:
  TestEventListener* eventListener;

 public:

  /**
     * Show the names of each test case.
   */
  bool showTestCases;

  /**
     * Show the names of each test.
   */
  bool showTestNames;

  /**
     * Show each success.
   */
  bool showSuccesses;

  /**
     * Show each failure as it occurs. You will also see it at the bottom after the full suite is run.
   */
  bool showInlineFailures;

  /**
     * Show the setup of the global environment.
   */
  bool showEnvironment;

  explicit ConfigurableEventListener(TestEventListener* theEventListener) : eventListener(theEventListener)
  {
    showTestCases = true;
    showTestNames = true;
    showSuccesses = true;
    showInlineFailures = true;
    showEnvironment = true;
  }

  virtual ~ConfigurableEventListener()
  {
    delete eventListener;
  }

  virtual void OnTestProgramStart(const UnitTest& unit_test)
  {
    eventListener->OnTestProgramStart(unit_test);
  }

  virtual void OnTestIterationStart(const UnitTest& unit_test, int iteration)
  {
    eventListener->OnTestIterationStart(unit_test, iteration);
  }

  virtual void OnEnvironmentsSetUpStart(const UnitTest& unit_test)
  {
    if(showEnvironment) {
      eventListener->OnEnvironmentsSetUpStart(unit_test);
    }
  }

  virtual void OnEnvironmentsSetUpEnd(const UnitTest& unit_test)
  {
    if(showEnvironment) {
      eventListener->OnEnvironmentsSetUpEnd(unit_test);
    }
  }

  virtual void OnTestCaseStart(const TestCase& test_case)
  {
    if(showTestCases) {
      eventListener->OnTestCaseStart(test_case);
    }
  }

  virtual void OnTestStart(const TestInfo& test_info)
  {
    if(showTestNames) {
      eventListener->OnTestStart(test_info);
    }
  }

  virtual void OnTestPartResult(const TestPartResult& result)
  {
    eventListener->OnTestPartResult(result);
  }

  virtual void OnTestEnd(const TestInfo& test_info)
  {
    if((showInlineFailures && test_info.result()->Failed()) || (showSuccesses && !test_info.result()->Failed())) {
      eventListener->OnTestEnd(test_info);
    }
  }

  virtual void OnTestCaseEnd(const TestCase& test_case)
  {
    if(showTestCases) {
      eventListener->OnTestCaseEnd(test_case);
    }
  }

  virtual void OnEnvironmentsTearDownStart(const UnitTest& unit_test)
  {
    if(showEnvironment) {
      eventListener->OnEnvironmentsTearDownStart(unit_test);
    }
  }

  virtual void OnEnvironmentsTearDownEnd(const UnitTest& unit_test)
  {
    if(showEnvironment) {
      eventListener->OnEnvironmentsTearDownEnd(unit_test);
    }
  }

  virtual void OnTestIterationEnd(const UnitTest& unit_test, int iteration)
  {
    eventListener->OnTestIterationEnd(unit_test, iteration);
  }

  virtual void OnTestProgramEnd(const UnitTest& unit_test)
  {
    eventListener->OnTestProgramEnd(unit_test);
  }

};


int main(int argc, char **argv) {
#if defined(HAVE_VEDA)
  load_device_lib();
#endif
  InitGoogleTest(&argc, argv);

  TestEventListeners& listeners = UnitTest::GetInstance()->listeners();
  auto default_printer = listeners.Release(listeners.default_result_printer());

  // add our listener, by default everything is on (the same as using the default listener)
  // here I am turning everything off so I only see the 3 lines for the result
  // (plus any failures at the end), like:

  // [==========] Running 149 tests from 53 test cases.
  // [==========] 149 tests from 53 test cases ran. (1 ms total)
  // [  PASSED  ] 149 tests.

  ConfigurableEventListener *listener = new ConfigurableEventListener(default_printer);
  listener->showEnvironment = true;
  listener->showTestCases = true;
  listener->showTestNames = true;
  listener->showSuccesses = true;
  listener->showInlineFailures = true;
  listeners.Append(listener);

  return RUN_ALL_TESTS();
}
