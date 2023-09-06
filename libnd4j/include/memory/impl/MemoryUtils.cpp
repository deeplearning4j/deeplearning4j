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
// Created by raver119 on 11.10.2017.
//
#include "memory/MemoryUtils.h"

#include <helpers/logger.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)

#else
// linux
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cstring>
#endif

bool sd::memory::MemoryUtils::retrieveMemoryStatistics(sd::memory::MemoryReport &report) {
#if defined(__APPLE__)
  sd_debug("APPLE route\n", "");
  struct rusage _usage;

  auto res = getrusage(RUSAGE_SELF, &_usage);

  report.setRSS(_usage.ru_maxrss);

  sd_debug("Usage: %lld; %lld; %lld; %lld;\n", _usage.ru_ixrss, _usage.ru_idrss, _usage.ru_isrss, _usage.ru_maxrss);

  return true;
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__MINGW32__) || defined(__CYGWIN__)
  sd_debug("WIN32 route\n", "");

#else
  sd_debug("LINUX route\n", "");
  int fd = open("/proc/self/statm", O_RDONLY, 0);
  if (fd >= 0) {
    char line[256];
    char* s;
    int n;
    lseek(fd, 0, SEEK_SET);
    if ((n = read(fd, line, sizeof(line))) > 0 && (s = (char*)memchr(line, ' ', n)) != NULL) {
      report.setRSS((sd::LongType)(atoll(s + 1) * getpagesize()));
    }
    close(fd);
  }


  return true;
#endif

  return false;
}
