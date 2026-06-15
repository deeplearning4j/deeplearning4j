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
#include <windows.h>
#else
// linux
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#endif

bool sd::memory::MemoryUtils::retrieveMemoryStatistics(MemoryReport &report) {
#if defined(__APPLE__)
  sd_debug("APPLE route\n", "");
  /*
      struct task_basic_info t_info;
      mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

      if (KERN_SUCCESS != task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&t_info, &t_info_count))
          return false;

      report.setVM(t_info.resident_size);
      report.setRSS(t_info.resident_size);


      sd_debug("RSS: %lld; VM: %lld;\n", report.getRSS(), report.getVM());
  */
  struct rusage _usage;

  auto res = getrusage(RUSAGE_SELF, &_usage);

  report.setRSS(_usage.ru_maxrss);

  sd_debug("Usage: %lld; %lld; %lld; %lld;\n", _usage.ru_ixrss, _usage.ru_idrss, _usage.ru_isrss, _usage.ru_maxrss);

  return true;
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
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
      report.setRSS((LongType)(atoll(s + 1) * getpagesize()));
    }
    close(fd);
  }


  return true;
#endif

  return false;
}

size_t sd::memory::MemoryUtils::getSystemFreeMemoryBytes() {
#if defined(__APPLE__)
  vm_statistics64_data_t vmstat;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  kern_return_t kr = host_statistics64(mach_host_self(), HOST_VM_INFO64,
                                       reinterpret_cast<host_info64_t>(&vmstat), &count);
  if (kr != KERN_SUCCESS) return 0;
  size_t pageSize = static_cast<size_t>(vm_page_size);
  return static_cast<size_t>(vmstat.free_count + vmstat.inactive_count) * pageSize;

#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (!GlobalMemoryStatusEx(&statex)) return 0;
  return static_cast<size_t>(statex.ullAvailPhys);

#else
  // Linux: parse MemAvailable from /proc/meminfo
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  size_t result = 0;
  while (fgets(line, sizeof(line), f)) {
    unsigned long long kbVal;
    if (sscanf(line, "MemAvailable: %llu kB", &kbVal) == 1) {
      result = static_cast<size_t>(kbVal) * 1024ULL;
      break;
    }
  }
  fclose(f);
  return result;
#endif
}
