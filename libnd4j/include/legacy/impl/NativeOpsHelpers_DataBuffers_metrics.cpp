/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
* the License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//
// Split from NativeOpsHelpers_DataBuffers.cpp to reduce object file size
// Contains: Transfer Metrics API Implementation
//

#include <legacy/NativeOps.h>
#include <helpers/TransferMetrics.h>
#include <types/types.h>

// =====================================================
// Transfer Metrics API Implementation
// =====================================================

void transferMetricsSetEnabled(bool enabled) {
  sd::TransferMetrics::getInstance().setEnabled(enabled);
}

bool transferMetricsIsEnabled() {
  return sd::TransferMetrics::getInstance().isEnabled();
}

void transferMetricsSetLogTransfers(bool log) {
  sd::TransferMetrics::getInstance().setLogTransfers(log);
}

void transferMetricsSetMinBytesForLogging(sd::LongType bytes) {
  sd::TransferMetrics::getInstance().setMinBytesForLogging(static_cast<uint64_t>(bytes));
}

sd::LongType transferMetricsGetH2DBytes() {
  return static_cast<sd::LongType>(
      sd::TransferMetrics::getInstance().getTotalBytes(sd::TransferType::HOST_TO_DEVICE));
}

sd::LongType transferMetricsGetD2HBytes() {
  return static_cast<sd::LongType>(
      sd::TransferMetrics::getInstance().getTotalBytes(sd::TransferType::DEVICE_TO_HOST));
}

sd::LongType transferMetricsGetD2DBytes() {
  return static_cast<sd::LongType>(
      sd::TransferMetrics::getInstance().getTotalBytes(sd::TransferType::DEVICE_TO_DEVICE));
}

sd::LongType transferMetricsGetP2PBytes() {
  return static_cast<sd::LongType>(
      sd::TransferMetrics::getInstance().getTotalBytes(sd::TransferType::PEER_TO_PEER));
}

sd::LongType transferMetricsGetH2DCount() {
  return static_cast<sd::LongType>(
      sd::TransferMetrics::getInstance().getTransferCount(sd::TransferType::HOST_TO_DEVICE));
}

sd::LongType transferMetricsGetD2HCount() {
  return static_cast<sd::LongType>(
      sd::TransferMetrics::getInstance().getTransferCount(sd::TransferType::DEVICE_TO_HOST));
}

sd::LongType transferMetricsGetD2DCount() {
  return static_cast<sd::LongType>(
      sd::TransferMetrics::getInstance().getTransferCount(sd::TransferType::DEVICE_TO_DEVICE));
}

sd::LongType transferMetricsGetP2PCount() {
  return static_cast<sd::LongType>(
      sd::TransferMetrics::getInstance().getTransferCount(sd::TransferType::PEER_TO_PEER));
}

sd::LongType transferMetricsGetTotalTimeNs() {
  return static_cast<sd::LongType>(
      sd::TransferMetrics::getInstance().getTotalTimeNsAllTypes());
}

double transferMetricsGetOverheadPercent() {
  return sd::TransferMetrics::getInstance().getOverheadPercent();
}

void transferMetricsReset() {
  sd::TransferMetrics::getInstance().reset();
}

void transferMetricsPrintSummary() {
  sd::TransferMetrics::getInstance().printSummary();
}

