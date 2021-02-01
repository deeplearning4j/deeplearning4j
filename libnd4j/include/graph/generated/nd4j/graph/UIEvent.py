#  /* ******************************************************************************
#   *
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
#   *  See the NOTICE file distributed with this work for additional
#   *  information regarding copyright ownership.
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   * License for the specific language governing permissions and limitations
#   * under the License.
#   *
#   * SPDX-License-Identifier: Apache-2.0
#   ******************************************************************************/

import flatbuffers

class UIEvent(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsUIEvent(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = UIEvent()
        x.Init(buf, n + offset)
        return x

    # UIEvent
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # UIEvent
    def EventType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # UIEvent
    def EventSubType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # UIEvent
    def NameIdx(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # UIEvent
    def Timestamp(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    # UIEvent
    def Iteration(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # UIEvent
    def Epoch(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # UIEvent
    def VariableId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 0

    # UIEvent
    def FrameIter(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .FrameIteration import FrameIteration
            obj = FrameIteration()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # UIEvent
    def Plugin(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

def UIEventStart(builder): builder.StartObject(9)
def UIEventAddEventType(builder, eventType): builder.PrependInt8Slot(0, eventType, 0)
def UIEventAddEventSubType(builder, eventSubType): builder.PrependInt8Slot(1, eventSubType, 0)
def UIEventAddNameIdx(builder, nameIdx): builder.PrependInt32Slot(2, nameIdx, 0)
def UIEventAddTimestamp(builder, timestamp): builder.PrependInt64Slot(3, timestamp, 0)
def UIEventAddIteration(builder, iteration): builder.PrependInt32Slot(4, iteration, 0)
def UIEventAddEpoch(builder, epoch): builder.PrependInt32Slot(5, epoch, 0)
def UIEventAddVariableId(builder, variableId): builder.PrependInt16Slot(6, variableId, 0)
def UIEventAddFrameIter(builder, frameIter): builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(frameIter), 0)
def UIEventAddPlugin(builder, plugin): builder.PrependUint16Slot(8, plugin, 0)
def UIEventEnd(builder): return builder.EndObject()
