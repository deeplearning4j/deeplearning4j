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

/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

import org.agrona.DirectBuffer;
import org.agrona.MutableDirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.UpdateEncoder"})
@SuppressWarnings("all")
public class UpdateEncoder {
    public static final int BLOCK_LENGTH = 32;
    public static final int TEMPLATE_ID = 2;
    public static final int SCHEMA_ID = 1;
    public static final int SCHEMA_VERSION = 0;

    private final UpdateEncoder parentMessage = this;
    private MutableDirectBuffer buffer;
    protected int offset;
    protected int limit;
    protected int actingBlockLength;
    protected int actingVersion;

    public int sbeBlockLength() {
        return BLOCK_LENGTH;
    }

    public int sbeTemplateId() {
        return TEMPLATE_ID;
    }

    public int sbeSchemaId() {
        return SCHEMA_ID;
    }

    public int sbeSchemaVersion() {
        return SCHEMA_VERSION;
    }

    public String sbeSemanticType() {
        return "";
    }

    public int offset() {
        return offset;
    }

    public UpdateEncoder wrap(final MutableDirectBuffer buffer, final int offset) {
        this.buffer = buffer;
        this.offset = offset;
        limit(offset + BLOCK_LENGTH);

        return this;
    }

    public int encodedLength() {
        return limit - offset;
    }

    public int limit() {
        return limit;
    }

    public void limit(final int limit) {
        this.limit = limit;
    }

    public static long timeNullValue() {
        return -9223372036854775808L;
    }

    public static long timeMinValue() {
        return -9223372036854775807L;
    }

    public static long timeMaxValue() {
        return 9223372036854775807L;
    }

    public UpdateEncoder time(final long value) {
        buffer.putLong(offset + 0, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static int deltaTimeNullValue() {
        return -2147483648;
    }

    public static int deltaTimeMinValue() {
        return -2147483647;
    }

    public static int deltaTimeMaxValue() {
        return 2147483647;
    }

    public UpdateEncoder deltaTime(final int value) {
        buffer.putInt(offset + 8, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static int iterationCountNullValue() {
        return -2147483648;
    }

    public static int iterationCountMinValue() {
        return -2147483647;
    }

    public static int iterationCountMaxValue() {
        return 2147483647;
    }

    public UpdateEncoder iterationCount(final int value) {
        buffer.putInt(offset + 12, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    private final UpdateFieldsPresentEncoder fieldsPresent = new UpdateFieldsPresentEncoder();

    public UpdateFieldsPresentEncoder fieldsPresent() {
        fieldsPresent.wrap(buffer, offset + 16);
        return fieldsPresent;
    }

    public static int statsCollectionDurationNullValue() {
        return -2147483648;
    }

    public static int statsCollectionDurationMinValue() {
        return -2147483647;
    }

    public static int statsCollectionDurationMaxValue() {
        return 2147483647;
    }

    public UpdateEncoder statsCollectionDuration(final int value) {
        buffer.putInt(offset + 20, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static double scoreNullValue() {
        return Double.NaN;
    }

    public static double scoreMinValue() {
        return 4.9E-324d;
    }

    public static double scoreMaxValue() {
        return 1.7976931348623157E308d;
    }

    public UpdateEncoder score(final double value) {
        buffer.putDouble(offset + 24, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    private final MemoryUseEncoder memoryUse = new MemoryUseEncoder();

    public static long memoryUseId() {
        return 100;
    }

    public MemoryUseEncoder memoryUseCount(final int count) {
        memoryUse.wrap(parentMessage, buffer, count);
        return memoryUse;
    }

    public static class MemoryUseEncoder {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
        private UpdateEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count) {
            if (count < 0 || count > 65534) {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int) 9);
            dimensions.numInGroup((int) count);
            index = -1;
            this.count = count;
            blockLength = 9;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 9;
        }

        public MemoryUseEncoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public MemoryUseEncoder memoryType(final MemoryType value) {
            buffer.putByte(offset + 0, (byte) value.value());
            return this;
        }

        public static long memoryBytesNullValue() {
            return -9223372036854775808L;
        }

        public static long memoryBytesMinValue() {
            return -9223372036854775807L;
        }

        public static long memoryBytesMaxValue() {
            return 9223372036854775807L;
        }

        public MemoryUseEncoder memoryBytes(final long value) {
            buffer.putLong(offset + 1, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }

    }

    private final PerformanceEncoder performance = new PerformanceEncoder();

    public static long performanceId() {
        return 200;
    }

    public PerformanceEncoder performanceCount(final int count) {
        performance.wrap(parentMessage, buffer, count);
        return performance;
    }

    public static class PerformanceEncoder {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
        private UpdateEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count) {
            if (count < 0 || count > 65534) {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int) 32);
            dimensions.numInGroup((int) count);
            index = -1;
            this.count = count;
            blockLength = 32;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 32;
        }

        public PerformanceEncoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static long totalRuntimeMsNullValue() {
            return -9223372036854775808L;
        }

        public static long totalRuntimeMsMinValue() {
            return -9223372036854775807L;
        }

        public static long totalRuntimeMsMaxValue() {
            return 9223372036854775807L;
        }

        public PerformanceEncoder totalRuntimeMs(final long value) {
            buffer.putLong(offset + 0, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static long totalExamplesNullValue() {
            return -9223372036854775808L;
        }

        public static long totalExamplesMinValue() {
            return -9223372036854775807L;
        }

        public static long totalExamplesMaxValue() {
            return 9223372036854775807L;
        }

        public PerformanceEncoder totalExamples(final long value) {
            buffer.putLong(offset + 8, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static long totalMinibatchesNullValue() {
            return -9223372036854775808L;
        }

        public static long totalMinibatchesMinValue() {
            return -9223372036854775807L;
        }

        public static long totalMinibatchesMaxValue() {
            return 9223372036854775807L;
        }

        public PerformanceEncoder totalMinibatches(final long value) {
            buffer.putLong(offset + 16, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static float examplesPerSecondNullValue() {
            return Float.NaN;
        }

        public static float examplesPerSecondMinValue() {
            return 1.401298464324817E-45f;
        }

        public static float examplesPerSecondMaxValue() {
            return 3.4028234663852886E38f;
        }

        public PerformanceEncoder examplesPerSecond(final float value) {
            buffer.putFloat(offset + 24, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static float minibatchesPerSecondNullValue() {
            return Float.NaN;
        }

        public static float minibatchesPerSecondMinValue() {
            return 1.401298464324817E-45f;
        }

        public static float minibatchesPerSecondMaxValue() {
            return 3.4028234663852886E38f;
        }

        public PerformanceEncoder minibatchesPerSecond(final float value) {
            buffer.putFloat(offset + 28, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }

    }

    private final GcStatsEncoder gcStats = new GcStatsEncoder();

    public static long gcStatsId() {
        return 300;
    }

    public GcStatsEncoder gcStatsCount(final int count) {
        gcStats.wrap(parentMessage, buffer, count);
        return gcStats;
    }

    public static class GcStatsEncoder {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
        private UpdateEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count) {
            if (count < 0 || count > 65534) {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int) 8);
            dimensions.numInGroup((int) count);
            index = -1;
            this.count = count;
            blockLength = 8;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 8;
        }

        public GcStatsEncoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int deltaGCCountNullValue() {
            return -2147483648;
        }

        public static int deltaGCCountMinValue() {
            return -2147483647;
        }

        public static int deltaGCCountMaxValue() {
            return 2147483647;
        }

        public GcStatsEncoder deltaGCCount(final int value) {
            buffer.putInt(offset + 0, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static int deltaGCTimeMsNullValue() {
            return -2147483648;
        }

        public static int deltaGCTimeMsMinValue() {
            return -2147483647;
        }

        public static int deltaGCTimeMsMaxValue() {
            return 2147483647;
        }

        public GcStatsEncoder deltaGCTimeMs(final int value) {
            buffer.putInt(offset + 4, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static int gcNameId() {
            return 1000;
        }

        public static String gcNameCharacterEncoding() {
            return "UTF-8";
        }

        public static String gcNameMetaAttribute(final MetaAttribute metaAttribute) {
            switch (metaAttribute) {
                case EPOCH:
                    return "unix";
                case TIME_UNIT:
                    return "nanosecond";
                case SEMANTIC_TYPE:
                    return "";
            }

            return "";
        }

        public static int gcNameHeaderLength() {
            return 4;
        }

        public GcStatsEncoder putGcName(final DirectBuffer src, final int srcOffset, final int length) {
            if (length > 1073741824) {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, src, srcOffset, length);

            return this;
        }

        public GcStatsEncoder putGcName(final byte[] src, final int srcOffset, final int length) {
            if (length > 1073741824) {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, src, srcOffset, length);

            return this;
        }

        public GcStatsEncoder gcName(final String value) {
            final byte[] bytes;
            try {
                bytes = value.getBytes("UTF-8");
            } catch (final java.io.UnsupportedEncodingException ex) {
                throw new RuntimeException(ex);
            }

            final int length = bytes.length;
            if (length > 1073741824) {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, bytes, 0, length);

            return this;
        }
    }

    private final ParamNamesEncoder paramNames = new ParamNamesEncoder();

    public static long paramNamesId() {
        return 350;
    }

    public ParamNamesEncoder paramNamesCount(final int count) {
        paramNames.wrap(parentMessage, buffer, count);
        return paramNames;
    }

    public static class ParamNamesEncoder {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
        private UpdateEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count) {
            if (count < 0 || count > 65534) {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int) 0);
            dimensions.numInGroup((int) count);
            index = -1;
            this.count = count;
            blockLength = 0;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 0;
        }

        public ParamNamesEncoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int paramNameId() {
            return 1100;
        }

        public static String paramNameCharacterEncoding() {
            return "UTF-8";
        }

        public static String paramNameMetaAttribute(final MetaAttribute metaAttribute) {
            switch (metaAttribute) {
                case EPOCH:
                    return "unix";
                case TIME_UNIT:
                    return "nanosecond";
                case SEMANTIC_TYPE:
                    return "";
            }

            return "";
        }

        public static int paramNameHeaderLength() {
            return 4;
        }

        public ParamNamesEncoder putParamName(final DirectBuffer src, final int srcOffset, final int length) {
            if (length > 1073741824) {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, src, srcOffset, length);

            return this;
        }

        public ParamNamesEncoder putParamName(final byte[] src, final int srcOffset, final int length) {
            if (length > 1073741824) {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, src, srcOffset, length);

            return this;
        }

        public ParamNamesEncoder paramName(final String value) {
            final byte[] bytes;
            try {
                bytes = value.getBytes("UTF-8");
            } catch (final java.io.UnsupportedEncodingException ex) {
                throw new RuntimeException(ex);
            }

            final int length = bytes.length;
            if (length > 1073741824) {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, bytes, 0, length);

            return this;
        }
    }

    private final LayerNamesEncoder layerNames = new LayerNamesEncoder();

    public static long layerNamesId() {
        return 351;
    }

    public LayerNamesEncoder layerNamesCount(final int count) {
        layerNames.wrap(parentMessage, buffer, count);
        return layerNames;
    }

    public static class LayerNamesEncoder {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
        private UpdateEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count) {
            if (count < 0 || count > 65534) {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int) 0);
            dimensions.numInGroup((int) count);
            index = -1;
            this.count = count;
            blockLength = 0;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 0;
        }

        public LayerNamesEncoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int layerNameId() {
            return 1101;
        }

        public static String layerNameCharacterEncoding() {
            return "UTF-8";
        }

        public static String layerNameMetaAttribute(final MetaAttribute metaAttribute) {
            switch (metaAttribute) {
                case EPOCH:
                    return "unix";
                case TIME_UNIT:
                    return "nanosecond";
                case SEMANTIC_TYPE:
                    return "";
            }

            return "";
        }

        public static int layerNameHeaderLength() {
            return 4;
        }

        public LayerNamesEncoder putLayerName(final DirectBuffer src, final int srcOffset, final int length) {
            if (length > 1073741824) {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, src, srcOffset, length);

            return this;
        }

        public LayerNamesEncoder putLayerName(final byte[] src, final int srcOffset, final int length) {
            if (length > 1073741824) {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, src, srcOffset, length);

            return this;
        }

        public LayerNamesEncoder layerName(final String value) {
            final byte[] bytes;
            try {
                bytes = value.getBytes("UTF-8");
            } catch (final java.io.UnsupportedEncodingException ex) {
                throw new RuntimeException(ex);
            }

            final int length = bytes.length;
            if (length > 1073741824) {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, bytes, 0, length);

            return this;
        }
    }

    private final PerParameterStatsEncoder perParameterStats = new PerParameterStatsEncoder();

    public static long perParameterStatsId() {
        return 400;
    }

    public PerParameterStatsEncoder perParameterStatsCount(final int count) {
        perParameterStats.wrap(parentMessage, buffer, count);
        return perParameterStats;
    }

    public static class PerParameterStatsEncoder {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
        private UpdateEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count) {
            if (count < 0 || count > 65534) {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int) 4);
            dimensions.numInGroup((int) count);
            index = -1;
            this.count = count;
            blockLength = 4;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 4;
        }

        public PerParameterStatsEncoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static float learningRateNullValue() {
            return Float.NaN;
        }

        public static float learningRateMinValue() {
            return 1.401298464324817E-45f;
        }

        public static float learningRateMaxValue() {
            return 3.4028234663852886E38f;
        }

        public PerParameterStatsEncoder learningRate(final float value) {
            buffer.putFloat(offset + 0, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        private final SummaryStatEncoder summaryStat = new SummaryStatEncoder();

        public static long summaryStatId() {
            return 402;
        }

        public SummaryStatEncoder summaryStatCount(final int count) {
            summaryStat.wrap(parentMessage, buffer, count);
            return summaryStat;
        }

        public static class SummaryStatEncoder {
            private static final int HEADER_SIZE = 4;
            private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
            private UpdateEncoder parentMessage;
            private MutableDirectBuffer buffer;
            private int blockLength;
            private int actingVersion;
            private int count;
            private int index;
            private int offset;

            public void wrap(final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count) {
                if (count < 0 || count > 65534) {
                    throw new IllegalArgumentException("count outside allowed range: count=" + count);
                }

                this.parentMessage = parentMessage;
                this.buffer = buffer;
                actingVersion = SCHEMA_VERSION;
                dimensions.wrap(buffer, parentMessage.limit());
                dimensions.blockLength((int) 10);
                dimensions.numInGroup((int) count);
                index = -1;
                this.count = count;
                blockLength = 10;
                parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
            }

            public static int sbeHeaderSize() {
                return HEADER_SIZE;
            }

            public static int sbeBlockLength() {
                return 10;
            }

            public SummaryStatEncoder next() {
                if (index + 1 >= count) {
                    throw new java.util.NoSuchElementException();
                }

                offset = parentMessage.limit();
                parentMessage.limit(offset + blockLength);
                ++index;

                return this;
            }

            public SummaryStatEncoder statType(final StatsType value) {
                buffer.putByte(offset + 0, (byte) value.value());
                return this;
            }

            public SummaryStatEncoder summaryType(final SummaryType value) {
                buffer.putByte(offset + 1, (byte) value.value());
                return this;
            }

            public static double valueNullValue() {
                return Double.NaN;
            }

            public static double valueMinValue() {
                return 4.9E-324d;
            }

            public static double valueMaxValue() {
                return 1.7976931348623157E308d;
            }

            public SummaryStatEncoder value(final double value) {
                buffer.putDouble(offset + 2, value, java.nio.ByteOrder.LITTLE_ENDIAN);
                return this;
            }

        }

        private final HistogramsEncoder histograms = new HistogramsEncoder();

        public static long histogramsId() {
            return 406;
        }

        public HistogramsEncoder histogramsCount(final int count) {
            histograms.wrap(parentMessage, buffer, count);
            return histograms;
        }

        public static class HistogramsEncoder {
            private static final int HEADER_SIZE = 4;
            private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
            private UpdateEncoder parentMessage;
            private MutableDirectBuffer buffer;
            private int blockLength;
            private int actingVersion;
            private int count;
            private int index;
            private int offset;

            public void wrap(final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count) {
                if (count < 0 || count > 65534) {
                    throw new IllegalArgumentException("count outside allowed range: count=" + count);
                }

                this.parentMessage = parentMessage;
                this.buffer = buffer;
                actingVersion = SCHEMA_VERSION;
                dimensions.wrap(buffer, parentMessage.limit());
                dimensions.blockLength((int) 21);
                dimensions.numInGroup((int) count);
                index = -1;
                this.count = count;
                blockLength = 21;
                parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
            }

            public static int sbeHeaderSize() {
                return HEADER_SIZE;
            }

            public static int sbeBlockLength() {
                return 21;
            }

            public HistogramsEncoder next() {
                if (index + 1 >= count) {
                    throw new java.util.NoSuchElementException();
                }

                offset = parentMessage.limit();
                parentMessage.limit(offset + blockLength);
                ++index;

                return this;
            }

            public HistogramsEncoder statType(final StatsType value) {
                buffer.putByte(offset + 0, (byte) value.value());
                return this;
            }

            public static double minValueNullValue() {
                return Double.NaN;
            }

            public static double minValueMinValue() {
                return 4.9E-324d;
            }

            public static double minValueMaxValue() {
                return 1.7976931348623157E308d;
            }

            public HistogramsEncoder minValue(final double value) {
                buffer.putDouble(offset + 1, value, java.nio.ByteOrder.LITTLE_ENDIAN);
                return this;
            }


            public static double maxValueNullValue() {
                return Double.NaN;
            }

            public static double maxValueMinValue() {
                return 4.9E-324d;
            }

            public static double maxValueMaxValue() {
                return 1.7976931348623157E308d;
            }

            public HistogramsEncoder maxValue(final double value) {
                buffer.putDouble(offset + 9, value, java.nio.ByteOrder.LITTLE_ENDIAN);
                return this;
            }


            public static int nBinsNullValue() {
                return -2147483648;
            }

            public static int nBinsMinValue() {
                return -2147483647;
            }

            public static int nBinsMaxValue() {
                return 2147483647;
            }

            public HistogramsEncoder nBins(final int value) {
                buffer.putInt(offset + 17, value, java.nio.ByteOrder.LITTLE_ENDIAN);
                return this;
            }


            private final HistogramCountsEncoder histogramCounts = new HistogramCountsEncoder();

            public static long histogramCountsId() {
                return 411;
            }

            public HistogramCountsEncoder histogramCountsCount(final int count) {
                histogramCounts.wrap(parentMessage, buffer, count);
                return histogramCounts;
            }

            public static class HistogramCountsEncoder {
                private static final int HEADER_SIZE = 4;
                private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
                private UpdateEncoder parentMessage;
                private MutableDirectBuffer buffer;
                private int blockLength;
                private int actingVersion;
                private int count;
                private int index;
                private int offset;

                public void wrap(final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count) {
                    if (count < 0 || count > 65534) {
                        throw new IllegalArgumentException("count outside allowed range: count=" + count);
                    }

                    this.parentMessage = parentMessage;
                    this.buffer = buffer;
                    actingVersion = SCHEMA_VERSION;
                    dimensions.wrap(buffer, parentMessage.limit());
                    dimensions.blockLength((int) 4);
                    dimensions.numInGroup((int) count);
                    index = -1;
                    this.count = count;
                    blockLength = 4;
                    parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
                }

                public static int sbeHeaderSize() {
                    return HEADER_SIZE;
                }

                public static int sbeBlockLength() {
                    return 4;
                }

                public HistogramCountsEncoder next() {
                    if (index + 1 >= count) {
                        throw new java.util.NoSuchElementException();
                    }

                    offset = parentMessage.limit();
                    parentMessage.limit(offset + blockLength);
                    ++index;

                    return this;
                }

                public static long binCountNullValue() {
                    return 4294967294L;
                }

                public static long binCountMinValue() {
                    return 0L;
                }

                public static long binCountMaxValue() {
                    return 4294967293L;
                }

                public HistogramCountsEncoder binCount(final long value) {
                    buffer.putInt(offset + 0, (int) value, java.nio.ByteOrder.LITTLE_ENDIAN);
                    return this;
                }

            }
        }
    }

    private final DataSetMetaDataBytesEncoder dataSetMetaDataBytes = new DataSetMetaDataBytesEncoder();

    public static long dataSetMetaDataBytesId() {
        return 500;
    }

    public DataSetMetaDataBytesEncoder dataSetMetaDataBytesCount(final int count) {
        dataSetMetaDataBytes.wrap(parentMessage, buffer, count);
        return dataSetMetaDataBytes;
    }

    public static class DataSetMetaDataBytesEncoder {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
        private UpdateEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count) {
            if (count < 0 || count > 65534) {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int) 0);
            dimensions.numInGroup((int) count);
            index = -1;
            this.count = count;
            blockLength = 0;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 0;
        }

        public DataSetMetaDataBytesEncoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        private final MetaDataBytesEncoder metaDataBytes = new MetaDataBytesEncoder();

        public static long metaDataBytesId() {
            return 501;
        }

        public MetaDataBytesEncoder metaDataBytesCount(final int count) {
            metaDataBytes.wrap(parentMessage, buffer, count);
            return metaDataBytes;
        }

        public static class MetaDataBytesEncoder {
            private static final int HEADER_SIZE = 4;
            private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
            private UpdateEncoder parentMessage;
            private MutableDirectBuffer buffer;
            private int blockLength;
            private int actingVersion;
            private int count;
            private int index;
            private int offset;

            public void wrap(final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count) {
                if (count < 0 || count > 65534) {
                    throw new IllegalArgumentException("count outside allowed range: count=" + count);
                }

                this.parentMessage = parentMessage;
                this.buffer = buffer;
                actingVersion = SCHEMA_VERSION;
                dimensions.wrap(buffer, parentMessage.limit());
                dimensions.blockLength((int) 1);
                dimensions.numInGroup((int) count);
                index = -1;
                this.count = count;
                blockLength = 1;
                parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
            }

            public static int sbeHeaderSize() {
                return HEADER_SIZE;
            }

            public static int sbeBlockLength() {
                return 1;
            }

            public MetaDataBytesEncoder next() {
                if (index + 1 >= count) {
                    throw new java.util.NoSuchElementException();
                }

                offset = parentMessage.limit();
                parentMessage.limit(offset + blockLength);
                ++index;

                return this;
            }

            public static byte bytesNullValue() {
                return (byte) -128;
            }

            public static byte bytesMinValue() {
                return (byte) -127;
            }

            public static byte bytesMaxValue() {
                return (byte) 127;
            }

            public MetaDataBytesEncoder bytes(final byte value) {
                buffer.putByte(offset + 0, value);
                return this;
            }

        }
    }

    public static int sessionIDId() {
        return 1200;
    }

    public static String sessionIDCharacterEncoding() {
        return "UTF-8";
    }

    public static String sessionIDMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int sessionIDHeaderLength() {
        return 4;
    }

    public UpdateEncoder putSessionID(final DirectBuffer src, final int srcOffset, final int length) {
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public UpdateEncoder putSessionID(final byte[] src, final int srcOffset, final int length) {
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public UpdateEncoder sessionID(final String value) {
        final byte[] bytes;
        try {
            bytes = value.getBytes("UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public static int typeIDId() {
        return 1201;
    }

    public static String typeIDCharacterEncoding() {
        return "UTF-8";
    }

    public static String typeIDMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int typeIDHeaderLength() {
        return 4;
    }

    public UpdateEncoder putTypeID(final DirectBuffer src, final int srcOffset, final int length) {
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public UpdateEncoder putTypeID(final byte[] src, final int srcOffset, final int length) {
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public UpdateEncoder typeID(final String value) {
        final byte[] bytes;
        try {
            bytes = value.getBytes("UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public static int workerIDId() {
        return 1202;
    }

    public static String workerIDCharacterEncoding() {
        return "UTF-8";
    }

    public static String workerIDMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int workerIDHeaderLength() {
        return 4;
    }

    public UpdateEncoder putWorkerID(final DirectBuffer src, final int srcOffset, final int length) {
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public UpdateEncoder putWorkerID(final byte[] src, final int srcOffset, final int length) {
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public UpdateEncoder workerID(final String value) {
        final byte[] bytes;
        try {
            bytes = value.getBytes("UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public static int dataSetMetaDataClassNameId() {
        return 1300;
    }

    public static String dataSetMetaDataClassNameCharacterEncoding() {
        return "UTF-8";
    }

    public static String dataSetMetaDataClassNameMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int dataSetMetaDataClassNameHeaderLength() {
        return 4;
    }

    public UpdateEncoder putDataSetMetaDataClassName(final DirectBuffer src, final int srcOffset, final int length) {
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public UpdateEncoder putDataSetMetaDataClassName(final byte[] src, final int srcOffset, final int length) {
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public UpdateEncoder dataSetMetaDataClassName(final String value) {
        final byte[] bytes;
        try {
            bytes = value.getBytes("UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824) {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int) length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public String toString() {
        return appendTo(new StringBuilder(100)).toString();
    }

    public StringBuilder appendTo(final StringBuilder builder) {
        UpdateDecoder writer = new UpdateDecoder();
        writer.wrap(buffer, offset, BLOCK_LENGTH, SCHEMA_VERSION);

        return writer.appendTo(builder);
    }
}
