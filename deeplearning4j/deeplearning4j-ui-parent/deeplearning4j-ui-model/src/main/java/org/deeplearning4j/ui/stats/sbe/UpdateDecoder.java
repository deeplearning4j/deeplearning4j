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

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.UpdateDecoder"})
@SuppressWarnings("all")
public class UpdateDecoder {
    public static final int BLOCK_LENGTH = 32;
    public static final int TEMPLATE_ID = 2;
    public static final int SCHEMA_ID = 1;
    public static final int SCHEMA_VERSION = 0;

    private final UpdateDecoder parentMessage = this;
    private DirectBuffer buffer;
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

    public UpdateDecoder wrap(final DirectBuffer buffer, final int offset, final int actingBlockLength,
                    final int actingVersion) {
        this.buffer = buffer;
        this.offset = offset;
        this.actingBlockLength = actingBlockLength;
        this.actingVersion = actingVersion;
        limit(offset + actingBlockLength);

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

    public static int timeId() {
        return 1;
    }

    public static String timeMetaAttribute(final MetaAttribute metaAttribute) {
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

    public static long timeNullValue() {
        return -9223372036854775808L;
    }

    public static long timeMinValue() {
        return -9223372036854775807L;
    }

    public static long timeMaxValue() {
        return 9223372036854775807L;
    }

    public long time() {
        return buffer.getLong(offset + 0, java.nio.ByteOrder.LITTLE_ENDIAN);
    }


    public static int deltaTimeId() {
        return 2;
    }

    public static String deltaTimeMetaAttribute(final MetaAttribute metaAttribute) {
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

    public static int deltaTimeNullValue() {
        return -2147483648;
    }

    public static int deltaTimeMinValue() {
        return -2147483647;
    }

    public static int deltaTimeMaxValue() {
        return 2147483647;
    }

    public int deltaTime() {
        return buffer.getInt(offset + 8, java.nio.ByteOrder.LITTLE_ENDIAN);
    }


    public static int iterationCountId() {
        return 3;
    }

    public static String iterationCountMetaAttribute(final MetaAttribute metaAttribute) {
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

    public static int iterationCountNullValue() {
        return -2147483648;
    }

    public static int iterationCountMinValue() {
        return -2147483647;
    }

    public static int iterationCountMaxValue() {
        return 2147483647;
    }

    public int iterationCount() {
        return buffer.getInt(offset + 12, java.nio.ByteOrder.LITTLE_ENDIAN);
    }


    public static int fieldsPresentId() {
        return 4;
    }

    public static String fieldsPresentMetaAttribute(final MetaAttribute metaAttribute) {
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

    private final UpdateFieldsPresentDecoder fieldsPresent = new UpdateFieldsPresentDecoder();

    public UpdateFieldsPresentDecoder fieldsPresent() {
        fieldsPresent.wrap(buffer, offset + 16);
        return fieldsPresent;
    }

    public static int statsCollectionDurationId() {
        return 5;
    }

    public static String statsCollectionDurationMetaAttribute(final MetaAttribute metaAttribute) {
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

    public static int statsCollectionDurationNullValue() {
        return -2147483648;
    }

    public static int statsCollectionDurationMinValue() {
        return -2147483647;
    }

    public static int statsCollectionDurationMaxValue() {
        return 2147483647;
    }

    public int statsCollectionDuration() {
        return buffer.getInt(offset + 20, java.nio.ByteOrder.LITTLE_ENDIAN);
    }


    public static int scoreId() {
        return 6;
    }

    public static String scoreMetaAttribute(final MetaAttribute metaAttribute) {
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

    public static double scoreNullValue() {
        return Double.NaN;
    }

    public static double scoreMinValue() {
        return 4.9E-324d;
    }

    public static double scoreMaxValue() {
        return 1.7976931348623157E308d;
    }

    public double score() {
        return buffer.getDouble(offset + 24, java.nio.ByteOrder.LITTLE_ENDIAN);
    }


    private final MemoryUseDecoder memoryUse = new MemoryUseDecoder();

    public static long memoryUseDecoderId() {
        return 100;
    }

    public MemoryUseDecoder memoryUse() {
        memoryUse.wrap(parentMessage, buffer);
        return memoryUse;
    }

    public static class MemoryUseDecoder implements Iterable<MemoryUseDecoder>, java.util.Iterator<MemoryUseDecoder> {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
        private UpdateDecoder parentMessage;
        private DirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateDecoder parentMessage, final DirectBuffer buffer) {
            this.parentMessage = parentMessage;
            this.buffer = buffer;
            dimensions.wrap(buffer, parentMessage.limit());
            blockLength = dimensions.blockLength();
            count = dimensions.numInGroup();
            index = -1;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 9;
        }

        public int actingBlockLength() {
            return blockLength;
        }

        public int count() {
            return count;
        }

        public java.util.Iterator<MemoryUseDecoder> iterator() {
            return this;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public boolean hasNext() {
            return (index + 1) < count;
        }

        public MemoryUseDecoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int memoryTypeId() {
            return 101;
        }

        public static String memoryTypeMetaAttribute(final MetaAttribute metaAttribute) {
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

        public MemoryType memoryType() {
            return MemoryType.get(((short) (buffer.getByte(offset + 0) & 0xFF)));
        }


        public static int memoryBytesId() {
            return 102;
        }

        public static String memoryBytesMetaAttribute(final MetaAttribute metaAttribute) {
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

        public static long memoryBytesNullValue() {
            return -9223372036854775808L;
        }

        public static long memoryBytesMinValue() {
            return -9223372036854775807L;
        }

        public static long memoryBytesMaxValue() {
            return 9223372036854775807L;
        }

        public long memoryBytes() {
            return buffer.getLong(offset + 1, java.nio.ByteOrder.LITTLE_ENDIAN);
        }


        public String toString() {
            return appendTo(new StringBuilder(100)).toString();
        }

        public StringBuilder appendTo(final StringBuilder builder) {
            builder.append('(');
            //Token{signal=BEGIN_FIELD, name='memoryType', description='null', id=101, version=0, encodedLength=0, offset=0, componentTokenCount=10, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=BEGIN_ENUM, name='MemoryType', description='null', id=-1, version=0, encodedLength=1, offset=0, componentTokenCount=8, encoding=Encoding{presence=REQUIRED, primitiveType=UINT8, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
            builder.append("memoryType=");
            builder.append(memoryType());
            builder.append('|');
            //Token{signal=BEGIN_FIELD, name='memoryBytes', description='null', id=102, version=0, encodedLength=0, offset=1, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=ENCODING, name='int64', description='null', id=-1, version=0, encodedLength=8, offset=1, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT64, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("memoryBytes=");
            builder.append(memoryBytes());
            builder.append(')');
            return builder;
        }
    }

    private final PerformanceDecoder performance = new PerformanceDecoder();

    public static long performanceDecoderId() {
        return 200;
    }

    public PerformanceDecoder performance() {
        performance.wrap(parentMessage, buffer);
        return performance;
    }

    public static class PerformanceDecoder
                    implements Iterable<PerformanceDecoder>, java.util.Iterator<PerformanceDecoder> {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
        private UpdateDecoder parentMessage;
        private DirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateDecoder parentMessage, final DirectBuffer buffer) {
            this.parentMessage = parentMessage;
            this.buffer = buffer;
            dimensions.wrap(buffer, parentMessage.limit());
            blockLength = dimensions.blockLength();
            count = dimensions.numInGroup();
            index = -1;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 32;
        }

        public int actingBlockLength() {
            return blockLength;
        }

        public int count() {
            return count;
        }

        public java.util.Iterator<PerformanceDecoder> iterator() {
            return this;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public boolean hasNext() {
            return (index + 1) < count;
        }

        public PerformanceDecoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int totalRuntimeMsId() {
            return 201;
        }

        public static String totalRuntimeMsMetaAttribute(final MetaAttribute metaAttribute) {
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

        public static long totalRuntimeMsNullValue() {
            return -9223372036854775808L;
        }

        public static long totalRuntimeMsMinValue() {
            return -9223372036854775807L;
        }

        public static long totalRuntimeMsMaxValue() {
            return 9223372036854775807L;
        }

        public long totalRuntimeMs() {
            return buffer.getLong(offset + 0, java.nio.ByteOrder.LITTLE_ENDIAN);
        }


        public static int totalExamplesId() {
            return 202;
        }

        public static String totalExamplesMetaAttribute(final MetaAttribute metaAttribute) {
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

        public static long totalExamplesNullValue() {
            return -9223372036854775808L;
        }

        public static long totalExamplesMinValue() {
            return -9223372036854775807L;
        }

        public static long totalExamplesMaxValue() {
            return 9223372036854775807L;
        }

        public long totalExamples() {
            return buffer.getLong(offset + 8, java.nio.ByteOrder.LITTLE_ENDIAN);
        }


        public static int totalMinibatchesId() {
            return 203;
        }

        public static String totalMinibatchesMetaAttribute(final MetaAttribute metaAttribute) {
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

        public static long totalMinibatchesNullValue() {
            return -9223372036854775808L;
        }

        public static long totalMinibatchesMinValue() {
            return -9223372036854775807L;
        }

        public static long totalMinibatchesMaxValue() {
            return 9223372036854775807L;
        }

        public long totalMinibatches() {
            return buffer.getLong(offset + 16, java.nio.ByteOrder.LITTLE_ENDIAN);
        }


        public static int examplesPerSecondId() {
            return 204;
        }

        public static String examplesPerSecondMetaAttribute(final MetaAttribute metaAttribute) {
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

        public static float examplesPerSecondNullValue() {
            return Float.NaN;
        }

        public static float examplesPerSecondMinValue() {
            return 1.401298464324817E-45f;
        }

        public static float examplesPerSecondMaxValue() {
            return 3.4028234663852886E38f;
        }

        public float examplesPerSecond() {
            return buffer.getFloat(offset + 24, java.nio.ByteOrder.LITTLE_ENDIAN);
        }


        public static int minibatchesPerSecondId() {
            return 205;
        }

        public static String minibatchesPerSecondMetaAttribute(final MetaAttribute metaAttribute) {
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

        public static float minibatchesPerSecondNullValue() {
            return Float.NaN;
        }

        public static float minibatchesPerSecondMinValue() {
            return 1.401298464324817E-45f;
        }

        public static float minibatchesPerSecondMaxValue() {
            return 3.4028234663852886E38f;
        }

        public float minibatchesPerSecond() {
            return buffer.getFloat(offset + 28, java.nio.ByteOrder.LITTLE_ENDIAN);
        }


        public String toString() {
            return appendTo(new StringBuilder(100)).toString();
        }

        public StringBuilder appendTo(final StringBuilder builder) {
            builder.append('(');
            //Token{signal=BEGIN_FIELD, name='totalRuntimeMs', description='null', id=201, version=0, encodedLength=0, offset=0, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=ENCODING, name='int64', description='null', id=-1, version=0, encodedLength=8, offset=0, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT64, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("totalRuntimeMs=");
            builder.append(totalRuntimeMs());
            builder.append('|');
            //Token{signal=BEGIN_FIELD, name='totalExamples', description='null', id=202, version=0, encodedLength=0, offset=8, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=ENCODING, name='int64', description='null', id=-1, version=0, encodedLength=8, offset=8, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT64, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("totalExamples=");
            builder.append(totalExamples());
            builder.append('|');
            //Token{signal=BEGIN_FIELD, name='totalMinibatches', description='null', id=203, version=0, encodedLength=0, offset=16, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=ENCODING, name='int64', description='null', id=-1, version=0, encodedLength=8, offset=16, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT64, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("totalMinibatches=");
            builder.append(totalMinibatches());
            builder.append('|');
            //Token{signal=BEGIN_FIELD, name='examplesPerSecond', description='null', id=204, version=0, encodedLength=0, offset=24, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=ENCODING, name='float', description='null', id=-1, version=0, encodedLength=4, offset=24, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=FLOAT, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("examplesPerSecond=");
            builder.append(examplesPerSecond());
            builder.append('|');
            //Token{signal=BEGIN_FIELD, name='minibatchesPerSecond', description='null', id=205, version=0, encodedLength=0, offset=28, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=ENCODING, name='float', description='null', id=-1, version=0, encodedLength=4, offset=28, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=FLOAT, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("minibatchesPerSecond=");
            builder.append(minibatchesPerSecond());
            builder.append(')');
            return builder;
        }
    }

    private final GcStatsDecoder gcStats = new GcStatsDecoder();

    public static long gcStatsDecoderId() {
        return 300;
    }

    public GcStatsDecoder gcStats() {
        gcStats.wrap(parentMessage, buffer);
        return gcStats;
    }

    public static class GcStatsDecoder implements Iterable<GcStatsDecoder>, java.util.Iterator<GcStatsDecoder> {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
        private UpdateDecoder parentMessage;
        private DirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateDecoder parentMessage, final DirectBuffer buffer) {
            this.parentMessage = parentMessage;
            this.buffer = buffer;
            dimensions.wrap(buffer, parentMessage.limit());
            blockLength = dimensions.blockLength();
            count = dimensions.numInGroup();
            index = -1;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 8;
        }

        public int actingBlockLength() {
            return blockLength;
        }

        public int count() {
            return count;
        }

        public java.util.Iterator<GcStatsDecoder> iterator() {
            return this;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public boolean hasNext() {
            return (index + 1) < count;
        }

        public GcStatsDecoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int deltaGCCountId() {
            return 301;
        }

        public static String deltaGCCountMetaAttribute(final MetaAttribute metaAttribute) {
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

        public static int deltaGCCountNullValue() {
            return -2147483648;
        }

        public static int deltaGCCountMinValue() {
            return -2147483647;
        }

        public static int deltaGCCountMaxValue() {
            return 2147483647;
        }

        public int deltaGCCount() {
            return buffer.getInt(offset + 0, java.nio.ByteOrder.LITTLE_ENDIAN);
        }


        public static int deltaGCTimeMsId() {
            return 302;
        }

        public static String deltaGCTimeMsMetaAttribute(final MetaAttribute metaAttribute) {
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

        public static int deltaGCTimeMsNullValue() {
            return -2147483648;
        }

        public static int deltaGCTimeMsMinValue() {
            return -2147483647;
        }

        public static int deltaGCTimeMsMaxValue() {
            return 2147483647;
        }

        public int deltaGCTimeMs() {
            return buffer.getInt(offset + 4, java.nio.ByteOrder.LITTLE_ENDIAN);
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

        public int gcNameLength() {
            final int limit = parentMessage.limit();
            return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        }

        public int getGcName(final MutableDirectBuffer dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public int getGcName(final byte[] dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public String gcName() {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            parentMessage.limit(limit + headerLength + dataLength);
            final byte[] tmp = new byte[dataLength];
            buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

            final String value;
            try {
                value = new String(tmp, "UTF-8");
            } catch (final java.io.UnsupportedEncodingException ex) {
                throw new RuntimeException(ex);
            }

            return value;
        }

        public String toString() {
            return appendTo(new StringBuilder(100)).toString();
        }

        public StringBuilder appendTo(final StringBuilder builder) {
            builder.append('(');
            //Token{signal=BEGIN_FIELD, name='deltaGCCount', description='null', id=301, version=0, encodedLength=0, offset=0, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=ENCODING, name='int32', description='null', id=-1, version=0, encodedLength=4, offset=0, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT32, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("deltaGCCount=");
            builder.append(deltaGCCount());
            builder.append('|');
            //Token{signal=BEGIN_FIELD, name='deltaGCTimeMs', description='null', id=302, version=0, encodedLength=0, offset=4, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=ENCODING, name='int32', description='null', id=-1, version=0, encodedLength=4, offset=4, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT32, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("deltaGCTimeMs=");
            builder.append(deltaGCTimeMs());
            builder.append('|');
            //Token{signal=BEGIN_VAR_DATA, name='gcName', description='null', id=1000, version=0, encodedLength=0, offset=8, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("gcName=");
            builder.append(gcName());
            builder.append(')');
            return builder;
        }
    }

    private final ParamNamesDecoder paramNames = new ParamNamesDecoder();

    public static long paramNamesDecoderId() {
        return 350;
    }

    public ParamNamesDecoder paramNames() {
        paramNames.wrap(parentMessage, buffer);
        return paramNames;
    }

    public static class ParamNamesDecoder
                    implements Iterable<ParamNamesDecoder>, java.util.Iterator<ParamNamesDecoder> {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
        private UpdateDecoder parentMessage;
        private DirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateDecoder parentMessage, final DirectBuffer buffer) {
            this.parentMessage = parentMessage;
            this.buffer = buffer;
            dimensions.wrap(buffer, parentMessage.limit());
            blockLength = dimensions.blockLength();
            count = dimensions.numInGroup();
            index = -1;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 0;
        }

        public int actingBlockLength() {
            return blockLength;
        }

        public int count() {
            return count;
        }

        public java.util.Iterator<ParamNamesDecoder> iterator() {
            return this;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public boolean hasNext() {
            return (index + 1) < count;
        }

        public ParamNamesDecoder next() {
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

        public int paramNameLength() {
            final int limit = parentMessage.limit();
            return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        }

        public int getParamName(final MutableDirectBuffer dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public int getParamName(final byte[] dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public String paramName() {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            parentMessage.limit(limit + headerLength + dataLength);
            final byte[] tmp = new byte[dataLength];
            buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

            final String value;
            try {
                value = new String(tmp, "UTF-8");
            } catch (final java.io.UnsupportedEncodingException ex) {
                throw new RuntimeException(ex);
            }

            return value;
        }

        public String toString() {
            return appendTo(new StringBuilder(100)).toString();
        }

        public StringBuilder appendTo(final StringBuilder builder) {
            builder.append('(');
            //Token{signal=BEGIN_VAR_DATA, name='paramName', description='null', id=1100, version=0, encodedLength=0, offset=0, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("paramName=");
            builder.append(paramName());
            builder.append(')');
            return builder;
        }
    }

    private final LayerNamesDecoder layerNames = new LayerNamesDecoder();

    public static long layerNamesDecoderId() {
        return 351;
    }

    public LayerNamesDecoder layerNames() {
        layerNames.wrap(parentMessage, buffer);
        return layerNames;
    }

    public static class LayerNamesDecoder
                    implements Iterable<LayerNamesDecoder>, java.util.Iterator<LayerNamesDecoder> {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
        private UpdateDecoder parentMessage;
        private DirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateDecoder parentMessage, final DirectBuffer buffer) {
            this.parentMessage = parentMessage;
            this.buffer = buffer;
            dimensions.wrap(buffer, parentMessage.limit());
            blockLength = dimensions.blockLength();
            count = dimensions.numInGroup();
            index = -1;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 0;
        }

        public int actingBlockLength() {
            return blockLength;
        }

        public int count() {
            return count;
        }

        public java.util.Iterator<LayerNamesDecoder> iterator() {
            return this;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public boolean hasNext() {
            return (index + 1) < count;
        }

        public LayerNamesDecoder next() {
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

        public int layerNameLength() {
            final int limit = parentMessage.limit();
            return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        }

        public int getLayerName(final MutableDirectBuffer dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public int getLayerName(final byte[] dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public String layerName() {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            parentMessage.limit(limit + headerLength + dataLength);
            final byte[] tmp = new byte[dataLength];
            buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

            final String value;
            try {
                value = new String(tmp, "UTF-8");
            } catch (final java.io.UnsupportedEncodingException ex) {
                throw new RuntimeException(ex);
            }

            return value;
        }

        public String toString() {
            return appendTo(new StringBuilder(100)).toString();
        }

        public StringBuilder appendTo(final StringBuilder builder) {
            builder.append('(');
            //Token{signal=BEGIN_VAR_DATA, name='layerName', description='null', id=1101, version=0, encodedLength=0, offset=0, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("layerName=");
            builder.append(layerName());
            builder.append(')');
            return builder;
        }
    }

    private final PerParameterStatsDecoder perParameterStats = new PerParameterStatsDecoder();

    public static long perParameterStatsDecoderId() {
        return 400;
    }

    public PerParameterStatsDecoder perParameterStats() {
        perParameterStats.wrap(parentMessage, buffer);
        return perParameterStats;
    }

    public static class PerParameterStatsDecoder
                    implements Iterable<PerParameterStatsDecoder>, java.util.Iterator<PerParameterStatsDecoder> {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
        private UpdateDecoder parentMessage;
        private DirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateDecoder parentMessage, final DirectBuffer buffer) {
            this.parentMessage = parentMessage;
            this.buffer = buffer;
            dimensions.wrap(buffer, parentMessage.limit());
            blockLength = dimensions.blockLength();
            count = dimensions.numInGroup();
            index = -1;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 4;
        }

        public int actingBlockLength() {
            return blockLength;
        }

        public int count() {
            return count;
        }

        public java.util.Iterator<PerParameterStatsDecoder> iterator() {
            return this;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public boolean hasNext() {
            return (index + 1) < count;
        }

        public PerParameterStatsDecoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int learningRateId() {
            return 401;
        }

        public static String learningRateMetaAttribute(final MetaAttribute metaAttribute) {
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

        public static float learningRateNullValue() {
            return Float.NaN;
        }

        public static float learningRateMinValue() {
            return 1.401298464324817E-45f;
        }

        public static float learningRateMaxValue() {
            return 3.4028234663852886E38f;
        }

        public float learningRate() {
            return buffer.getFloat(offset + 0, java.nio.ByteOrder.LITTLE_ENDIAN);
        }


        private final SummaryStatDecoder summaryStat = new SummaryStatDecoder();

        public static long summaryStatDecoderId() {
            return 402;
        }

        public SummaryStatDecoder summaryStat() {
            summaryStat.wrap(parentMessage, buffer);
            return summaryStat;
        }

        public static class SummaryStatDecoder
                        implements Iterable<SummaryStatDecoder>, java.util.Iterator<SummaryStatDecoder> {
            private static final int HEADER_SIZE = 4;
            private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
            private UpdateDecoder parentMessage;
            private DirectBuffer buffer;
            private int blockLength;
            private int actingVersion;
            private int count;
            private int index;
            private int offset;

            public void wrap(final UpdateDecoder parentMessage, final DirectBuffer buffer) {
                this.parentMessage = parentMessage;
                this.buffer = buffer;
                dimensions.wrap(buffer, parentMessage.limit());
                blockLength = dimensions.blockLength();
                count = dimensions.numInGroup();
                index = -1;
                parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
            }

            public static int sbeHeaderSize() {
                return HEADER_SIZE;
            }

            public static int sbeBlockLength() {
                return 10;
            }

            public int actingBlockLength() {
                return blockLength;
            }

            public int count() {
                return count;
            }

            public java.util.Iterator<SummaryStatDecoder> iterator() {
                return this;
            }

            public void remove() {
                throw new UnsupportedOperationException();
            }

            public boolean hasNext() {
                return (index + 1) < count;
            }

            public SummaryStatDecoder next() {
                if (index + 1 >= count) {
                    throw new java.util.NoSuchElementException();
                }

                offset = parentMessage.limit();
                parentMessage.limit(offset + blockLength);
                ++index;

                return this;
            }

            public static int statTypeId() {
                return 403;
            }

            public static String statTypeMetaAttribute(final MetaAttribute metaAttribute) {
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

            public StatsType statType() {
                return StatsType.get(((short) (buffer.getByte(offset + 0) & 0xFF)));
            }


            public static int summaryTypeId() {
                return 404;
            }

            public static String summaryTypeMetaAttribute(final MetaAttribute metaAttribute) {
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

            public SummaryType summaryType() {
                return SummaryType.get(((short) (buffer.getByte(offset + 1) & 0xFF)));
            }


            public static int valueId() {
                return 405;
            }

            public static String valueMetaAttribute(final MetaAttribute metaAttribute) {
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

            public static double valueNullValue() {
                return Double.NaN;
            }

            public static double valueMinValue() {
                return 4.9E-324d;
            }

            public static double valueMaxValue() {
                return 1.7976931348623157E308d;
            }

            public double value() {
                return buffer.getDouble(offset + 2, java.nio.ByteOrder.LITTLE_ENDIAN);
            }


            public String toString() {
                return appendTo(new StringBuilder(100)).toString();
            }

            public StringBuilder appendTo(final StringBuilder builder) {
                builder.append('(');
                //Token{signal=BEGIN_FIELD, name='statType', description='null', id=403, version=0, encodedLength=0, offset=0, componentTokenCount=8, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                //Token{signal=BEGIN_ENUM, name='StatsType', description='null', id=-1, version=0, encodedLength=1, offset=0, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=UINT8, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
                builder.append("statType=");
                builder.append(statType());
                builder.append('|');
                //Token{signal=BEGIN_FIELD, name='summaryType', description='null', id=404, version=0, encodedLength=0, offset=1, componentTokenCount=7, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                //Token{signal=BEGIN_ENUM, name='SummaryType', description='null', id=-1, version=0, encodedLength=1, offset=1, componentTokenCount=5, encoding=Encoding{presence=REQUIRED, primitiveType=UINT8, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
                builder.append("summaryType=");
                builder.append(summaryType());
                builder.append('|');
                //Token{signal=BEGIN_FIELD, name='value', description='null', id=405, version=0, encodedLength=0, offset=2, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                //Token{signal=ENCODING, name='double', description='null', id=-1, version=0, encodedLength=8, offset=2, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=DOUBLE, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                builder.append("value=");
                builder.append(value());
                builder.append(')');
                return builder;
            }
        }

        private final HistogramsDecoder histograms = new HistogramsDecoder();

        public static long histogramsDecoderId() {
            return 406;
        }

        public HistogramsDecoder histograms() {
            histograms.wrap(parentMessage, buffer);
            return histograms;
        }

        public static class HistogramsDecoder
                        implements Iterable<HistogramsDecoder>, java.util.Iterator<HistogramsDecoder> {
            private static final int HEADER_SIZE = 4;
            private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
            private UpdateDecoder parentMessage;
            private DirectBuffer buffer;
            private int blockLength;
            private int actingVersion;
            private int count;
            private int index;
            private int offset;

            public void wrap(final UpdateDecoder parentMessage, final DirectBuffer buffer) {
                this.parentMessage = parentMessage;
                this.buffer = buffer;
                dimensions.wrap(buffer, parentMessage.limit());
                blockLength = dimensions.blockLength();
                count = dimensions.numInGroup();
                index = -1;
                parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
            }

            public static int sbeHeaderSize() {
                return HEADER_SIZE;
            }

            public static int sbeBlockLength() {
                return 21;
            }

            public int actingBlockLength() {
                return blockLength;
            }

            public int count() {
                return count;
            }

            public java.util.Iterator<HistogramsDecoder> iterator() {
                return this;
            }

            public void remove() {
                throw new UnsupportedOperationException();
            }

            public boolean hasNext() {
                return (index + 1) < count;
            }

            public HistogramsDecoder next() {
                if (index + 1 >= count) {
                    throw new java.util.NoSuchElementException();
                }

                offset = parentMessage.limit();
                parentMessage.limit(offset + blockLength);
                ++index;

                return this;
            }

            public static int statTypeId() {
                return 407;
            }

            public static String statTypeMetaAttribute(final MetaAttribute metaAttribute) {
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

            public StatsType statType() {
                return StatsType.get(((short) (buffer.getByte(offset + 0) & 0xFF)));
            }


            public static int minValueId() {
                return 408;
            }

            public static String minValueMetaAttribute(final MetaAttribute metaAttribute) {
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

            public static double minValueNullValue() {
                return Double.NaN;
            }

            public static double minValueMinValue() {
                return 4.9E-324d;
            }

            public static double minValueMaxValue() {
                return 1.7976931348623157E308d;
            }

            public double minValue() {
                return buffer.getDouble(offset + 1, java.nio.ByteOrder.LITTLE_ENDIAN);
            }


            public static int maxValueId() {
                return 409;
            }

            public static String maxValueMetaAttribute(final MetaAttribute metaAttribute) {
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

            public static double maxValueNullValue() {
                return Double.NaN;
            }

            public static double maxValueMinValue() {
                return 4.9E-324d;
            }

            public static double maxValueMaxValue() {
                return 1.7976931348623157E308d;
            }

            public double maxValue() {
                return buffer.getDouble(offset + 9, java.nio.ByteOrder.LITTLE_ENDIAN);
            }


            public static int nBinsId() {
                return 410;
            }

            public static String nBinsMetaAttribute(final MetaAttribute metaAttribute) {
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

            public static int nBinsNullValue() {
                return -2147483648;
            }

            public static int nBinsMinValue() {
                return -2147483647;
            }

            public static int nBinsMaxValue() {
                return 2147483647;
            }

            public int nBins() {
                return buffer.getInt(offset + 17, java.nio.ByteOrder.LITTLE_ENDIAN);
            }


            private final HistogramCountsDecoder histogramCounts = new HistogramCountsDecoder();

            public static long histogramCountsDecoderId() {
                return 411;
            }

            public HistogramCountsDecoder histogramCounts() {
                histogramCounts.wrap(parentMessage, buffer);
                return histogramCounts;
            }

            public static class HistogramCountsDecoder
                            implements Iterable<HistogramCountsDecoder>, java.util.Iterator<HistogramCountsDecoder> {
                private static final int HEADER_SIZE = 4;
                private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
                private UpdateDecoder parentMessage;
                private DirectBuffer buffer;
                private int blockLength;
                private int actingVersion;
                private int count;
                private int index;
                private int offset;

                public void wrap(final UpdateDecoder parentMessage, final DirectBuffer buffer) {
                    this.parentMessage = parentMessage;
                    this.buffer = buffer;
                    dimensions.wrap(buffer, parentMessage.limit());
                    blockLength = dimensions.blockLength();
                    count = dimensions.numInGroup();
                    index = -1;
                    parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
                }

                public static int sbeHeaderSize() {
                    return HEADER_SIZE;
                }

                public static int sbeBlockLength() {
                    return 4;
                }

                public int actingBlockLength() {
                    return blockLength;
                }

                public int count() {
                    return count;
                }

                public java.util.Iterator<HistogramCountsDecoder> iterator() {
                    return this;
                }

                public void remove() {
                    throw new UnsupportedOperationException();
                }

                public boolean hasNext() {
                    return (index + 1) < count;
                }

                public HistogramCountsDecoder next() {
                    if (index + 1 >= count) {
                        throw new java.util.NoSuchElementException();
                    }

                    offset = parentMessage.limit();
                    parentMessage.limit(offset + blockLength);
                    ++index;

                    return this;
                }

                public static int binCountId() {
                    return 412;
                }

                public static String binCountMetaAttribute(final MetaAttribute metaAttribute) {
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

                public static long binCountNullValue() {
                    return 4294967294L;
                }

                public static long binCountMinValue() {
                    return 0L;
                }

                public static long binCountMaxValue() {
                    return 4294967293L;
                }

                public long binCount() {
                    return (buffer.getInt(offset + 0, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
                }


                public String toString() {
                    return appendTo(new StringBuilder(100)).toString();
                }

                public StringBuilder appendTo(final StringBuilder builder) {
                    builder.append('(');
                    //Token{signal=BEGIN_FIELD, name='binCount', description='null', id=412, version=0, encodedLength=0, offset=0, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                    //Token{signal=ENCODING, name='uint32', description='null', id=-1, version=0, encodedLength=4, offset=0, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=UINT32, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                    builder.append("binCount=");
                    builder.append(binCount());
                    builder.append(')');
                    return builder;
                }
            }

            public String toString() {
                return appendTo(new StringBuilder(100)).toString();
            }

            public StringBuilder appendTo(final StringBuilder builder) {
                builder.append('(');
                //Token{signal=BEGIN_FIELD, name='statType', description='null', id=407, version=0, encodedLength=0, offset=0, componentTokenCount=8, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                //Token{signal=BEGIN_ENUM, name='StatsType', description='null', id=-1, version=0, encodedLength=1, offset=0, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=UINT8, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
                builder.append("statType=");
                builder.append(statType());
                builder.append('|');
                //Token{signal=BEGIN_FIELD, name='minValue', description='null', id=408, version=0, encodedLength=0, offset=1, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                //Token{signal=ENCODING, name='double', description='null', id=-1, version=0, encodedLength=8, offset=1, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=DOUBLE, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                builder.append("minValue=");
                builder.append(minValue());
                builder.append('|');
                //Token{signal=BEGIN_FIELD, name='maxValue', description='null', id=409, version=0, encodedLength=0, offset=9, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                //Token{signal=ENCODING, name='double', description='null', id=-1, version=0, encodedLength=8, offset=9, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=DOUBLE, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                builder.append("maxValue=");
                builder.append(maxValue());
                builder.append('|');
                //Token{signal=BEGIN_FIELD, name='nBins', description='null', id=410, version=0, encodedLength=0, offset=17, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                //Token{signal=ENCODING, name='int32', description='null', id=-1, version=0, encodedLength=4, offset=17, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT32, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                builder.append("nBins=");
                builder.append(nBins());
                builder.append('|');
                //Token{signal=BEGIN_GROUP, name='histogramCounts', description='null', id=411, version=0, encodedLength=4, offset=21, componentTokenCount=9, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
                builder.append("histogramCounts=[");
                HistogramCountsDecoder histogramCounts = histogramCounts();
                if (histogramCounts.count() > 0) {
                    while (histogramCounts.hasNext()) {
                        histogramCounts.next().appendTo(builder);
                        builder.append(',');
                    }
                    builder.setLength(builder.length() - 1);
                }
                builder.append(']');
                builder.append(')');
                return builder;
            }
        }

        public String toString() {
            return appendTo(new StringBuilder(100)).toString();
        }

        public StringBuilder appendTo(final StringBuilder builder) {
            builder.append('(');
            //Token{signal=BEGIN_FIELD, name='learningRate', description='null', id=401, version=0, encodedLength=0, offset=0, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=ENCODING, name='float', description='null', id=-1, version=0, encodedLength=4, offset=0, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=FLOAT, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("learningRate=");
            builder.append(learningRate());
            builder.append('|');
            //Token{signal=BEGIN_GROUP, name='summaryStat', description='null', id=402, version=0, encodedLength=10, offset=4, componentTokenCount=24, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
            builder.append("summaryStat=[");
            SummaryStatDecoder summaryStat = summaryStat();
            if (summaryStat.count() > 0) {
                while (summaryStat.hasNext()) {
                    summaryStat.next().appendTo(builder);
                    builder.append(',');
                }
                builder.setLength(builder.length() - 1);
            }
            builder.append(']');
            builder.append('|');
            //Token{signal=BEGIN_GROUP, name='histograms', description='null', id=406, version=0, encodedLength=21, offset=-1, componentTokenCount=32, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
            builder.append("histograms=[");
            HistogramsDecoder histograms = histograms();
            if (histograms.count() > 0) {
                while (histograms.hasNext()) {
                    histograms.next().appendTo(builder);
                    builder.append(',');
                }
                builder.setLength(builder.length() - 1);
            }
            builder.append(']');
            builder.append(')');
            return builder;
        }
    }

    private final DataSetMetaDataBytesDecoder dataSetMetaDataBytes = new DataSetMetaDataBytesDecoder();

    public static long dataSetMetaDataBytesDecoderId() {
        return 500;
    }

    public DataSetMetaDataBytesDecoder dataSetMetaDataBytes() {
        dataSetMetaDataBytes.wrap(parentMessage, buffer);
        return dataSetMetaDataBytes;
    }

    public static class DataSetMetaDataBytesDecoder
                    implements Iterable<DataSetMetaDataBytesDecoder>, java.util.Iterator<DataSetMetaDataBytesDecoder> {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
        private UpdateDecoder parentMessage;
        private DirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final UpdateDecoder parentMessage, final DirectBuffer buffer) {
            this.parentMessage = parentMessage;
            this.buffer = buffer;
            dimensions.wrap(buffer, parentMessage.limit());
            blockLength = dimensions.blockLength();
            count = dimensions.numInGroup();
            index = -1;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 0;
        }

        public int actingBlockLength() {
            return blockLength;
        }

        public int count() {
            return count;
        }

        public java.util.Iterator<DataSetMetaDataBytesDecoder> iterator() {
            return this;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public boolean hasNext() {
            return (index + 1) < count;
        }

        public DataSetMetaDataBytesDecoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        private final MetaDataBytesDecoder metaDataBytes = new MetaDataBytesDecoder();

        public static long metaDataBytesDecoderId() {
            return 501;
        }

        public MetaDataBytesDecoder metaDataBytes() {
            metaDataBytes.wrap(parentMessage, buffer);
            return metaDataBytes;
        }

        public static class MetaDataBytesDecoder
                        implements Iterable<MetaDataBytesDecoder>, java.util.Iterator<MetaDataBytesDecoder> {
            private static final int HEADER_SIZE = 4;
            private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
            private UpdateDecoder parentMessage;
            private DirectBuffer buffer;
            private int blockLength;
            private int actingVersion;
            private int count;
            private int index;
            private int offset;

            public void wrap(final UpdateDecoder parentMessage, final DirectBuffer buffer) {
                this.parentMessage = parentMessage;
                this.buffer = buffer;
                dimensions.wrap(buffer, parentMessage.limit());
                blockLength = dimensions.blockLength();
                count = dimensions.numInGroup();
                index = -1;
                parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
            }

            public static int sbeHeaderSize() {
                return HEADER_SIZE;
            }

            public static int sbeBlockLength() {
                return 1;
            }

            public int actingBlockLength() {
                return blockLength;
            }

            public int count() {
                return count;
            }

            public java.util.Iterator<MetaDataBytesDecoder> iterator() {
                return this;
            }

            public void remove() {
                throw new UnsupportedOperationException();
            }

            public boolean hasNext() {
                return (index + 1) < count;
            }

            public MetaDataBytesDecoder next() {
                if (index + 1 >= count) {
                    throw new java.util.NoSuchElementException();
                }

                offset = parentMessage.limit();
                parentMessage.limit(offset + blockLength);
                ++index;

                return this;
            }

            public static int bytesId() {
                return 502;
            }

            public static String bytesMetaAttribute(final MetaAttribute metaAttribute) {
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

            public static byte bytesNullValue() {
                return (byte) -128;
            }

            public static byte bytesMinValue() {
                return (byte) -127;
            }

            public static byte bytesMaxValue() {
                return (byte) 127;
            }

            public byte bytes() {
                return buffer.getByte(offset + 0);
            }


            public String toString() {
                return appendTo(new StringBuilder(100)).toString();
            }

            public StringBuilder appendTo(final StringBuilder builder) {
                builder.append('(');
                //Token{signal=BEGIN_FIELD, name='bytes', description='null', id=502, version=0, encodedLength=0, offset=0, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                //Token{signal=ENCODING, name='int8', description='null', id=-1, version=0, encodedLength=1, offset=0, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT8, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
                builder.append("bytes=");
                builder.append(bytes());
                builder.append(')');
                return builder;
            }
        }

        public String toString() {
            return appendTo(new StringBuilder(100)).toString();
        }

        public StringBuilder appendTo(final StringBuilder builder) {
            builder.append('(');
            //Token{signal=BEGIN_GROUP, name='metaDataBytes', description='null', id=501, version=0, encodedLength=1, offset=0, componentTokenCount=9, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
            builder.append("metaDataBytes=[");
            MetaDataBytesDecoder metaDataBytes = metaDataBytes();
            if (metaDataBytes.count() > 0) {
                while (metaDataBytes.hasNext()) {
                    metaDataBytes.next().appendTo(builder);
                    builder.append(',');
                }
                builder.setLength(builder.length() - 1);
            }
            builder.append(']');
            builder.append(')');
            return builder;
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

    public int sessionIDLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getSessionID(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getSessionID(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String sessionID() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
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

    public int typeIDLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getTypeID(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getTypeID(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String typeID() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
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

    public int workerIDLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getWorkerID(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getWorkerID(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String workerID() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
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

    public int dataSetMetaDataClassNameLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getDataSetMetaDataClassName(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getDataSetMetaDataClassName(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String dataSetMetaDataClassName() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public String toString() {
        return appendTo(new StringBuilder(100)).toString();
    }

    public StringBuilder appendTo(final StringBuilder builder) {
        final int originalLimit = limit();
        limit(offset + actingBlockLength);
        builder.append("[Update](sbeTemplateId=");
        builder.append(TEMPLATE_ID);
        builder.append("|sbeSchemaId=");
        builder.append(SCHEMA_ID);
        builder.append("|sbeSchemaVersion=");
        if (actingVersion != SCHEMA_VERSION) {
            builder.append(actingVersion);
            builder.append('/');
        }
        builder.append(SCHEMA_VERSION);
        builder.append("|sbeBlockLength=");
        if (actingBlockLength != BLOCK_LENGTH) {
            builder.append(actingBlockLength);
            builder.append('/');
        }
        builder.append(BLOCK_LENGTH);
        builder.append("):");
        //Token{signal=BEGIN_FIELD, name='time', description='null', id=1, version=0, encodedLength=0, offset=0, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='int64', description='null', id=-1, version=0, encodedLength=8, offset=0, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT64, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("time=");
        builder.append(time());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='deltaTime', description='null', id=2, version=0, encodedLength=0, offset=8, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='int32', description='null', id=-1, version=0, encodedLength=4, offset=8, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT32, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("deltaTime=");
        builder.append(deltaTime());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='iterationCount', description='null', id=3, version=0, encodedLength=0, offset=12, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='int32', description='null', id=-1, version=0, encodedLength=4, offset=12, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT32, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("iterationCount=");
        builder.append(iterationCount());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='fieldsPresent', description='null', id=4, version=0, encodedLength=0, offset=16, componentTokenCount=26, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=BEGIN_SET, name='UpdateFieldsPresent', description='null', id=-1, version=0, encodedLength=4, offset=16, componentTokenCount=24, encoding=Encoding{presence=REQUIRED, primitiveType=UINT32, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='UpdateFieldsPresent'}}
        builder.append("fieldsPresent=");
        builder.append(fieldsPresent());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='statsCollectionDuration', description='null', id=5, version=0, encodedLength=0, offset=20, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='int32', description='null', id=-1, version=0, encodedLength=4, offset=20, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT32, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("statsCollectionDuration=");
        builder.append(statsCollectionDuration());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='score', description='null', id=6, version=0, encodedLength=0, offset=24, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='double', description='null', id=-1, version=0, encodedLength=8, offset=24, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=DOUBLE, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("score=");
        builder.append(score());
        builder.append('|');
        //Token{signal=BEGIN_GROUP, name='memoryUse', description='null', id=100, version=0, encodedLength=9, offset=32, componentTokenCount=19, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
        builder.append("memoryUse=[");
        MemoryUseDecoder memoryUse = memoryUse();
        if (memoryUse.count() > 0) {
            while (memoryUse.hasNext()) {
                memoryUse.next().appendTo(builder);
                builder.append(',');
            }
            builder.setLength(builder.length() - 1);
        }
        builder.append(']');
        builder.append('|');
        //Token{signal=BEGIN_GROUP, name='performance', description='null', id=200, version=0, encodedLength=32, offset=-1, componentTokenCount=21, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
        builder.append("performance=[");
        PerformanceDecoder performance = performance();
        if (performance.count() > 0) {
            while (performance.hasNext()) {
                performance.next().appendTo(builder);
                builder.append(',');
            }
            builder.setLength(builder.length() - 1);
        }
        builder.append(']');
        builder.append('|');
        //Token{signal=BEGIN_GROUP, name='gcStats', description='null', id=300, version=0, encodedLength=8, offset=-1, componentTokenCount=18, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
        builder.append("gcStats=[");
        GcStatsDecoder gcStats = gcStats();
        if (gcStats.count() > 0) {
            while (gcStats.hasNext()) {
                gcStats.next().appendTo(builder);
                builder.append(',');
            }
            builder.setLength(builder.length() - 1);
        }
        builder.append(']');
        builder.append('|');
        //Token{signal=BEGIN_GROUP, name='paramNames', description='null', id=350, version=0, encodedLength=0, offset=-1, componentTokenCount=12, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
        builder.append("paramNames=[");
        ParamNamesDecoder paramNames = paramNames();
        if (paramNames.count() > 0) {
            while (paramNames.hasNext()) {
                paramNames.next().appendTo(builder);
                builder.append(',');
            }
            builder.setLength(builder.length() - 1);
        }
        builder.append(']');
        builder.append('|');
        //Token{signal=BEGIN_GROUP, name='layerNames', description='null', id=351, version=0, encodedLength=0, offset=-1, componentTokenCount=12, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
        builder.append("layerNames=[");
        LayerNamesDecoder layerNames = layerNames();
        if (layerNames.count() > 0) {
            while (layerNames.hasNext()) {
                layerNames.next().appendTo(builder);
                builder.append(',');
            }
            builder.setLength(builder.length() - 1);
        }
        builder.append(']');
        builder.append('|');
        //Token{signal=BEGIN_GROUP, name='perParameterStats', description='null', id=400, version=0, encodedLength=4, offset=-1, componentTokenCount=65, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
        builder.append("perParameterStats=[");
        PerParameterStatsDecoder perParameterStats = perParameterStats();
        if (perParameterStats.count() > 0) {
            while (perParameterStats.hasNext()) {
                perParameterStats.next().appendTo(builder);
                builder.append(',');
            }
            builder.setLength(builder.length() - 1);
        }
        builder.append(']');
        builder.append('|');
        //Token{signal=BEGIN_GROUP, name='dataSetMetaDataBytes', description='null', id=500, version=0, encodedLength=0, offset=-1, componentTokenCount=15, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
        builder.append("dataSetMetaDataBytes=[");
        DataSetMetaDataBytesDecoder dataSetMetaDataBytes = dataSetMetaDataBytes();
        if (dataSetMetaDataBytes.count() > 0) {
            while (dataSetMetaDataBytes.hasNext()) {
                dataSetMetaDataBytes.next().appendTo(builder);
                builder.append(',');
            }
            builder.setLength(builder.length() - 1);
        }
        builder.append(']');
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='sessionID', description='null', id=1200, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("sessionID=");
        builder.append(sessionID());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='typeID', description='null', id=1201, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("typeID=");
        builder.append(typeID());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='workerID', description='null', id=1202, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("workerID=");
        builder.append(workerID());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='dataSetMetaDataClassName', description='null', id=1300, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("dataSetMetaDataClassName=");
        builder.append(dataSetMetaDataClassName());

        limit(originalLimit);

        return builder;
    }
}
