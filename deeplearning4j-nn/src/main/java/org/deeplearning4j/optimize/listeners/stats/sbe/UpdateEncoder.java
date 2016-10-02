/* Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.optimize.listeners.stats.sbe;

import org.agrona.MutableDirectBuffer;
import org.agrona.DirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.optimize.listeners.stats.sbe.UpdateEncoder"})
@SuppressWarnings("all")
public class UpdateEncoder
{
    public static final int BLOCK_LENGTH = 28;
    public static final int TEMPLATE_ID = 2;
    public static final int SCHEMA_ID = 1;
    public static final int SCHEMA_VERSION = 0;

    private final UpdateEncoder parentMessage = this;
    private MutableDirectBuffer buffer;
    protected int offset;
    protected int limit;
    protected int actingBlockLength;
    protected int actingVersion;

    public int sbeBlockLength()
    {
        return BLOCK_LENGTH;
    }

    public int sbeTemplateId()
    {
        return TEMPLATE_ID;
    }

    public int sbeSchemaId()
    {
        return SCHEMA_ID;
    }

    public int sbeSchemaVersion()
    {
        return SCHEMA_VERSION;
    }

    public String sbeSemanticType()
    {
        return "";
    }

    public int offset()
    {
        return offset;
    }

    public UpdateEncoder wrap(final MutableDirectBuffer buffer, final int offset)
    {
        this.buffer = buffer;
        this.offset = offset;
        limit(offset + BLOCK_LENGTH);

        return this;
    }

    public int encodedLength()
    {
        return limit - offset;
    }

    public int limit()
    {
        return limit;
    }

    public void limit(final int limit)
    {
        this.limit = limit;
    }

    public static long timeNullValue()
    {
        return -9223372036854775808L;
    }

    public static long timeMinValue()
    {
        return -9223372036854775807L;
    }

    public static long timeMaxValue()
    {
        return 9223372036854775807L;
    }

    public UpdateEncoder time(final long value)
    {
        buffer.putLong(offset + 0, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static int deltaTimeNullValue()
    {
        return -2147483648;
    }

    public static int deltaTimeMinValue()
    {
        return -2147483647;
    }

    public static int deltaTimeMaxValue()
    {
        return 2147483647;
    }

    public UpdateEncoder deltaTime(final int value)
    {
        buffer.putInt(offset + 8, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    private final UpdateFieldsPresentEncoder fieldsPresent = new UpdateFieldsPresentEncoder();

    public UpdateFieldsPresentEncoder fieldsPresent()
    {
        fieldsPresent.wrap(buffer, offset + 12);
        return fieldsPresent;
    }

    public static int statsCollectionDurationNullValue()
    {
        return -2147483648;
    }

    public static int statsCollectionDurationMinValue()
    {
        return -2147483647;
    }

    public static int statsCollectionDurationMaxValue()
    {
        return 2147483647;
    }

    public UpdateEncoder statsCollectionDuration(final int value)
    {
        buffer.putInt(offset + 16, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static double scoreNullValue()
    {
        return Double.NaN;
    }

    public static double scoreMinValue()
    {
        return 4.9E-324d;
    }

    public static double scoreMaxValue()
    {
        return 1.7976931348623157E308d;
    }

    public UpdateEncoder score(final double value)
    {
        buffer.putDouble(offset + 20, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    private final MemoryUseEncoder memoryUse = new MemoryUseEncoder();

    public static long memoryUseId()
    {
        return 100;
    }

    public MemoryUseEncoder memoryUseCount(final int count)
    {
        memoryUse.wrap(parentMessage, buffer, count);
        return memoryUse;
    }

    public static class MemoryUseEncoder
    {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
        private UpdateEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(
            final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count)
        {
            if (count < 0 || count > 65534)
            {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int)9);
            dimensions.numInGroup((int)count);
            index = -1;
            this.count = count;
            blockLength = 9;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize()
        {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength()
        {
            return 9;
        }

        public MemoryUseEncoder next()
        {
            if (index + 1 >= count)
            {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }
        public MemoryUseEncoder memoryType(final MemoryType value)
        {
            buffer.putByte(offset + 0, (byte)value.value());
            return this;
        }

        public static long memoryBytesNullValue()
        {
            return -9223372036854775808L;
        }

        public static long memoryBytesMinValue()
        {
            return -9223372036854775807L;
        }

        public static long memoryBytesMaxValue()
        {
            return 9223372036854775807L;
        }

        public MemoryUseEncoder memoryBytes(final long value)
        {
            buffer.putLong(offset + 1, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }

    }

    private final PerformanceEncoder performance = new PerformanceEncoder();

    public static long performanceId()
    {
        return 200;
    }

    public PerformanceEncoder performanceCount(final int count)
    {
        performance.wrap(parentMessage, buffer, count);
        return performance;
    }

    public static class PerformanceEncoder
    {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
        private UpdateEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(
            final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count)
        {
            if (count < 0 || count > 65534)
            {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int)32);
            dimensions.numInGroup((int)count);
            index = -1;
            this.count = count;
            blockLength = 32;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize()
        {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength()
        {
            return 32;
        }

        public PerformanceEncoder next()
        {
            if (index + 1 >= count)
            {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static long totalRuntimeMsNullValue()
        {
            return -9223372036854775808L;
        }

        public static long totalRuntimeMsMinValue()
        {
            return -9223372036854775807L;
        }

        public static long totalRuntimeMsMaxValue()
        {
            return 9223372036854775807L;
        }

        public PerformanceEncoder totalRuntimeMs(final long value)
        {
            buffer.putLong(offset + 0, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static long totalExamplesNullValue()
        {
            return -9223372036854775808L;
        }

        public static long totalExamplesMinValue()
        {
            return -9223372036854775807L;
        }

        public static long totalExamplesMaxValue()
        {
            return 9223372036854775807L;
        }

        public PerformanceEncoder totalExamples(final long value)
        {
            buffer.putLong(offset + 8, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static long totalMinibatchesNullValue()
        {
            return -9223372036854775808L;
        }

        public static long totalMinibatchesMinValue()
        {
            return -9223372036854775807L;
        }

        public static long totalMinibatchesMaxValue()
        {
            return 9223372036854775807L;
        }

        public PerformanceEncoder totalMinibatches(final long value)
        {
            buffer.putLong(offset + 16, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static float examplesPerSecondNullValue()
        {
            return Float.NaN;
        }

        public static float examplesPerSecondMinValue()
        {
            return 1.401298464324817E-45f;
        }

        public static float examplesPerSecondMaxValue()
        {
            return 3.4028234663852886E38f;
        }

        public PerformanceEncoder examplesPerSecond(final float value)
        {
            buffer.putFloat(offset + 24, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static float minibatchesPerSecondNullValue()
        {
            return Float.NaN;
        }

        public static float minibatchesPerSecondMinValue()
        {
            return 1.401298464324817E-45f;
        }

        public static float minibatchesPerSecondMaxValue()
        {
            return 3.4028234663852886E38f;
        }

        public PerformanceEncoder minibatchesPerSecond(final float value)
        {
            buffer.putFloat(offset + 28, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }

    }

    private final GcStatsEncoder gcStats = new GcStatsEncoder();

    public static long gcStatsId()
    {
        return 300;
    }

    public GcStatsEncoder gcStatsCount(final int count)
    {
        gcStats.wrap(parentMessage, buffer, count);
        return gcStats;
    }

    public static class GcStatsEncoder
    {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
        private UpdateEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(
            final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count)
        {
            if (count < 0 || count > 65534)
            {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int)8);
            dimensions.numInGroup((int)count);
            index = -1;
            this.count = count;
            blockLength = 8;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize()
        {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength()
        {
            return 8;
        }

        public GcStatsEncoder next()
        {
            if (index + 1 >= count)
            {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int deltaGCCountNullValue()
        {
            return -2147483648;
        }

        public static int deltaGCCountMinValue()
        {
            return -2147483647;
        }

        public static int deltaGCCountMaxValue()
        {
            return 2147483647;
        }

        public GcStatsEncoder deltaGCCount(final int value)
        {
            buffer.putInt(offset + 0, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static int deltaGCTimeMsNullValue()
        {
            return -2147483648;
        }

        public static int deltaGCTimeMsMinValue()
        {
            return -2147483647;
        }

        public static int deltaGCTimeMsMaxValue()
        {
            return 2147483647;
        }

        public GcStatsEncoder deltaGCTimeMs(final int value)
        {
            buffer.putInt(offset + 4, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static int gcNameId()
        {
            return 1000;
        }

        public static String gcNameCharacterEncoding()
        {
            return "UTF-8";
        }

        public static String gcNameMetaAttribute(final MetaAttribute metaAttribute)
        {
            switch (metaAttribute)
            {
                case EPOCH: return "unix";
                case TIME_UNIT: return "nanosecond";
                case SEMANTIC_TYPE: return "";
            }

            return "";
        }

        public static int gcNameHeaderLength()
        {
            return 4;
        }

        public GcStatsEncoder putGcName(final DirectBuffer src, final int srcOffset, final int length)
        {
            if (length > 1073741824)
            {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, src, srcOffset, length);

            return this;
        }

        public GcStatsEncoder putGcName(final byte[] src, final int srcOffset, final int length)
        {
            if (length > 1073741824)
            {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, src, srcOffset, length);

            return this;
        }

        public GcStatsEncoder gcName(final String value)
        {
            final byte[] bytes;
            try
            {
                bytes = value.getBytes("UTF-8");
            }
            catch (final java.io.UnsupportedEncodingException ex)
            {
                throw new RuntimeException(ex);
            }

            final int length = bytes.length;
            if (length > 1073741824)
            {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, bytes, 0, length);

            return this;
        }
    }

    private final PerParameterStatsEncoder perParameterStats = new PerParameterStatsEncoder();

    public static long perParameterStatsId()
    {
        return 400;
    }

    public PerParameterStatsEncoder perParameterStatsCount(final int count)
    {
        perParameterStats.wrap(parentMessage, buffer, count);
        return perParameterStats;
    }

    public static class PerParameterStatsEncoder
    {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
        private UpdateEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(
            final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count)
        {
            if (count < 0 || count > 65534)
            {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int)2);
            dimensions.numInGroup((int)count);
            index = -1;
            this.count = count;
            blockLength = 2;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize()
        {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength()
        {
            return 2;
        }

        public PerParameterStatsEncoder next()
        {
            if (index + 1 >= count)
            {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int paramIDNullValue()
        {
            return 65535;
        }

        public static int paramIDMinValue()
        {
            return 0;
        }

        public static int paramIDMaxValue()
        {
            return 65534;
        }

        public PerParameterStatsEncoder paramID(final int value)
        {
            buffer.putShort(offset + 0, (short)value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        private final SummaryStatEncoder summaryStat = new SummaryStatEncoder();

        public static long summaryStatId()
        {
            return 402;
        }

        public SummaryStatEncoder summaryStatCount(final int count)
        {
            summaryStat.wrap(parentMessage, buffer, count);
            return summaryStat;
        }

        public static class SummaryStatEncoder
        {
            private static final int HEADER_SIZE = 4;
            private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
            private UpdateEncoder parentMessage;
            private MutableDirectBuffer buffer;
            private int blockLength;
            private int actingVersion;
            private int count;
            private int index;
            private int offset;

            public void wrap(
                final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count)
            {
                if (count < 0 || count > 65534)
                {
                    throw new IllegalArgumentException("count outside allowed range: count=" + count);
                }

                this.parentMessage = parentMessage;
                this.buffer = buffer;
                actingVersion = SCHEMA_VERSION;
                dimensions.wrap(buffer, parentMessage.limit());
                dimensions.blockLength((int)10);
                dimensions.numInGroup((int)count);
                index = -1;
                this.count = count;
                blockLength = 10;
                parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
            }

            public static int sbeHeaderSize()
            {
                return HEADER_SIZE;
            }

            public static int sbeBlockLength()
            {
                return 10;
            }

            public SummaryStatEncoder next()
            {
                if (index + 1 >= count)
                {
                    throw new java.util.NoSuchElementException();
                }

                offset = parentMessage.limit();
                parentMessage.limit(offset + blockLength);
                ++index;

                return this;
            }
            public SummaryStatEncoder statType(final StatsType value)
            {
                buffer.putByte(offset + 0, (byte)value.value());
                return this;
            }
            public SummaryStatEncoder summaryType(final SummaryType value)
            {
                buffer.putByte(offset + 1, (byte)value.value());
                return this;
            }

            public static double valueNullValue()
            {
                return Double.NaN;
            }

            public static double valueMinValue()
            {
                return 4.9E-324d;
            }

            public static double valueMaxValue()
            {
                return 1.7976931348623157E308d;
            }

            public SummaryStatEncoder value(final double value)
            {
                buffer.putDouble(offset + 2, value, java.nio.ByteOrder.LITTLE_ENDIAN);
                return this;
            }

        }

        private final HistogramsEncoder histograms = new HistogramsEncoder();

        public static long histogramsId()
        {
            return 406;
        }

        public HistogramsEncoder histogramsCount(final int count)
        {
            histograms.wrap(parentMessage, buffer, count);
            return histograms;
        }

        public static class HistogramsEncoder
        {
            private static final int HEADER_SIZE = 4;
            private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
            private UpdateEncoder parentMessage;
            private MutableDirectBuffer buffer;
            private int blockLength;
            private int actingVersion;
            private int count;
            private int index;
            private int offset;

            public void wrap(
                final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count)
            {
                if (count < 0 || count > 65534)
                {
                    throw new IllegalArgumentException("count outside allowed range: count=" + count);
                }

                this.parentMessage = parentMessage;
                this.buffer = buffer;
                actingVersion = SCHEMA_VERSION;
                dimensions.wrap(buffer, parentMessage.limit());
                dimensions.blockLength((int)21);
                dimensions.numInGroup((int)count);
                index = -1;
                this.count = count;
                blockLength = 21;
                parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
            }

            public static int sbeHeaderSize()
            {
                return HEADER_SIZE;
            }

            public static int sbeBlockLength()
            {
                return 21;
            }

            public HistogramsEncoder next()
            {
                if (index + 1 >= count)
                {
                    throw new java.util.NoSuchElementException();
                }

                offset = parentMessage.limit();
                parentMessage.limit(offset + blockLength);
                ++index;

                return this;
            }
            public HistogramsEncoder statType(final StatsType value)
            {
                buffer.putByte(offset + 0, (byte)value.value());
                return this;
            }

            public static double minValueNullValue()
            {
                return Double.NaN;
            }

            public static double minValueMinValue()
            {
                return 4.9E-324d;
            }

            public static double minValueMaxValue()
            {
                return 1.7976931348623157E308d;
            }

            public HistogramsEncoder minValue(final double value)
            {
                buffer.putDouble(offset + 1, value, java.nio.ByteOrder.LITTLE_ENDIAN);
                return this;
            }


            public static double maxValueNullValue()
            {
                return Double.NaN;
            }

            public static double maxValueMinValue()
            {
                return 4.9E-324d;
            }

            public static double maxValueMaxValue()
            {
                return 1.7976931348623157E308d;
            }

            public HistogramsEncoder maxValue(final double value)
            {
                buffer.putDouble(offset + 9, value, java.nio.ByteOrder.LITTLE_ENDIAN);
                return this;
            }


            public static int nBinsNullValue()
            {
                return -2147483648;
            }

            public static int nBinsMinValue()
            {
                return -2147483647;
            }

            public static int nBinsMaxValue()
            {
                return 2147483647;
            }

            public HistogramsEncoder nBins(final int value)
            {
                buffer.putInt(offset + 17, value, java.nio.ByteOrder.LITTLE_ENDIAN);
                return this;
            }


            private final HistogramCountsEncoder histogramCounts = new HistogramCountsEncoder();

            public static long histogramCountsId()
            {
                return 411;
            }

            public HistogramCountsEncoder histogramCountsCount(final int count)
            {
                histogramCounts.wrap(parentMessage, buffer, count);
                return histogramCounts;
            }

            public static class HistogramCountsEncoder
            {
                private static final int HEADER_SIZE = 4;
                private final GroupSizeEncodingEncoder dimensions = new GroupSizeEncodingEncoder();
                private UpdateEncoder parentMessage;
                private MutableDirectBuffer buffer;
                private int blockLength;
                private int actingVersion;
                private int count;
                private int index;
                private int offset;

                public void wrap(
                    final UpdateEncoder parentMessage, final MutableDirectBuffer buffer, final int count)
                {
                    if (count < 0 || count > 65534)
                    {
                        throw new IllegalArgumentException("count outside allowed range: count=" + count);
                    }

                    this.parentMessage = parentMessage;
                    this.buffer = buffer;
                    actingVersion = SCHEMA_VERSION;
                    dimensions.wrap(buffer, parentMessage.limit());
                    dimensions.blockLength((int)4);
                    dimensions.numInGroup((int)count);
                    index = -1;
                    this.count = count;
                    blockLength = 4;
                    parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
                }

                public static int sbeHeaderSize()
                {
                    return HEADER_SIZE;
                }

                public static int sbeBlockLength()
                {
                    return 4;
                }

                public HistogramCountsEncoder next()
                {
                    if (index + 1 >= count)
                    {
                        throw new java.util.NoSuchElementException();
                    }

                    offset = parentMessage.limit();
                    parentMessage.limit(offset + blockLength);
                    ++index;

                    return this;
                }

                public static long binCountNullValue()
                {
                    return 4294967294L;
                }

                public static long binCountMinValue()
                {
                    return 0L;
                }

                public static long binCountMaxValue()
                {
                    return 4294967293L;
                }

                public HistogramCountsEncoder binCount(final long value)
                {
                    buffer.putInt(offset + 0, (int)value, java.nio.ByteOrder.LITTLE_ENDIAN);
                    return this;
                }

            }
        }
    }
    public String toString()
    {
        return appendTo(new StringBuilder(100)).toString();
    }

    public StringBuilder appendTo(final StringBuilder builder)
    {
        UpdateDecoder writer = new UpdateDecoder();
        writer.wrap(buffer, offset, BLOCK_LENGTH, SCHEMA_VERSION);

        return writer.appendTo(builder);
    }
}
