/* Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.optimize.listeners.stats.sbe;

import org.agrona.MutableDirectBuffer;
import org.agrona.DirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.optimize.listeners.stats.sbe.UpdateEncoder"})
@SuppressWarnings("all")
public class UpdateEncoder
{
    public static final int BLOCK_LENGTH = 8;
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
