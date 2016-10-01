/* Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.optimize.listeners.stats.sbe.init;

import org.agrona.MutableDirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.optimize.listeners.stats.sbe.init.InitVarDataUTF8Encoder"})
@SuppressWarnings("all")
public class InitVarDataUTF8Encoder
{
    public static final int ENCODED_LENGTH = -1;
    private MutableDirectBuffer buffer;
    private int offset;

    public InitVarDataUTF8Encoder wrap(final MutableDirectBuffer buffer, final int offset)
    {
        this.buffer = buffer;
        this.offset = offset;

        return this;
    }

    public int encodedLength()
    {
        return ENCODED_LENGTH;
    }

    public static long lengthNullValue()
    {
        return 4294967294L;
    }

    public static long lengthMinValue()
    {
        return 0L;
    }

    public static long lengthMaxValue()
    {
        return 1073741824L;
    }

    public InitVarDataUTF8Encoder length(final long value)
    {
        buffer.putInt(offset + 0, (int)value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static short varDataNullValue()
    {
        return (short)255;
    }

    public static short varDataMinValue()
    {
        return (short)0;
    }

    public static short varDataMaxValue()
    {
        return (short)254;
    }
    public String toString()
    {
        return appendTo(new StringBuilder(100)).toString();
    }

    public StringBuilder appendTo(final StringBuilder builder)
    {
        InitVarDataUTF8Decoder writer = new InitVarDataUTF8Decoder();
        writer.wrap(buffer, offset);

        return writer.appendTo(builder);
    }
}
