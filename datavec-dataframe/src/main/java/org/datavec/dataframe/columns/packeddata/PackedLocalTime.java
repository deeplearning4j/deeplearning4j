package org.datavec.dataframe.columns.packeddata;

import com.google.common.base.Strings;
import com.google.common.primitives.Ints;

import java.time.LocalTime;

/**
 * A localTime with millisecond precision packed into a single int value.
 * <p>
 * The bytes are packed into the int as:
 * First byte: hourOfDay
 * next byte: minuteOfHour
 * last two bytes (short): millisecond of minute
 * <p>
 * Storing the millisecond of minute in an short requires that we treat the short as if it were unsigned. Unfortunately,
 * Neither Java nor Guava provide unsigned short support so we use char, which is a 16-bit unsigned int to
 * store values of up to 60,000 milliseconds (60 secs * 1000)
 */
public class PackedLocalTime {

  public static final int MIDNIGHT = pack(LocalTime.MIDNIGHT);
  public static final int NOON = pack(LocalTime.NOON);

  public static byte getHour(int time) {
    return (byte) (time >> 24);
  }

  public static char getMillisecondOfMinute(int time) {
    byte byte1 = (byte) (time >> 8);
    byte byte2 = (byte) time;
    return (char) ((byte1 << 8) | (byte2 & 0xFF));
  }

  public static int getNano(int time) {
    long millis = getMillisecondOfMinute(time);
    millis = millis * 1_000_000L; // convert to nanos of minute
    byte seconds = getSecond(time);
    long nanos = seconds * 1_000_000_000L;
    millis = millis - nanos;         // remove the part in seconds
    return (int) millis;
  }

  public static int getMilliseconds(int time) {
    long millis = getMillisecondOfMinute(time);
    millis = millis * 1_000_000L; // convert to nanos of minute
    byte seconds = getSecond(time);
    long nanos = seconds * 1_000_000_000L;
    millis = millis - nanos;         // remove the part in seconds
    return (int) (millis / 1_000_000L);
  }

  public static long toNanoOfDay(int time) {
    long nano = getHour(time) * 3_600_000_000_000L;
    nano += getMinute(time) * 60_000_000_000L;
    nano += getSecond(time) * 1_000_000_000L;
    nano += getNano(time);
    return nano;
  }

  public static LocalTime asLocalTime(int time) {
    if (time == -1) {
      return null;
    }

    byte hourByte = (byte) (time >> 24);
    byte minuteByte = (byte) (time >> 16);
    byte millisecondByte1 = (byte) (time >> 8);
    byte millisecondByte2 = (byte) time;
    char millis = (char) ((millisecondByte1 << 8) | (millisecondByte2 & 0xFF));
    int second = millis / 1000;
    int nanoOfSecond = (millis % 1000) * 1_000_000;
    return LocalTime.of(
        hourByte,
        minuteByte,
        second,
        nanoOfSecond);
  }

  public static byte getMinute(int time) {
    return (byte) (time >> 16);
  }

  public static int pack(LocalTime time) {
    byte hour = (byte) time.getHour();
    byte minute = (byte) time.getMinute();
    char millis = (char) (time.getNano() / 1_000_000.0);
    millis = (char) (millis + (char) (time.getSecond() * 1000));
    byte m1 = (byte) (millis >> 8);
    byte m2 = (byte) millis;

    return Ints.fromBytes(
        hour,
        minute,
        m1,
        m2);
  }

  public static byte getSecond(int packedLocalTime) {
    return (byte) (getMillisecondOfMinute(packedLocalTime) / 1000);
  }

  public static int getMinuteOfDay(int packedLocalTime) {
    return getHour(packedLocalTime) * 60 + getMinute(packedLocalTime);
  }

  public static int getSecondOfDay(int packedLocalTime) {
    int total = getHour(packedLocalTime) * 60 * 60;
    total += getMinute(packedLocalTime) * 60;
    total += getSecond(packedLocalTime);
    return total;
  }

  public static int getMillisecondOfDay(int packedLocalTime) {
    return (int) (toNanoOfDay(packedLocalTime) / 1000_000);
  }

  public static String toShortTimeString(int time) {
    if (time == -1) {
      return "NA";
    }

    byte hourByte = (byte) (time >> 24);
    byte minuteByte = (byte) (time >> 16);
    byte millisecondByte1 = (byte) (time >> 8);
    byte millisecondByte2 = (byte) time;
    char millis = (char) ((millisecondByte1 << 8) | (millisecondByte2 & 0xFF));
    int second = millis / 1000;
    int millisOnly = millis % 1000;

    return String.format("%s:%s:%s",
        Strings.padStart(Byte.toString(hourByte), 2, '0'),
        Strings.padStart(Byte.toString(minuteByte), 2, '0'),
        Strings.padStart(Integer.toString(second), 2, '0'));
  }

  public static boolean isMidnight(int packedTime) {
    return packedTime == MIDNIGHT;
  }

  public static boolean isNoon(int packedTime) {
    return packedTime == NOON;
  }

  public static boolean isAfter(int packedTime, int value) {
    return packedTime > value;
  }

  public static boolean isBefore(int packedTime, int value) {
    return packedTime < value;
  }

  public static boolean isEqualTo(int packedTime, int value) {
    return packedTime == value;
  }

  /**
   * Returns true if the time is in the AM or "before noon".
   * Note: we follow the convention that 12:00 NOON is PM and 12 MIDNIGHT is AM
   */
  public static boolean AM(int packedTime) {
    return packedTime < NOON;
  }

  /**
   * Returns true if the time is in the PM or "after noon".
   * Note: we follow the convention that 12:00 NOON is PM and 12 MIDNIGHT is AM
   */
  public static boolean PM(int packedTime) {
    return packedTime >= NOON;
  }
}
