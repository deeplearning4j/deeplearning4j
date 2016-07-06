package musicg.dsp;

import org.junit.Assert;
import org.junit.Test;

public class TestFastFourierTransform {

  @Test
  public void testFastFourierTransform() {
    FastFourierTransform fft = new FastFourierTransform();
    double[] amplitudes = new double[]{3.0, 4.0, 0.5, 7.8, 6.9, -6.5, 8.5, 4.6};
    double[] frequencies = fft.getMagnitudes(amplitudes);

    Assert.assertEquals(2, frequencies.length);
    Assert.assertArrayEquals(new double[]{21.335,18.513}, frequencies, 0.005);
  }
}
