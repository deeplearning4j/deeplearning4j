#include <cmath>
#include <string>
#include <jni.h>
#include <algorithm>
using namespace std;
/**
 * CPU math operations for:
 * linear transforms
 * reductions
 *
 * @author Adam Gibson
 */
class Loop {


public:


    void execDoubleTransform(
            double *data,
            double *pairData
            , int length
            , int offset,
            int yOffset,
            int resultOffset
            , int stride,
            int yStride
            ,int resultStride
            , const std::string  operation,
            double *otherParams
            , double *result) {
        if(operation.compare("add") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = data[offset + (i * stride)] + pairData[yOffset + (i * yStride)];
            }
        }
        else if(operation.compare("sub") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i * stride)] - pairData[yOffset + (i * yStride)];

        }
        else if(operation.compare("rsub") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = pairData[yOffset + (i * yStride)] - data[offset + (i * stride)];

        }
        else if(operation.compare("mul") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i * stride)] * pairData[yOffset + (i * yStride)];

        }
        else if(operation.compare("div") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i * stride)] / pairData[yOffset + (i * yStride)];

        }
        else if(operation.compare("rdiv") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] =  pairData[yOffset + (i * yStride)] / data[offset + (i * stride)];

        }
        else if(operation.compare("copy") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i * stride)];

        }
    }


    void execFloatTransform(
            float *data,
            float *pairData
            , int length
            , int offset,
            int yOffset,
            int resultOffset
            , int stride,
            int yStride
            ,int resultStride
            , const std::string  operation,
            float *otherParams
            , float *result) {
        if(operation.compare("add") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = data[offset + (i * stride)] + pairData[yOffset + (i * yStride)];
            }
        }
        else if(operation.compare("sub") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i * stride)] - pairData[yOffset + (i * yStride)];

        }
        else if(operation.compare("rsub") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = pairData[yOffset + (i * yStride)] - data[offset + (i * stride)];

        }
        else if(operation.compare("mul") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i * stride)] * pairData[yOffset + (i * yStride)];

        }
        else if(operation.compare("div") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i * stride)] / pairData[yOffset + (i * yStride)];

        }
        else if(operation.compare("rdiv") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] =  pairData[yOffset + (i * yStride)] / data[offset + (i * stride)];

        }
        else if(operation.compare("copy") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i * stride)];

        }
    }

    void execFloatTransform(float *data, int length, int offset, int resultOffset,int stride,int resultStride, const std::string operation,
                            float *otherParams,float *result) {
        if(operation.compare("tanh") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = tanhf(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("exp") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = expf(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("cos") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = cosf(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("abs") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = fabs(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("acos") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = acosf(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("asin") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = asin(data[offset + (i  * stride)]);
            }
        }

        else if(operation.compare("atan") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = atan(data[offset + (i  * stride)]);
            }
        }
          else if(operation.compare("softplus") == 0) {
                    for(int i = 0; i < length; i++) {
                        result[resultOffset + (i * resultStride)] = log( 1 + exp(data[offset + (i  * stride)]));
                    }
         }
        else if(operation.compare("setrange") == 0) {
            float min = otherParams[0];
            float max = otherParams[1];
            for(int i = 0; i < length; i++) {
                float origin = data[offset + (i * stride)];
                if (origin >= min && origin <= max)
                    result[resultOffset + (i * resultStride)] = origin;
                else if (min == 0 && max == 1) {
                    float val = 1 / (1 + expf(-origin));
                    result[resultOffset + (i * resultStride)] =  floorf(val * (max - min)) + min;
                }
                else {
                    result[resultOffset + (i * resultStride)] = floorf(origin * (max - min)) + min;

                }
            }
        }

        else if(operation.compare("ceil") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = ceilf(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("floor") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = floorf(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("hardtanh") == 0) {
            for(int i = 0; i < length; i++) {
                float tanh2 = tanhf(data[offset + (i  * stride)]);
                if(tanh2 < -1)
                    tanh2 = -1;
                if(tanh2 > 1)
                    tanh2 = 1;
                result[resultOffset + (i * resultStride)] = tanh2;
            }
        }
        else if(operation.compare("log") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = logf(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("neg") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = -data[offset + (i  * stride)];
            }
        }
        else if(operation.compare("oneminus") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] =  1 - data[offset + (i  * stride)];
            }
        }
        else if(operation.compare("ones") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] =  1;
            }
        }
        else if(operation.compare("pow") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] =  powf(data[offset + (i  * stride)],(float) otherParams[0]);
            }
        }
        else if(operation.compare("sigmoid") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] =  1.0 / (1.0 + expf(-data[offset + (i  * stride)]));
            }
        }
        else if(operation.compare("sign") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = (d1 > 0) - (d1 < 0);
            }
        }
        else if(operation.compare("round") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = roundf(d1);
            }
        }
        else if(operation.compare("softmax") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = roundf(d1);
            }
        }

        else if(operation.compare("sqrt") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = sqrtf(d1);
            }
        }

    }



    void execScalarDouble(
            double *data
            ,double *result
            ,int length
            ,int offset,
            int resultOffset
            ,int stride
            ,int resultStride
            ,const std::string operation
            ,double *otherParams) {
        double scalar = otherParams[0];
        if(operation.compare("equals_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                double d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 == scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthan_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                double d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 >= scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthanorequal_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                double d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 == scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("lessthan_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                double d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 < scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("lessthanorequal_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                double d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 <= scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthan_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                double d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 > scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("add_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i  * stride)] + scalar;

        }
        else if(operation.compare("div_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i  * stride)] / scalar;

        }
        else if(operation.compare("max_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = max(data[offset + (i  * stride)],scalar);

        }
        else if(operation.compare("mul_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i  * stride)] * scalar;

        }
        else if(operation.compare("rdiv_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = scalar / data[offset + (i  * stride)];

        }
        else if(operation.compare("rsub_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = scalar - data[offset + (i  * stride)];

        }

        else if(operation.compare("sub_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i  * stride)] - scalar;

        }

    }

    void execScalarFloat(
            float *data
            ,float *result
            ,int length
            ,int offset,
            int resultOffset
            ,int stride
            ,int resultStride
            ,const std::string operation
            ,float *otherParams) {
        float scalar = otherParams[0];
        if(operation.compare("equals_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 == scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthan_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 >= scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthanorequal_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 == scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("lessthan_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 < scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("lessthanorequal_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 <= scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthan_scalar") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = d1 > scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("add_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i  * stride)] + scalar;

        }
        else if(operation.compare("div_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i  * stride)] / scalar;

        }
        else if(operation.compare("max_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = max(data[offset + (i  * stride)],scalar);

        }
        else if(operation.compare("mul_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i  * stride)] * scalar;

        }
        else if(operation.compare("rdiv_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = scalar / data[offset + (i  * stride)];

        }
        else if(operation.compare("rsub_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = scalar - data[offset + (i  * stride)];

        }

        else if(operation.compare("sub_scalar") == 0) {
            for(int i = 0; i < length; i++)
                result[resultOffset + (i * resultStride)] = data[offset + (i  * stride)] - scalar;

        }
    }

    void execDoubleTransform(double *data, int length, int offset,int resultOffset, int stride,int resultStride, const std::string operation,
                             double *otherParams,double *result) {
        if(operation.compare("tanh") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = tanh(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("exp") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = exp(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("cos") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = cos(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("abs") == 0) {
            for(int i = 0; i < length; i++) {
                double d = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = abs(d);
            }
        }
        else if(operation.compare("acos") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = acos(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("setrange") == 0) {
            double min = otherParams[0];
            double max = otherParams[1];
            for(int i = 0; i < length; i++) {
                double origin = data[offset + (i * stride)];
                if (origin >= min && origin <= max)
                    result[resultOffset + (i * resultStride)] = origin;
                else if (min == 0 && max == 1) {
                    double val = 1 / (1 + exp(-origin));
                    result[resultOffset + (i * resultStride)] =  floor(val * (max - min)) + min;
                }
                else {
                    result[resultOffset + (i * resultStride)] = floor(origin * (max - min)) + min;

                }
            }
        }
        else if(operation.compare("asin") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = asinf(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("asin") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = atan(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("ceil") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = floorf(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("hardtanh") == 0) {
            for(int i = 0; i < length; i++) {
                double tanh2 = tanh(data[offset + (i  * stride)]);
                if(tanh2 < -1)
                    tanh2 = -1;
                if(tanh2 > 1)
                    tanh2 = 1;
                result[resultOffset + (i * resultStride)] = tanh2;
            }
        }
        else if(operation.compare("log") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = log(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("neg") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] = -data[offset + (i  * stride)];
            }
        }
        else if(operation.compare("oneminus") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] =  1 - data[offset + (i  * stride)];
            }
        }
        else if(operation.compare("ones") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] =  1;
            }
        }

        else if(operation.compare("pow") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] =  pow(data[offset + (i  * stride)],(double) otherParams[0]);
            }
        }
        else if(operation.compare("sigmoid") == 0) {
            for(int i = 0; i < length; i++) {
                result[resultOffset + (i * resultStride)] =  1.0 / (1.0 + exp(-data[offset + (i  * stride)]));
            }
        }
        else if(operation.compare("sign") == 0) {
            for(int i = 0; i < length; i++) {
                double d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = (d1 > 0) - (d1 < 0);
            }
        }
        else if(operation.compare("round") == 0) {
            for(int i = 0; i < length; i++) {
                double d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = round(d1);
            }
        }

        else if(operation.compare("softmax") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = roundf(d1);
            }
        }
        else if(operation.compare("sqrt") == 0) {
            for(int i = 0; i < length; i++) {
                float d1 = data[offset + (i  * stride)];
                result[resultOffset + (i * resultStride)] = sqrt(d1);
            }
        }

           else if(operation.compare("softplus") == 0) {
                for(int i = 0; i < length; i++) {
                     result[resultOffset + (i * resultStride)] = logf( 1 + expf(data[offset + (i  * stride)]));
                }
          }
    }

    double reduce3(double *data, double *data2,int length, int xOffset, int yOffset,int xStride,int yStride, const std::string operation,
                   double *otherParams) {
        double startingValue = otherParams[0];

        if(operation.compare("cosinesimilarity") == 0) {
            double constantNormalizedByNorm2X = otherParams[1];
            double constantNormalizedByNorm2Y = otherParams[2];
            for(int i = 0; i < length; i++) {
                double val1 = data[(i + xOffset) * xStride];
                double val2 = data2[(i + yOffset) * yStride];
                startingValue += (val1 * val2);
            }
            startingValue =  startingValue / constantNormalizedByNorm2X / constantNormalizedByNorm2Y;
            return startingValue;
        }
        else if(operation.compare("euclidean") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += pow(data[(i + xOffset) * xStride] - data2[(i + yOffset) * yStride],2);
            }
            startingValue /= sqrt(startingValue);

        }
        else if(operation.compare("manhattan") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += (data[(i + xOffset) * xStride] - data2[(i + yOffset) * yStride]);
            }
        }

        return startingValue;
    }

    double reduce(double *data, int length, int offset, int stride, const std::string operation,
                  double *otherParams) {
        double startingValue = otherParams[0];
        if(operation.compare("sum") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += data[offset + (i  * stride)];
            }

        }
        else if(operation.compare("prod") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue *= data[offset + (i  * stride)];
            }

        }
        else if(operation.compare("mean") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += data[offset + (i  * stride)];
            }

            startingValue /= (double) length;
        }
        else if(operation.compare("max") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue = max(data[offset + (i  * stride)],startingValue);
            }
        }
        else if(operation.compare("bias") == 0) {
            double mean = otherParams[1];
            for(int i = 0; i < length; i++) {
                double val = data[offset + (i  * stride)];
                double subMean = val - mean;
                startingValue += subMean;
            }
        }
        else if(operation.compare("var") == 0) {
            double bias = otherParams[1];
            double mean = otherParams[2];
            for(int i = 0; i < length; i++) {
                startingValue += powf(data[offset + (i  * stride)] - mean,2.0);
            }
            startingValue = (startingValue - (pow(bias,2.0) / length)) / (double) (length - 1.0);
        }
        else if(operation.compare("std") == 0) {
            double bias = otherParams[1];
            double mean = otherParams[2];
            for(int i = 0; i < length; i++) {
                startingValue += powf(data[offset + (i  * stride)] - mean,2.0);
            }
            startingValue = sqrt((startingValue - (pow(bias,2.0) / length)) / (double) (length - 1.0));
        }
        else if(operation.compare("min") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue = min(data[offset + (i  * stride)],startingValue);
            }
        }
        else if(operation.compare("norm1") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += abs(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("norm2") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += pow(data[offset + (i  * stride)],2);
            }

            startingValue = sqrt(startingValue);
        }
        else if(operation.compare("normmax") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue = max(abs(startingValue),abs(data[offset + (i  * stride)]));
            }
        }


        return  startingValue;
    }

    float reduce3Float(float *data, float *data2,int length, int xOffset, int yOffset,int xStride,int yStride, const std::string operation,
                       float *otherParams) {
        float startingValue = otherParams[0];


        if(operation.compare("cosinesimilarity") == 0) {
            float constantNormalizedByNorm2X = otherParams[1];
            float constantNormalizedByNorm2Y = otherParams[2];
            for(int i = 0; i < length; i++) {
                float val1 = data[(i + xOffset) * xStride];
                float val2 = data2[(i + yOffset) * yStride];
                startingValue += (val1 * val2);
            }
            startingValue =  startingValue / constantNormalizedByNorm2X / constantNormalizedByNorm2Y;
        }
        else if(operation.compare("euclidean") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += powf(data[(i + xOffset) * xStride] - data2[(i + yOffset) * yStride],2);
            }
            startingValue /= sqrtf(startingValue);

        }
        else if(operation.compare("manhattan") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += (data[(i + xOffset) * xStride] - data2[(i + yOffset) * yStride]);
            }
        }

        return startingValue;
    }

    float reduceFloat(float *data, int length, int offset, int stride, const std::string operation,
                      float *otherParams) {
        float startingValue = otherParams[0];
        if(operation.compare("sum") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += data[offset + (i  * stride)];
            }

        }
        else if(operation.compare("prod") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue *= data[offset + (i  * stride)];
            }

        }
        else if(operation.compare("mean") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += data[offset + (i  * stride)];
            }

            startingValue /= (float) length;
        }
        else if(operation.compare("max") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue = fmaxf(data[offset + (i  * stride)],startingValue);
            }
        }
        else if(operation.compare("bias") == 0) {
            float mean = otherParams[1];
            for(int i = 0; i < length; i++) {
                float val = data[offset + (i  * stride)];
                float subMean = val - mean;
                startingValue += subMean;
            }
        }
        else if(operation.compare("var") == 0) {
            float bias = otherParams[1];
            float mean = otherParams[2];
            for(int i = 0; i < length; i++) {
                startingValue += powf(data[offset + (i  * stride)] - mean,2.0);
            }
            startingValue = (startingValue - (powf(bias,2.0) / length)) / (float) (length - 1.0);
        }
        else if(operation.compare("std") == 0) {
            float bias = otherParams[1];
            float mean = otherParams[2];
            for(int i = 0; i < length; i++) {
                startingValue += powf(data[offset + (i  * stride)] - mean,2.0);
            }
            startingValue = sqrtf((startingValue - (powf(bias,2.0) / length)) / (float) (length - 1.0));
        }
        else if(operation.compare("min") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue = fminf(data[offset + (i  * stride)],startingValue);
            }
        }
        else if(operation.compare("norm1") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += fabsf(data[offset + (i  * stride)]);
            }
        }
        else if(operation.compare("norm2") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue += powf(data[offset + (i  * stride)],2);
            }

            startingValue = sqrtf(startingValue);

        }
        else if(operation.compare("normmax") == 0) {
            for(int i = 0; i < length; i++) {
                startingValue = fmaxf(abs(startingValue),abs(data[offset + (i  * stride)]));
            }
        }


        return  startingValue;
    }






};