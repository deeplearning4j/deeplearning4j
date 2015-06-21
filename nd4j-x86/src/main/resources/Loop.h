#include <cmath>
#include <string>
#include <jni.h>
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


    void execFloatTransform(float *data, int length, int offset, int stride,int resultStride, const std::string operation,
                            float *otherParams,float *result) {
        if(operation.compare("tanh") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = tanhf(data[i * stride]);
            }
        }
        else if(operation.compare("exp") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = expf(data[i * stride]);
            }
        }
        else if(operation.compare("cos") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = cosf(data[i * stride]);
            }
        }
        else if(operation.compare("abs") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = fabs(data[i * stride]);
            }
        }
        else if(operation.compare("acos") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = acosf(data[i * stride]);
            }
        }
        else if(operation.compare("asin") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = asin(data[i * stride]);
            }
        }

        else if(operation.compare("atan") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = atan(data[i * stride]);
            }
        }
        else if(operation.compare("ceil") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = ceil(data[i * stride]);
            }
        }
        else if(operation.compare("floor") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = floor(data[i * stride]);
            }
        }
        else if(operation.compare("hardtanh") == 0) {
            for(int i = offset; i < length; i++) {
                float tanh2 = tanhf(data[i * stride]);
                if(tanh2 < -1)
                    tanh2 = -1;
                if(tanh2 > 1)
                    tanh2 = 1;
                result[i * resultStride] = tanh2;
            }
        }
        else if(operation.compare("log") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = logf(data[i * stride]);
            }
        }
        else if(operation.compare("neg") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = -data[i * stride];
            }
        }
        else if(operation.compare("oneminus") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] =  1 - data[i * stride];
            }
        }
        else if(operation.compare("ones") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] =  1;
            }
        }
        else if(operation.compare("pow") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] =  powf(data[i * stride],(float) otherParams[0]);
            }
        }
        else if(operation.compare("sigmoid") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] =  1.0 / (1.0 + expf(-data[i * stride]));
            }
        }
        else if(operation.compare("sign") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = (d1 > 0) - (d1 < 0);
            }
        }
        else if(operation.compare("round") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = roundf(d1);
            }
        }
        else if(operation.compare("softmax") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = roundf(d1);
            }
        }

        else if(operation.compare("sqrt") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = sqrtf(d1);
            }
        }

    }



    void execScalarDouble(
            double *data
            ,double *result
            ,int length
            ,int offset
            ,int stride
            ,int resultStride
            ,const std::string operation
            ,double *otherParams) {
        double scalar = otherParams[0];
        if(operation.compare("equals_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                double d1 = data[i * stride];
                result[i * resultStride] = d1 == scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthan_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                double d1 = data[i * stride];
                result[i * resultStride] = d1 >= scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthanorequal_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                double d1 = data[i * stride];
                result[i * resultStride] = d1 == scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("lessthan_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                double d1 = data[i * stride];
                result[i * resultStride] = d1 < scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("lessthanorequal_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                double d1 = data[i * stride];
                result[i * resultStride] = d1 <= scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthan_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                double d1 = data[i * stride];
                result[i * resultStride] = d1 > scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("add_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = data[i * stride] + scalar;

        }
        else if(operation.compare("div_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = data[i * stride] / scalar;

        }
        else if(operation.compare("max_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = max(data[i * stride],scalar);

        }
        else if(operation.compare("mul_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = data[i * stride] * scalar;

        }
        else if(operation.compare("rdiv_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = scalar / data[i * stride];

        }
        else if(operation.compare("rsub_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = scalar - data[i * stride];

        }

        else if(operation.compare("sub_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = data[i * stride] - scalar;

        }

    }

    void execScalarFloat(
            float *data
            ,float *result
            ,int length
            ,int offset
            ,int stride
            ,int resultStride
            ,const std::string operation
            ,float *otherParams) {
        float scalar = otherParams[0];
        if(operation.compare("equals_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = d1 == scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthan_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = d1 >= scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthanorequal_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = d1 == scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("lessthan_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = d1 < scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("lessthanorequal_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = d1 <= scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("greaterthan_scalar") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = d1 > scalar ? 1 : 0.0;
            }
        }
        else if(operation.compare("add_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = data[i * stride] + scalar;

        }
        else if(operation.compare("div_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = data[i * stride] / scalar;

        }
        else if(operation.compare("max_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = max(data[i * stride],scalar);

        }
        else if(operation.compare("mul_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = data[i * stride] * scalar;

        }
        else if(operation.compare("rdiv_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = scalar / data[i * stride];

        }
        else if(operation.compare("rsub_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = scalar - data[i * stride];

        }

        else if(operation.compare("sub_scalar") == 0) {
            for(int i = offset; i < length; i++)
                result[i * resultStride] = data[i * stride] - scalar;

        }
    }

    void execDoubleTransform(double *data, int length, int offset, int stride,int resultStride, const std::string operation,
                             double *otherParams,double *result) {
        if(operation.compare("tanh") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = tanh(data[i * stride]);
            }
        }
        else if(operation.compare("exp") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = exp(data[i * stride]);
            }
        }
        else if(operation.compare("cos") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = cos(data[i * stride]);
            }
        }
        else if(operation.compare("abs") == 0) {
            for(int i = offset; i < length; i++) {
                double d = data[i * stride];
                result[i * resultStride] = abs(d);
            }
        }
        else if(operation.compare("acos") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = acos(data[i * stride]);
            }
        }
        else if(operation.compare("asin") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = asinf(data[i * stride]);
            }
        }
        else if(operation.compare("asin") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = atan(data[i * stride]);
            }
        }
        else if(operation.compare("ceil") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = floorf(data[i * stride]);
            }
        }
        else if(operation.compare("hardtanh") == 0) {
            for(int i = offset; i < length; i++) {
                double tanh2 = tanh(data[i * stride]);
                if(tanh2 < -1)
                    tanh2 = -1;
                if(tanh2 > 1)
                    tanh2 = 1;
                result[i * resultStride] = tanh2;
            }
        }
        else if(operation.compare("log") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = log(data[i * stride]);
            }
        }
        else if(operation.compare("neg") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] = -data[i * stride];
            }
        }
        else if(operation.compare("oneminus") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] =  1 - data[i * stride];
            }
        }
        else if(operation.compare("ones") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] =  1;
            }
        }

        else if(operation.compare("pow") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] =  pow(data[i * stride],(double) otherParams[0]);
            }
        }
        else if(operation.compare("sigmoid") == 0) {
            for(int i = offset; i < length; i++) {
                result[i * resultStride] =  1.0 / (1.0 + exp(-data[i * stride]));
            }
        }
        else if(operation.compare("sign") == 0) {
            for(int i = offset; i < length; i++) {
                double d1 = data[i * stride];
                result[i * resultStride] = (d1 > 0) - (d1 < 0);
            }
        }
        else if(operation.compare("round") == 0) {
            for(int i = offset; i < length; i++) {
                double d1 = data[i * stride];
                result[i * resultStride] = round(d1);
            }
        }

        else if(operation.compare("softmax") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = roundf(d1);
            }
        }
        else if(operation.compare("sqrt") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                result[i * resultStride] = sqrt(d1);
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
                startingValue += (data[(i + xOffset) * xStride] * data2[(i + yOffset) * yStride]);
            }
            startingValue /=  constantNormalizedByNorm2X / constantNormalizedByNorm2Y;

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
            for(int i = offset; i < length; i++) {
                startingValue += data[i * stride];
            }

        }
        else if(operation.compare("prod") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue *= data[i * stride];
            }

        }
        else if(operation.compare("mean") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue += data[i * stride];
            }

            startingValue /= (double) length;
        }
        else if(operation.compare("max") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue = max(data[i * stride],startingValue);
            }
        }
        else if(operation.compare("bias") == 0) {
            double mean = otherParams[0];
            for(int i = offset; i < length; i++) {
                startingValue += data[i * stride] - mean;
            }
        }
        else if(operation.compare("var") == 0) {
            double bias = otherParams[1];
            double mean = otherParams[2];
            for(int i = offset; i < length; i++) {
                startingValue += powf(data[i * stride] - mean,2.0);
            }
            startingValue = (startingValue - (pow(bias,2.0) / length)) / (double) (length - 1.0);
        }
        else if(operation.compare("std") == 0) {
            double bias = otherParams[1];
            double mean = otherParams[2];
            for(int i = offset; i < length; i++) {
                startingValue += powf(data[i * stride] - mean,2.0);
            }
            startingValue = sqrt((startingValue - (pow(bias,2.0) / length)) / (double) (length - 1.0));
        }
        else if(operation.compare("min") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue = min(data[i * stride],startingValue);
            }
        }
        else if(operation.compare("norm1") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue += abs(data[i * stride]);
            }
        }
        else if(operation.compare("norm2") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue += pow(data[i * stride],2);
            }
        }
        else if(operation.compare("normmax") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue = max(abs(startingValue),abs(data[i * stride]));
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
                startingValue += (data[(i + xOffset) * xStride] * data2[(i + yOffset) * yStride]);
            }
            startingValue /=  constantNormalizedByNorm2X / constantNormalizedByNorm2Y;

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
            for(int i = offset; i < length; i++) {
                startingValue += data[i * stride];
            }

        }
        else if(operation.compare("prod") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue *= data[i * stride];
            }

        }
        else if(operation.compare("mean") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue += data[i * stride];
            }

            startingValue /= (float) length;
        }
        else if(operation.compare("max") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue = fmaxf(data[i * stride],startingValue);
            }
        }
        else if(operation.compare("bias") == 0) {
            float mean = otherParams[0];
            for(int i = offset; i < length; i++) {
                startingValue += data[i * stride] - mean;
            }
        }
        else if(operation.compare("var") == 0) {
            float bias = otherParams[1];
            float mean = otherParams[2];
            for(int i = offset; i < length; i++) {
                startingValue += powf(data[i * stride] - mean,2.0);
            }
            startingValue = (startingValue - (powf(bias,2.0) / length)) / (float) (length - 1.0);
        }
        else if(operation.compare("std") == 0) {
            float bias = otherParams[1];
            float mean = otherParams[2];
            for(int i = offset; i < length; i++) {
                startingValue += powf(data[i * stride] - mean,2.0);
            }
            startingValue = sqrtf((startingValue - (powf(bias,2.0) / length)) / (float) (length - 1.0));
        }
        else if(operation.compare("min") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue = fminf(data[i * stride],startingValue);
            }
        }
        else if(operation.compare("norm1") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue += fabsf(data[i * stride]);
            }
        }
        else if(operation.compare("norm2") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue += powf(data[i * stride],2);
            }
        }
        else if(operation.compare("normmax") == 0) {
            for(int i = offset; i < length; i++) {
                startingValue = fmaxf(abs(startingValue),abs(data[i * stride]));
            }
        }


        return  startingValue;
    }






};