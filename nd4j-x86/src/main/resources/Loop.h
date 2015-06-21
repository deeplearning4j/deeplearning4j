#include <cmath>
#include <jni.h>
#include <string>
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


    void execFloatTransform(float *data, int length, int offset, int stride, const std::string operation,
                            float *otherParams) {
        if(operation.compare("tanh") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = tanhf(data[i * stride]);
            }
        }
        else if(operation.compare("exp") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = expf(data[i * stride]);
            }
        }
        else if(operation.compare("cos") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = cosf(data[i * stride]);
            }
        }
        else if(operation.compare("abs") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = fabs(data[i * stride]);
            }
        }
        else if(operation.compare("acos") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = acosf(data[i * stride]);
            }
        }
        else if(operation.compare("asin") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = asin(data[i * stride]);
            }
        }

        else if(operation.compare("atan") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = atan(data[i * stride]);
            }
        }
        else if(operation.compare("ceil") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = ceil(data[i * stride]);
            }
        }
        else if(operation.compare("floor") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = floor(data[i * stride]);
            }
        }
        else if(operation.compare("hardtanh") == 0) {
            for(int i = offset; i < length; i++) {
                float tanh2 = tanhf(data[i * stride]);
                if(tanh2 < -1)
                    tanh2 = -1;
                if(tanh2 > 1)
                    tanh2 = 1;
                data[i * stride] = tanh2;
            }
        }
        else if(operation.compare("log") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = logf(data[i * stride]);
            }
        }
        else if(operation.compare("neg") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = -data[i * stride];
            }
        }
        else if(operation.compare("oneminus") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] =  1 - data[i * stride];
            }
        }
        else if(operation.compare("ones") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] =  1;
            }
        }
        else if(operation.compare("pow") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] =  powf(data[i * stride],(float) otherParams[0]);
            }
        }
        else if(operation.compare("sigmoid") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] =  1.0 / (1.0 + expf(-data[i * stride]));
            }
        }
        else if(operation.compare("sign") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                data[i * stride] = (d1 > 0) - (d1 < 0);
            }
        }
        else if(operation.compare("round") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                data[i * stride] = roundf(d1);
            }
        }
        else if(operation.compare("softmax") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                data[i * stride] = roundf(d1);
            }
        }

        else if(operation.compare("sqrt") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                data[i * stride] = sqrtf(d1);
            }
        }

    }

    void execDoubleTransform(double *data, int length, int offset, int stride, const std::string operation,
                             double *otherParams) {
        if(operation.compare("tanh") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = tanh(data[i * stride]);
            }
        }
        else if(operation.compare("exp") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = exp(data[i * stride]);
            }
        }
        else if(operation.compare("cos") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = cos(data[i * stride]);
            }
        }
        else if(operation.compare("abs") == 0) {
            for(int i = offset; i < length; i++) {
                double d = data[i * stride];
                data[i * stride] = abs(d);
            }
        }
        else if(operation.compare("acos") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = acos(data[i * stride]);
            }
        }
        else if(operation.compare("asin") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = asinf(data[i * stride]);
            }
        }
        else if(operation.compare("asin") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = atan(data[i * stride]);
            }
        }
        else if(operation.compare("ceil") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = floorf(data[i * stride]);
            }
        }
        else if(operation.compare("hardtanh") == 0) {
            for(int i = offset; i < length; i++) {
                double tanh2 = tanh(data[i * stride]);
                if(tanh2 < -1)
                    tanh2 = -1;
                if(tanh2 > 1)
                    tanh2 = 1;
                data[i * stride] = tanh2;
            }
        }
        else if(operation.compare("log") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = log(data[i * stride]);
            }
        }
        else if(operation.compare("neg") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] = -data[i * stride];
            }
        }
        else if(operation.compare("oneminus") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] =  1 - data[i * stride];
            }
        }
        else if(operation.compare("ones") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] =  1;
            }
        }

        else if(operation.compare("pow") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] =  pow(data[i * stride],(double) otherParams[0]);
            }
        }
        else if(operation.compare("sigmoid") == 0) {
            for(int i = offset; i < length; i++) {
                data[i * stride] =  1.0 / (1.0 + exp(-data[i * stride]));
            }
        }
        else if(operation.compare("sign") == 0) {
            for(int i = offset; i < length; i++) {
                double d1 = data[i * stride];
                data[i * stride] = (d1 > 0) - (d1 < 0);
            }
        }
        else if(operation.compare("round") == 0) {
            for(int i = offset; i < length; i++) {
                double d1 = data[i * stride];
                data[i * stride] = round(d1);
            }
        }

        else if(operation.compare("softmax") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                data[i * stride] = roundf(d1);
            }
        }
        else if(operation.compare("sqrt") == 0) {
            for(int i = offset; i < length; i++) {
                float d1 = data[i * stride];
                data[i * stride] = sqrt(d1);
            }
        }
    }


};