#include <iostream>
#include <iomanip>
#include <math.h>
using namespace std;

class Loop {


 public:

    void exec(double *data,int length,int offset,int stride,std::string operation) {
          if(operation == "tanh") {
          for(int i = offset; i < length; i++) {
               data[i * stride] = tanh(data[i * stride]);
          }
       }
       else if(operation == "exp") {
           for(int i = offset; i < length; i++) {
                          data[i * stride] = exp(data[i * stride]);
                     }
       }
       else if(operation == "cos") {
           for(int i = offset; i < length; i++) {
                          data[i * stride] = cos(data[i * stride]);
                     }
       }



    }


}