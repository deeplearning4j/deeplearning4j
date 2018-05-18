//
// @author raver119@gmail.com
//

#include <helpers/helper_hash.h>
#include <helpers/logger.h>

namespace nd4j {
    namespace ops {

        HashHelper* HashHelper::getInstance() {
            if (_INSTANCE == 0)
                _INSTANCE = new HashHelper();

            return _INSTANCE;
        }

        Nd4jLong HashHelper::getLongHash(std::string& str) {
            _locker.lock();
            if (!_isInit) {
                nd4j_verbose("Building HashUtil table\n","");

                Nd4jLong h = 0x544B2FBACAAF1684L;
                for (int i = 0; i < 256; i++) {
                    for (int j = 0; j < 31; j++) {
                        h = (((unsigned long long) h) >> 7) ^ h;
                        h = (h << 11) ^ h;
                        h = (((unsigned long long) h) >> 10) ^ h;
                    }
                    _byteTable[i] = h;
                }


                _isInit = true;
            }

            _locker.unlock();

            Nd4jLong h = HSTART;
            Nd4jLong hmult = HMULT;
            Nd4jLong len = str.size();
            for (int i = 0; i < len; i++) {
                char ch = str.at(i);
                auto uch = (unsigned char) ch;
                h = (h * hmult) ^ _byteTable[ch & 0xff];
                h = (h * hmult) ^ _byteTable[(uch >> 8) & 0xff];
            }

            return h;
        }

        nd4j::ops::HashHelper* nd4j::ops::HashHelper::_INSTANCE = 0;
    }
}

