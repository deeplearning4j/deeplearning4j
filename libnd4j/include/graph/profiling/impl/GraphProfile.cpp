//
//  @author raver119@gmail.com
//

#include <graph/profiling/GraphProfile.h>
#include <helpers/logger.h>
#include <chrono>

namespace nd4j {
    namespace graph {
        GraphProfile::GraphProfile() {
            updateLast();
        }

        GraphProfile::~GraphProfile() {
            // releasing NodeProfile pointers
            for (auto v: _profiles)
                delete v;

            _timings.clear();
        }

        void GraphProfile::addToTotal(Nd4jLong bytes) {
            _memoryTotal += bytes;
        }

        void GraphProfile::addToActivations(Nd4jLong bytes) {
            _memoryActivations += bytes;
        }
        
        void GraphProfile::addToTemporary(Nd4jLong bytes) {
            _memoryTemporary += bytes;
        }
        
        void GraphProfile::addToObjects(Nd4jLong bytes) {
            _memoryObjects += bytes;
        }

        void GraphProfile::setBuildTime(Nd4jLong nanos) {
            _buildTime = nanos;
        }

        void GraphProfile::setExecutionTime(Nd4jLong nanos) {
            _executionTime = nanos;
        }


        Nd4jLong GraphProfile::currentTime() {
            auto t = std::chrono::system_clock::now();
            auto v = std::chrono::time_point_cast<std::chrono::nanoseconds> (t);
            auto epoch = v.time_since_epoch();
            return (Nd4jLong) std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();
        }
        
        Nd4jLong GraphProfile::relativeTime(Nd4jLong time) {
            auto t1 = currentTime();
            return t1 - time;
        }

        void GraphProfile::updateLast() {
            _last = std::chrono::system_clock::now();
        }

        void GraphProfile::startEvent(const char *name) {
            std::string k = name;
            _timers[k] = std::chrono::system_clock::now();
        }

        void GraphProfile::recordEvent(const char *name) {
            std::string k = name;
            if (_timers.count(k) == 0) {
                nd4j_printf("Can't find timer key: [%s]", name);
                throw "Missing timer key";
            }
            auto t0 = _timers[k];
            auto t1 = std::chrono::system_clock::now();
            auto v = (Nd4jLong) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

            _timings[k] = v;
            _timers.erase(k);
        }
        
        void GraphProfile::deleteEvent(const char *name) {
            std::string k = name;
            _timers.erase(k);
        }
            
        void GraphProfile::spotEvent(const char *name) {
            auto t = std::chrono::system_clock::now();
            auto d = (Nd4jLong) std::chrono::duration_cast<std::chrono::nanoseconds>(t - _last).count();
            std::string k = name;
            _timings[k] = d;
            updateLast();
        }

        NodeProfile* GraphProfile::nodeById(int id, const char *name) {
            if (_profilesById.count(id) == 0) {
                auto node = new NodeProfile(id, name);
                _profiles.emplace_back(node);
                _profilesById[id] = node;
                return node;
            }

            return _profilesById[id];
        }

        void GraphProfile::merge(GraphProfile *other) {
            _merges += other->_merges;
            _memoryActivations += other->_memoryActivations;
            _memoryTemporary += other->_memoryTemporary;
            _memoryTotal += other->_memoryTotal;
            _memoryObjects += other->_memoryObjects;

            _executionTime += other->_executionTime;
            _buildTime += other->_buildTime;


            for (auto v:_profilesById) {
                if (!other->nodeExists(v.first))
                    continue;

                v.second->merge(other->nodeById(v.first));
            }
        }

        void GraphProfile::assign(GraphProfile *other) {
            _merges = other->_merges;
            _memoryActivations = other->_memoryActivations;
            _memoryTemporary = other->_memoryTemporary;
            _memoryTotal = other->_memoryTotal;
            _memoryObjects = other->_memoryObjects;

            _executionTime = other->_executionTime;
            _buildTime = other->_buildTime;


            for (auto v: other->_profilesById) {
                nodeById(v.first, v.second->name().c_str())->assign(v.second);
            }
        }

        bool GraphProfile::nodeExists(int id) {
            return _profilesById.count(id) > 0;
        }

        void GraphProfile::printOut() {
            nd4j_printf("Graph profile: %i executions\n", _merges);
            nd4j_printf("\nMemory:\n", "");

            Nd4jLong tmp = 0L;
            Nd4jLong obj = 0L;
            Nd4jLong act = 0L;
            Nd4jLong ttl = 0L;
            for (auto v: _profiles) {
                tmp += v->getTemporarySize();
                obj += v->getObjectsSize();
                act += v->getActivationsSize();
                ttl += v->getTotalSize();
            }

            nd4j_printf("ACT: %lld; TMP: %lld; OBJ: %lld; TTL: %lld;\n", act / _merges, tmp / _merges, obj / _merges, ttl / _merges);

            nd4j_printf("\nTime:\n", "");
            nd4j_printf("Construction time: %lld ns;\n", _buildTime / _merges);
            nd4j_printf("Execution time: %lld ns;\n", _executionTime / _merges);

            nd4j_printf("\nPer-node reports:\n", "");
            if (_profiles.empty())
                nd4j_printf("No nodes in graph\n","");

            for (auto v: _profiles)
                v->printOut();
            
            nd4j_printf("\nSpecial timers:\n", "");
            if (_timings.empty())
                nd4j_printf("No special timers were set\n","");

            for (auto v: _timings)
                nd4j_printf("%s: %lld ns;\n", v.first.c_str(), v.second);
        }
    }
}