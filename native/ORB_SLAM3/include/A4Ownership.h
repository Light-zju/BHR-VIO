#pragma once
#include <mutex>
namespace ORB_SLAM3 { namespace a4_owner {
inline std::mutex& mutex(){static std::mutex m;return m;}
// Commit the map's ownership before its separate MapPoint/KF insertions.
// Never reclaim a previously map-owned pixel after outlier removal.
template<class K>inline bool mapAcquire(K*k,size_t i){
 std::lock_guard<std::mutex>l(mutex());
 if(i>=k->mvA3Consumed.size())return false;
 if(k->mvA3Consumed[i])return true;
 if(i<k->mvA4MapClaimed.size())k->mvA4MapClaimed[i]=1;
 return false;
}
template<class K>inline bool seedAcquire(K*k,size_t i){
 std::lock_guard<std::mutex>l(mutex());
 if(i>=k->mvA3Consumed.size()||i>=k->mvA4MapClaimed.size()||k->mvA3Consumed[i]||k->mvA4MapClaimed[i]||k->GetMapPoint(i))return false;
 __atomic_store_n(&k->mvA3Consumed[i],1,__ATOMIC_RELEASE);return true;
}
} }
