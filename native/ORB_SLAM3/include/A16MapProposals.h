#include "A17ReferenceIdentity.h"
#pragma once
#include "Frame.h"
#include "MapPoint.h"
#include "A14TemporalProposals.h"
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <sstream>
#include <fstream>
#include <mutex>
#include <iomanip>
namespace ORB_SLAM3 { namespace a16_map_route {
inline void log(const std::string&s){if(const char*p=std::getenv("VIO_A16_LOG")){static std::mutex mutex;std::lock_guard<std::mutex>lock(mutex);std::ofstream f(p,std::ios::app);f<<s;}}
inline unsigned long& currentGraph(){static thread_local unsigned long id=0;return id;}
inline long long monotonicNs(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct GraphScope{unsigned long previous,id;const char*kind;GraphScope(const char*k):kind(k){static std::atomic<unsigned long>counter{0};previous=currentGraph();id=++counter;currentGraph()=id;std::ostringstream s;s<<"graph_begin,"<<id<<","<<kind<<","<<monotonicNs()<<"\n";log(s.str());}~GraphScope(){std::ostringstream s;s<<"graph_end,"<<id<<","<<kind<<","<<monotonicNs()<<"\n";log(s.str());currentGraph()=previous;}};
inline void published(unsigned long mapID,int change,bool immediate){std::ostringstream s;s<<"graph_published,"<<currentGraph()<<","<<monotonicNs()<<","<<mapID<<","<<change<<","<<immediate<<"\n";log(s.str());}
inline MapPoint* canonical(MapPoint*p){for(int i=0;p&&i<32;++i){auto*r=p->GetReplaced();if(!r)return p->isBad()?nullptr:p;p=r;}return nullptr;}
inline void prepare(Frame*b){
 if(!std::getenv("VIO_A16_PROPOSE"))return;
 b->mvA16U.assign(b->N,0);b->mvA16V.assign(b->N,0);b->mvA16Right.assign(b->N,-1);b->mvA16Valid.assign(b->N,0);b->mvA16Point.assign(b->N,0);
 Frame*a=b->mpPrevFrame;if(!a||a->Nleft!=-1||b->Nleft!=-1)return;
 std::unordered_map<unsigned long,int> target;for(int j=0;j<b->N;++j){auto*p=canonical(b->mvpMapPoints[j]);if(p&&b->mvuRight[j]>=0)target.emplace(p->mnId,j);}
 std::vector<int> matches(a->N,-1);std::unordered_map<unsigned long,bool> seen;
 for(int i=0;i<a->N;++i){auto*p=canonical(a->mvpMapPoints[i]);if(!p||seen[p->mnId])continue;auto it=target.find(p->mnId);if(it!=target.end()){matches[i]=it->second;seen[p->mnId]=true;}}
 auto candidates=a14::propose(a,b,matches);
 std::vector<unsigned char> eligible(b->N,0);for(int j:matches)if(j>=0)eligible[j]=1;
 auto anchored=a17_identity::propose(b,canonical,eligible);
 const bool useAnchor=std::getenv("VIO_A17_ANCHOR")!=nullptr;

 for(int i=0;i<a->N;++i){int j=matches[i];if(j<0)continue;auto*p=canonical(b->mvpMapPoints[j]);const auto&v=useAnchor?anchored.proposals[j]:candidates[i];if(!p)continue; if(useAnchor&&!anchored.registered[j])continue; const unsigned long hostFrame=useAnchor?anchored.anchors[j].frame:a->mnId;const int hostIndex=useAnchor?anchored.anchors[j].index:i;
  if(v.valid){b->mvA16U[j]=v.u;b->mvA16V[j]=v.v;b->mvA16Right[j]=v.right;b->mvA16Valid[j]=v.stereo?2:1;b->mvA16Point[j]=p->mnId;}
  std::ostringstream s;s<<std::setprecision(17)<<"proposal,"<<b->mTimeStamp<<","<<hostFrame<<","<<hostIndex<<","<<b->mnId<<","<<j<<","<<p->mnId<<","<<v.valid<<","<<v.u<<","<<v.v<<","<<v.right<<","<<v.stereo<<"\n";log(s.str());
 }
}
template<class Edge,class F>inline void configure(Edge*e,F*f,size_t i,unsigned long pointID,bool persistent){
 if(!std::getenv("VIO_A16_PROPOSE")||i>=f->mvA16Valid.size()||!f->mvA16Valid[i]||f->mvA16Point[i]!=pointID)return;
 bool enabled=std::getenv("VIO_A16_ACTIVE")&&(!persistent||std::getenv("VIO_A16_HISTORY"));
 if(std::getenv("VIO_A17_ANCHOR")&&a17_identity::isReference(pointID,f->mTimeStamp,i)){
  enabled=false;std::ostringstream r;r<<std::setprecision(17)<<"reference_preserved,"<<pointID<<","<<f->mTimeStamp<<","<<i<<","<<persistent<<","<<currentGraph()<<"\n";a17_identity::log(r.str());
 }

 Eigen::Vector3d alt(f->mvA16U[i],f->mvA16V[i],f->mvA16Right[i]);bool stereo=f->mvA16Valid[i]==2;
 if(std::getenv("VIO_A16_DUPLICATE")){alt<<f->mvKeysUn[i].pt.x,f->mvKeysUn[i].pt.y,f->mvuRight[i];stereo=true;}
 e->setA16Alternative(alt,stereo,enabled);e->a16Point=pointID;e->a16Time=f->mTimeStamp;e->a16Feature=i;e->a16Persistent=persistent;e->a16Graph=currentGraph();
 std::ostringstream s;s<<std::setprecision(17)<<"edge,"<<f->mTimeStamp<<","<<pointID<<","<<i<<","<<persistent<<","<<e->a16Active<<","<<stereo<<","<<alt.transpose()<<","<<currentGraph()<<"\n";log(s.str());
}
}}
