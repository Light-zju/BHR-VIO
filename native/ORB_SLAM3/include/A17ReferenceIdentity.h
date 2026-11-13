#pragma once
#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "A14TemporalProposals.h"
#include <map>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <tuple>
#include <sstream>
#include <vector>
#include <cstdlib>
namespace ORB_SLAM3 { namespace a17_identity {
// Raw query identity is immutable for a canonical landmark. It is a data
// observation, not an additional pose variable or an independent measurement.
struct Anchor {double stamp=0;unsigned long frame=0,keyframe=0;int index=-1;cv::KeyPoint key;};
inline std::map<unsigned long,Anchor>& registry(){static std::map<unsigned long,Anchor> r;return r;}
inline std::mutex& mutex(){static std::mutex m;return m;}
inline void log(const std::string&s){if(const char*p=std::getenv("VIO_A17_IDENTITY_LOG")){static std::mutex m;std::lock_guard<std::mutex>g(m);std::ofstream f(p,std::ios::app);f<<s;}}
inline bool anchorFor(MapPoint*p,const Frame*b,Anchor&out){
 {std::lock_guard<std::mutex>g(mutex());auto i=registry().find(p->mnId);if(i!=registry().end()){out=i->second;return out.stamp<b->mTimeStamp;}}
 KeyFrame*k=p->GetReferenceKeyFrame();if(!k||k->mTimeStamp>=b->mTimeStamp)return false;
 int i=std::get<0>(p->GetIndexInKeyFrame(k));if(i<0||i>=int(k->mvKeysUn.size()))return false;
 Anchor a;a.stamp=k->mTimeStamp;a.frame=k->mnFrameId;a.keyframe=k->mnId;a.index=i;a.key=k->mvKeysUn[i];
 {std::lock_guard<std::mutex>g(mutex());out=registry().emplace(p->mnId,a).first->second;}
 return out.stamp<b->mTimeStamp;
}
inline bool isReference(unsigned long point,double stamp,size_t index){std::lock_guard<std::mutex>g(mutex());auto i=registry().find(point);return i!=registry().end()&&i->second.stamp==stamp&&i->second.index==int(index);}
struct View {double mTimeStamp=0;std::vector<cv::KeyPoint> mvKeysUn;std::vector<int> matches;std::vector<unsigned long> points;unsigned long frame=0;};
struct Result {std::vector<a14::Proposal> proposals;std::vector<Anchor> anchors;std::vector<unsigned char> registered;};
template<class Canonical>inline Result propose(Frame*b,Canonical canonical,const std::vector<unsigned char>&eligible){
 Result result;result.proposals.resize(b->N);result.anchors.resize(b->N);result.registered.assign(b->N,0);
 std::map<double,View> groups;std::map<unsigned long,bool> seen;
 for(int j=0;j<b->N;++j){if(j>=int(eligible.size())||!eligible[j])continue;auto*p=canonical(b->mvpMapPoints[j]);if(!p||b->mvuRight[j]<0||seen[p->mnId])continue;seen[p->mnId]=true;Anchor a;if(!anchorFor(p,b,a))continue;
  auto&v=groups[a.stamp];v.mTimeStamp=a.stamp;v.frame=a.frame;
  if(v.matches.size()<=size_t(a.index)){v.matches.resize(a.index+1,-1);v.mvKeysUn.resize(a.index+1);v.points.resize(a.index+1,0);}
  // An original pixel cannot simultaneously be the anchor of two live points
  // in one query batch. Preserve evidence and decline the conflicting query.
  if(v.matches[a.index]>=0){std::ostringstream s;s<<"anchor_collision,"<<p->mnId<<","<<v.points[a.index]<<","<<a.frame<<","<<a.index<<"\n";log(s.str());continue;}
  v.matches[a.index]=j;v.mvKeysUn[a.index]=a.key;v.points[a.index]=p->mnId;result.anchors[j]=a;result.registered[j]=1;
 }
 for(auto&entry:groups){auto&v=entry.second;auto proposals=a14::propose(&v,b,v.matches);
  for(size_t i=0;i<v.matches.size();++i){int j=v.matches[i];if(j<0)continue;result.proposals[j]=proposals[i];std::ostringstream s;s<<std::setprecision(17)<<"query,"<<v.points[i]<<","<<v.mTimeStamp<<","<<v.frame<<","<<i<<","<<b->mTimeStamp<<","<<b->mnId<<","<<j<<","<<proposals[i].valid<<","<<proposals[i].stereo<<","<<v.mvKeysUn[i].pt.x<<","<<v.mvKeysUn[i].pt.y<<","<<b->mvKeysUn[j].pt.x<<","<<b->mvKeysUn[j].pt.y<<"\n";log(s.str());}
 }
 return result;
}
}}
