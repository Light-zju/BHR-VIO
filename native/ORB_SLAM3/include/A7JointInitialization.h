#pragma once
#include "G2oTypes.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Map.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <chrono>
namespace ORB_SLAM3 { namespace a7_init {
struct State {KeyFrame*k;Sophus::SE3f pose;Eigen::Vector3f velocity;IMU::Bias bias;};
struct Pending {Map*map=nullptr;std::vector<State> states;std::vector<std::pair<MapPoint*,Eigen::Vector3f>> points;};
inline Pending& pending(){static thread_local Pending value;return value;}

inline void publish(Map*map,const std::vector<KeyFrame*>&queue){
 auto&p=pending();if(!p.map)return;if(p.map!=map)throw std::runtime_error("A7 pending map mismatch");
 std::map<KeyFrame*,Sophus::SE3f> before,after;
 std::map<KeyFrame*,Eigen::Vector3f> velocityBefore,velocityAfter;
 for(auto*k:map->GetAllKeyFrames()){before[k]=k->GetPose();after[k]=before[k];velocityBefore[k]=k->GetVelocity();velocityAfter[k]=velocityBefore[k];}
 for(auto*k:queue)if(k&&k->GetMap()==map){before[k]=k->GetPose();velocityBefore[k]=k->GetVelocity();}
 for(auto&s:p.states){after[s.k]=s.pose;velocityAfter[s.k]=s.velocity;}
 auto ordered=queue;std::sort(ordered.begin(),ordered.end(),[](KeyFrame*a,KeyFrame*b){return a->mnId<b->mnId;});
 std::vector<State> transported;
 for(auto*k:ordered){
  if(!k||k->isBad()||k->GetMap()!=map||after.count(k))continue;
  auto*parent=k->mPrevKF;
  if(!parent||!before.count(parent)||!after.count(parent))throw std::runtime_error("A7 queued predecessor not transportable");
  after[k]=(before[k]*before[parent].inverse())*after[parent];
  Eigen::Matrix3f rotation=after[parent].inverse().rotationMatrix()*before[parent].rotationMatrix();
  velocityAfter[k]=velocityAfter[parent]+rotation*(velocityBefore[k]-velocityBefore[parent]);
  transported.push_back({k,after[k],velocityAfter[k],p.states.back().bias});
  if(const char*path=std::getenv("VIO_A7_PUBLISH_LOG")){std::ofstream f(path,std::ios::app);f<<std::setprecision(17)<<"queue_transport,"<<std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()<<","<<k->mTimeStamp<<","<<k->mnId<<","<<parent->mnId<<","<<(after[k].translation()-before[k].translation()).norm()<<","<<(velocityAfter[k]-velocityBefore[k]).norm()<<"\n";}
 }
 std::set<MapPoint*> optimized;for(auto&m:p.points)optimized.insert(m.first);
 std::vector<std::pair<MapPoint*,Eigen::Vector3f>> transportedPoints;
 for(auto*mp:map->GetAllMapPoints())if(mp&&!mp->isBad()&&!optimized.count(mp)){
  auto*k=mp->GetReferenceKeyFrame();if(k&&before.count(k)&&after.count(k))transportedPoints.emplace_back(mp,after[k].inverse()*(before[k]*mp->GetWorldPos()));
 }
 p.states.insert(p.states.end(),transported.begin(),transported.end());
 for(auto&s:p.states){if(s.k->isBad())continue;s.k->SetPose(s.pose);s.k->SetVelocity(s.velocity);bool reintegrate=(s.k->GetGyroBias()-Eigen::Vector3f(s.bias.bwx,s.bias.bwy,s.bias.bwz)).norm()>.01;s.k->SetNewBias(s.bias);if(reintegrate&&s.k->mpImuPreintegrated)s.k->mpImuPreintegrated->Reintegrate();}
 p.points.insert(p.points.end(),transportedPoints.begin(),transportedPoints.end());
 for(auto&m:p.points)if(!m.first->isBad()){m.first->SetWorldPos(m.second);m.first->UpdateNormalAndDepth();}
 if(const char*path=std::getenv("VIO_A7_INIT_LOG")){std::ofstream f(path,std::ios::app);f<<"published,"<<p.states.size()<<","<<p.points.size()<<","<<transported.size()<<","<<transportedPoints.size()<<"\n";}
 p=Pending();
}
inline void rotateQueued(Map*map,const std::vector<KeyFrame*>&queue,const Sophus::SE3f&T,float scale){
 auto mapped=map->GetAllKeyFrames();std::set<KeyFrame*> existing(mapped.begin(),mapped.end());size_t count=0;
 for(auto*k:queue)if(k&&!k->isBad()&&k->GetMap()==map&&!existing.count(k)){
  Sophus::SE3f Twc=k->GetPoseInverse();Twc.translation()*=scale;k->SetPose((T*Twc).inverse());k->SetVelocity(T.rotationMatrix()*k->GetVelocity()*scale);++count;
  if(const char*path=std::getenv("VIO_A7_PUBLISH_LOG")){std::ofstream f(path,std::ios::app);f<<std::setprecision(17)<<"queue_gravity_rotation,"<<std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()<<","<<k->mTimeStamp<<","<<k->mnId<<"\n";}
 }
 if(const char*path=std::getenv("VIO_A7_INIT_LOG")){std::ofstream f(path,std::ios::app);f<<"rotated_queued,"<<count<<"\n";}
}
inline std::vector<std::pair<MapPoint*,g2o::VertexSBAPointXYZ*>> visual(g2o::SparseOptimizer&optimizer,Map*map,unsigned long maxKFid){
 std::vector<std::pair<MapPoint*,g2o::VertexSBAPointXYZ*>> out;
 int next=0;for(const auto&v:optimizer.vertices())next=std::max(next,v.first+1);
 for(auto*mp:map->GetAllMapPoints()){
  if(!mp||mp->isBad())continue;
  std::vector<std::pair<KeyFrame*,int>> observations;
  for(const auto&o:mp->GetObservations()){auto*k=o.first;int i=std::get<0>(o.second);if(k&&!k->isBad()&&k->mnId<=maxKFid&&i>=0&&optimizer.vertex(k->mnId)){if(k->NLeft!=-1)throw std::runtime_error("A7 expected common rectified stereo");observations.emplace_back(k,i);}}
  if(observations.empty())continue;
  auto*point=new g2o::VertexSBAPointXYZ();point->setId(next++);point->setEstimate(mp->GetWorldPos().cast<double>());point->setMarginalized(true);optimizer.addVertex(point);
  for(const auto&o:observations){auto*k=o.first;int i=o.second;auto*vp=dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(k->mnId));auto kp=k->mvKeysUn[i];double info=k->mvInvLevelSigma2[kp.octave];
   if(k->mvuRight[i]>=0){auto*e=new EdgeStereo(0);e->setVertex(0,point);e->setVertex(1,vp);e->setMeasurement(Eigen::Vector3d(kp.pt.x,kp.pt.y,k->mvuRight[i]));e->setInformation(info*Eigen::Matrix3d::Identity());auto*r=new g2o::RobustKernelHuber;r->setDelta(std::sqrt(7.815));e->setRobustKernel(r);optimizer.addEdge(e);}
   else{auto*e=new EdgeMono(0);e->setVertex(0,point);e->setVertex(1,vp);e->setMeasurement(Eigen::Vector2d(kp.pt.x,kp.pt.y));e->setInformation(info*Eigen::Matrix2d::Identity());auto*r=new g2o::RobustKernelHuber;r->setDelta(std::sqrt(5.991));e->setRobustKernel(r);optimizer.addEdge(e);}
  }
  out.emplace_back(mp,point);
 }
 return out;
}
} }
