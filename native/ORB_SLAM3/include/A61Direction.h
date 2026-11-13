#pragma once
#include "G2oTypes.h"
#include "KeyFrame.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include <fstream>
#include <map>
namespace ORB_SLAM3 { namespace a61 {
inline std::map<long,Eigen::Vector3d> rays(KeyFrame *k){
 std::map<long,Eigen::Vector3d> out;
 std::ifstream f("/grid/"+std::to_string(std::llround(k->mTimeStamp*1e6))+".txt");
 long id;double x,y,d;while(f>>id>>x>>y>>d)out.emplace(id,Eigen::Vector3d((x-k->cx)/k->fx,(y-k->cy)/k->fy,1.));return out;
}
class Direction:public g2o::BaseBinaryEdge<1,double,VertexPose,VertexPose>{
public:
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 Eigen::Vector3d h,k;double fxh,fyh,fxk,fyk;
 Direction(Eigen::Vector3d x,Eigen::Vector3d y,KeyFrame *a,KeyFrame *b):h(x),k(y),fxh(a->fx),fyh(a->fy),fxk(b->fx),fyk(b->fy){setMeasurement(0);setInformation(Eigen::Matrix<double,1,1>::Identity());}
 bool read(std::istream&) override{return false;}bool write(std::ostream&)const override{return false;}
 double signedPixel(bool &defined)const{
   const auto &a=static_cast<const VertexPose*>(_vertices[0])->estimate(), &b=static_cast<const VertexPose*>(_vertices[1])->estimate();
   const Eigen::Matrix3d R=b.Rcw[0]*a.Rcw[0].transpose();
   const Eigen::Vector3d t=b.tcw[0]-R*a.tcw[0];
   const Eigen::Vector3d l=t.cross(R*h),m=R.transpose()*k.cross(t);
   const double v=std::pow(l.x()/fxk,2)+std::pow(l.y()/fyk,2)+std::pow(m.x()/fxh,2)+std::pow(m.y()/fyh,2);
   defined=std::isfinite(v)&&v>1e-24;
   return defined?k.dot(l)/std::sqrt(v):0.;
 }
 void computeError()override{bool ok;_error[0]=signedPixel(ok);}
};
inline std::pair<int,int> add(g2o::SparseOptimizer& opt,const std::vector<KeyFrame*>& kfs,bool insert=true,int* proposed=nullptr){
 int eligible=0,inserted=0;
 for(auto* b:kfs){
   auto *a=b->mPrevKF;if(!a||a->isBad()||b->isBad()||a->mnId%2!=0||b->mnId%2!=1)continue;
   auto *va=dynamic_cast<VertexPose*>(opt.vertex(a->mnId));auto *vb=dynamic_cast<VertexPose*>(opt.vertex(b->mnId));
   if(!va||!vb||(va->fixed()&&vb->fixed()))continue;
   auto x=rays(a),y=rays(b);
   for(const auto &h:x){
     auto k=y.find(h.first);if(k==y.end())continue;++eligible;
     auto e=std::unique_ptr<Direction>(new Direction(h.second,k->second,a,b));e->setVertex(0,va);e->setVertex(1,vb);
     bool ok;double err=e->signedPixel(ok);if(!ok || !std::isfinite(err)|| err*err>3.841459)continue;
     auto *ker=new g2o::RobustKernelHuber;ker->setDelta(std::sqrt(3.841459));e->setRobustKernel(ker);
     if(proposed)++*proposed;if(!insert)continue;
     if(!opt.addEdge(e.get()))throw std::runtime_error("A61 factor insertion failed");e.release();++inserted;
   }
 }
 return {eligible,inserted};
}
}}
