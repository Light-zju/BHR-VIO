#pragma once
// v30 research: SVO seed-to-pose entry + v25 depth integral / ownership.
// Existing algorithms are controls, not a standalone novelty claim.
#include "G2oTypes.h"
#include "ORBmatcher.h"
#include "A1BirthPolicy.h"
#include "A4Ownership.h"
#include "A5SourceRecord.h"
#include "A14TemporalProposals.h"
#include <set>
#include <atomic>
#include <tuple>
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include <boost/math/distributions/normal.hpp>
#include <array>
#include <chrono>
namespace ORB_SLAM3 { namespace a3_seed {
inline bool active(){return std::getenv("VIO_A3_SEED")!=nullptr;}
inline bool history(){return std::getenv("VIO_A3_HISTORY")!=nullptr;}
template<class T>inline bool owned(const T*f,size_t i){return i<f->mvA3Consumed.size()&&__atomic_load_n(&f->mvA3Consumed[i],__ATOMIC_ACQUIRE);}
inline void log(const char*env,const std::string&s){if(const char*p=std::getenv(env))a1_birth::writeLine(p,s);}

inline void event(const char*phase,unsigned long call,unsigned long graph,VertexPose*A,VertexPose*B){
 if(!std::getenv("VIO_A6_EVENT_LOG"))return;
 const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
 std::ostringstream o;o<<std::setprecision(17)<<ns<<","<<phase<<","<<call<<","<<graph;
 for(auto*v:{A,B}){o<<","<<v->id()<<","<<v->fixed();auto p=v->estimate();for(int i=0;i<3;++i)for(int j=0;j<3;++j)o<<","<<p.Rwb(i,j);for(int i=0;i<3;++i)o<<","<<p.twb[i];}
 o<<"\n";log("VIO_A6_EVENT_LOG",o.str());
}
class Edge : public g2o::BaseBinaryEdge<4,Eigen::Vector3d,VertexPose,VertexPose>,public A5SourceRecord {
public:
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 Eigen::Matrix<double,4,12> J;double posterior_metric=0,posterior_bearing=0,posterior_outlier=0;
 Eigen::Matrix<double,12,12> componentH=Eigen::Matrix<double,12,12>::Zero();
 Eigen::Matrix<double,12,12> marginalH=Eigen::Matrix<double,12,12>::Zero();
 Eigen::Matrix<double,12,12> selectedH()const{return std::getenv("VIO_A8_MARGINAL_H")?marginalH:componentH;}
 template<class Anchor> Edge(const Anchor*a,const Frame*b,int i,int j){
  fx=b->fx;fy=b->fy;cx=b->cx;cy=b->cy;bf=b->mbf;stereo=b->mvuRight[j]>=0;
  sourceU=a->mvKeysUn[i].pt.x;sourceV=a->mvKeysUn[i].pt.y;sourceRight=a->mvuRight[i];sourceSigma=std::sqrt(a->mvLevelSigma2[a->mvKeysUn[i].octave]);sourceFx=a->fx;sourceFy=a->fy;sourceCx=a->cx;sourceCy=a->cy;sourceBf=a->mbf;
  ray<< (a->mvKeysUn[i].pt.x-a->cx)/a->fx,(a->mvKeysUn[i].pt.y-a->cy)/a->fy,1.;
  obs<<b->mvKeysUn[j].pt.x,b->mvKeysUn[j].pt.y,b->mvuRight[j];
  alternativeU=obs[0];alternativeV=obs[1];alternativeRight=obs[2];alternativeStereo=stereo;completeAlternative=std::getenv("VIO_A15_METRIC_ALT")!=nullptr;
  sigma=std::sqrt(b->mvLevelSigma2[b->mvKeysUn[j].octave]);broad=std::max(b->mnMaxX-b->mnMinX,b->mnMaxY-b->mnMinY);
  const double mu=(a->mvKeysUn[i].pt.x-a->mvuRight[i])/a->mbf;
  const double sd=std::sqrt(2.*a->mvLevelSigma2[a->mvKeysUn[i].octave])/a->mbf;
  boost::math::normal normal;const double lo=boost::math::cdf(normal,-mu/sd);
  for(int k=0;k<9;++k)rho[k]=mu+sd*boost::math::quantile(normal,lo+(1-lo)*(k+.5)/9.);
  components=std::getenv("VIO_A5_JOINT")?45:9;
  int at=0;const double root3=std::sqrt(3.);const double sourceSigma=std::sqrt(a->mvLevelSigma2[a->mvKeysUn[i].octave]);
  for(int k=0;k<9;++k){
   const int angles=components==45?5:1;
   for(int n=0;n<angles;++n){
    double u=a->mvKeysUn[i].pt.x,v=a->mvKeysUn[i].pt.y,w=1.;
    if(angles==5){u+=.5*a->mbf*(rho[k]-mu);w=n==0?1./3:1./6;
     if(n==1)u+=root3*sourceSigma/std::sqrt(2.);if(n==2)u-=root3*sourceSigma/std::sqrt(2.);
     if(n==3)v+=root3*sourceSigma;if(n==4)v-=root3*sourceSigma;
    }
    Eigen::Vector3d d((u-a->cx)/a->fx,(v-a->cy)/a->fy,1.);double length=d.norm();
    directions[at]=d/length;inverseDistances[at]=rho[k]/length;logWeights[at]=std::log(.9*w/9.);++at;
   }
  }

  if(std::getenv("VIO_A12_PARTIAL_GEOMETRY")){
   if(components!=45)throw std::runtime_error("A12 requires common joint geometry");
   for(int k=0;k<45;++k)logWeights[k]+=std::log(.8/.9);
   // v25 broad inverse-depth branch: left bearing remains informative when
   // stereo depth is not. Fixed prior support is inherited, not sequence fitted.
   for(int k=0;k<65;++k)for(int n=0;n<5;++n){
    const double rhoB=std::exp(std::log(.0005)+k*(std::log(5.)-std::log(.0005))/64.);
    double u=sourceU,v=sourceV,w=n==0?1./3:1./6;
    if(n==1)u+=root3*sourceSigma;if(n==2)u-=root3*sourceSigma;
    if(n==3)v+=root3*sourceSigma;if(n==4)v-=root3*sourceSigma;
    Eigen::Vector3d d((u-sourceCx)/sourceFx,(v-sourceCy)/sourceFy,1.);double norm=d.norm();
    directions[at]=d/norm;inverseDistances[at]=rhoB/norm;logWeights[at]=std::log(.1*w/65.);++at;
   }
   components=at;if(components!=370)throw std::runtime_error("A12 component count");
  }
  if(std::getenv("VIO_A14_ASSOCIATION")){
   if(components!=370)throw std::runtime_error("A14 requires partial source");
   for(int k=45;k<370;++k){logWeights[k]-=std::log(2.);directions[k+325]=directions[k];inverseDistances[k+325]=inverseDistances[k];logWeights[k+325]=logWeights[k];}
   components=695;
  }
  if(std::getenv("VIO_A15_BUNDLE")){
   if(components!=695)throw std::runtime_error("A15 requires common association geometry");
   const auto dd=directions;const auto ii=inverseDistances;const auto ww=logWeights;
   for(int k=0;k<45;++k){directions[k]=dd[k];inverseDistances[k]=ii[k];logWeights[k]=ww[k]-std::log(2.);directions[k+45]=dd[k];inverseDistances[k+45]=ii[k];logWeights[k+45]=logWeights[k];}
   for(int k=45;k<695;++k){directions[k+45]=dd[k];inverseDistances[k+45]=ii[k];logWeights[k+45]=ww[k];}
   components=740;
  }
  setInformation(Eigen::Matrix4d::Identity());setMeasurement(obs);
 }
 explicit Edge(const A5SourceRecord&r){static_cast<A5SourceRecord&>(*this)=r;setInformation(Eigen::Matrix4d::Identity());setMeasurement(obs);}
 bool read(std::istream&)override{return false;}bool write(std::ostream&)const override{return false;}
 void evaluate(const ImuCamPose&A,const ImuCamPose&B,Eigen::Vector4d&r,Eigen::Matrix<double,4,12>&jac){
  using namespace Eigen;std::array<Vector3d,741>rr;std::array<Matrix<double,3,12>,741>jj;
  std::array<double,741>ll,upper;const int D=stereo?3:2;const int count=components+1;
  for(int k=0;k<count;++k){rr[k].setZero();jj[k].setZero();ll[k]=-1e100;upper[k]=k<components?logWeights[k]:0;}
  for(int k=0;k<components;++k){
   double z=inverseDistances[k];Vector3d a=A.Rbc[0]*directions[k]+z*A.tbc[0];
   Vector3d body=B.Rwb.transpose()*(A.Rwb*a+z*(A.twb-B.twb));
   Vector3d p=B.Rcb[0]*(body-z*B.tbc[0]);
   if(!(p.z()>1e-9)||!p.allFinite())continue;
   Vector3d prediction;prediction<<fx*p.x()/p.z()+cx,fy*p.y()/p.z()+cy,fx*p.x()/p.z()+cx-bf*z/p.z();
   const bool bundle=components==740;
   const bool bearing=bundle?k>=90:(components>=370&&k>=45);
   const bool alternative=bundle?(k>=415||(k>=45&&k<90&&completeAlternative)):(components==695&&k>=370);
   const Vector3d measurement=alternative?Vector3d(alternativeU,alternativeV,alternativeRight):obs;
   const bool observedStereo=bundle?(alternative?alternativeStereo:stereo):stereo;
   rr[k]=(prediction-measurement)/sigma;if(!observedStereo)rr[k][2]=0;
   if(bundle){
    Vector3d unused=alternative?obs:Vector3d(alternativeU,alternativeV,alternativeRight);const bool unusedStereo=alternative?stereo:alternativeStereo;
    unused=(unused-Vector3d(cx,cy,cx))/broad;if(!unusedStereo)unused[2]=0;
    upper[k]-=(unusedStereo?3:2)*std::log(broad/sigma)+.5*unused.squaredNorm();
   }else if(components==695){const double unusedU=k>=370?obs[0]:alternativeU,unusedV=k>=370?obs[1]:alternativeV;upper[k]-=2*std::log(broad/sigma)+.5*(std::pow((unusedU-cx)/broad,2)+std::pow((unusedV-cy)/broad,2));}
   if(bearing&&observedStereo){rr[k][2]=(measurement[2]-cx)/broad;upper[k]-=std::log(broad/sigma);}
   Matrix3d h;h<<fx/p.z(),0,-fx*p.x()/(p.z()*p.z()),0,fy/p.z(),-fy*p.y()/(p.z()*p.z()),fx/p.z(),0,(-fx*p.x()+bf*z)/(p.z()*p.z());h/=sigma;if(!observedStereo||bearing)h.row(2).setZero();
   jj[k].block<3,3>(0,0)=h*(-B.Rcb[0]*B.Rwb.transpose()*A.Rwb*Skew(a));
   jj[k].block<3,3>(0,3)=h*(z*B.Rcb[0]*B.Rwb.transpose()*A.Rwb);
   jj[k].block<3,3>(0,6)=h*(B.Rcb[0]*Skew(body));
   jj[k].block<3,3>(0,9)=h*(-z*B.Rcb[0]);
   ll[k]=upper[k]-.5*rr[k].squaredNorm();
  }
  Vector3d center(cx,cy,cx),q=(obs-center)/broad;if(!stereo)q[2]=0;
  upper[components]=std::log(.1)-D*std::log(broad/sigma)-.5*q.squaredNorm();if(components==740){Vector3d qAlt(alternativeU-cx,alternativeV-cy,alternativeRight-cx);qAlt/=broad;if(!alternativeStereo)qAlt[2]=0;upper[components]-=(alternativeStereo?3:2)*std::log(broad/sigma)+.5*qAlt.squaredNorm();}else if(components==695)upper[components]-=2*std::log(broad/sigma)+.5*(std::pow((alternativeU-cx)/broad,2)+std::pow((alternativeV-cy)/broad,2));ll[components]=upper[components];
  int best=std::max_element(ll.begin(),ll.begin()+count)-ll.begin();double peak=ll[best],sum=0;
  for(int k=0;k<count;++k)sum+=std::exp(ll[k]-peak);double lse=peak+std::log(sum);posterior_outlier=std::exp(ll[components]-lse);posterior_bearing=0;if(components>=370)for(int k=components==740?90:45;k<components;++k)posterior_bearing+=std::exp(ll[k]-lse);posterior_metric=1-posterior_outlier-posterior_bearing;
  r.head<3>()=rr[best];r[3]=std::sqrt(std::max(1e-16,2*(std::log(components==740?741.:(components==695?696.:46.))+1-upper[best]-(lse-ll[best]))));
  Matrix<double,1,12>g=Matrix<double,1,12>::Zero();for(int k=0;k<count;++k)if(ll[k]>-1e90)g+=std::exp(ll[k]-lse)*(rr[k].transpose()*jj[k]);
  jac.topRows<3>()=jj[best];jac.row(3)=(g-rr[best].transpose()*jj[best])/r[3];
  // Korotkine et al., RA-L 2024, Eq.30; navlie HSM operator.
  // PSD expected component curvature, NOT the exact marginal Hessian.
  componentH.setZero();
  for(int k=0;k<count;++k)if(ll[k]>-1e90)
    componentH.noalias()+=std::exp(ll[k]-lse)*(jj[k].transpose()*jj[k]);
  marginalH=componentH+g.transpose()*g;
  for(int k=0;k<count;++k)if(ll[k]>-1e90){
    const Matrix<double,12,1>score=jj[k].transpose()*rr[k];
    marginalH.noalias()-=std::exp(ll[k]-lse)*(score*score.transpose());
  }


 }
 void computeError()override{evaluate(static_cast<VertexPose*>(_vertices[0])->estimate(),static_cast<VertexPose*>(_vertices[1])->estimate(),_error,J);}
 void linearizeOplus()override{computeError();_jacobianOplusXi=J.leftCols<6>();_jacobianOplusXj=J.rightCols<6>();}
 Eigen::Matrix<double,12,12> hessian(){computeError();if(std::getenv("VIO_A8_COMPONENT_H")||std::getenv("VIO_A8_MARGINAL_H"))return selectedH();return J.transpose()*J;}
 void constructQuadraticForm()override{
  if(!std::getenv("VIO_A8_COMPONENT_H")&&!std::getenv("VIO_A8_MARGINAL_H")){
    g2o::BaseBinaryEdge<4,Eigen::Vector3d,VertexPose,VertexPose>::constructQuadraticForm();return;
  }
  const Eigen::Matrix<double,12,12> normalH=selectedH();
  assert(!robustKernel());assert((_information-Eigen::Matrix4d::Identity()).norm()<1e-14);
  auto*A=static_cast<VertexPose*>(_vertices[0]);auto*B=static_cast<VertexPose*>(_vertices[1]);
  const Eigen::Matrix<double,12,1> gradient=J.transpose()*_error;
  #ifdef G2O_OPENMP
  A->lockQuadraticForm();B->lockQuadraticForm();
  #endif
  if(!A->fixed()){
    A->b().noalias()-=gradient.head<6>();A->A().noalias()+=normalH.topLeftCorner<6,6>();
    if(!B->fixed()){
      if(_hessianRowMajor)_hessianTransposed.noalias()+=normalH.bottomLeftCorner<6,6>();
      else _hessian.noalias()+=normalH.topRightCorner<6,6>();
    }
  }
  if(!B->fixed()){B->b().noalias()-=gradient.tail<6>();B->A().noalias()+=normalH.bottomRightCorner<6,6>();}
  #ifdef G2O_OPENMP
  B->unlockQuadraticForm();A->unlockQuadraticForm();
  #endif
 }

};

inline void recordSource(Frame*cur,Edge*e){
 std::ostringstream o;o<<std::setprecision(17)<<cur->mTimeStamp<<","<<e->hostFrame<<","<<e->hostFeature<<","<<e->targetFrame<<","<<e->targetFeature<<","<<(e->host?long(e->host->mnId):-1)<<","<<e->components;
 for(double v:{e->sourceU,e->sourceV,e->sourceRight,e->sourceSigma,e->sourceFx,e->sourceFy,e->sourceCx,e->sourceCy,e->sourceBf})o<<","<<v;
 for(int k=0;k<3;++k)o<<","<<e->obs[k];for(double v:{e->fx,e->fy,e->cx,e->cy,e->bf,e->sigma,e->broad})o<<","<<v;
 o<<","<<e->stereo<<"\n";log("VIO_A5_RECORD_LOG",o.str());
}
inline void snapshot(Frame*cur,Edge*e){
 event("current_after",cur->mnId,0,static_cast<VertexPose*>(e->vertex(0)),static_cast<VertexPose*>(e->vertex(1)));
 if(!std::getenv("VIO_A5_SNAPSHOT_LOG"))return;e->computeError();

 if(std::getenv("VIO_A12_BRANCH_LOG")){std::ostringstream z;z<<std::setprecision(17)<<cur->mTimeStamp<<","<<e->hostFrame<<","<<e->hostFeature<<","<<e->targetFrame<<","<<e->targetFeature<<","<<e->components<<","<<e->posterior_metric<<","<<e->posterior_bearing<<","<<e->posterior_outlier<<"\n";log("VIO_A12_BRANCH_LOG",z.str());}
 auto A=static_cast<VertexPose*>(e->vertex(0))->estimate(),B=static_cast<VertexPose*>(e->vertex(1))->estimate();
 std::ostringstream o;o<<std::setprecision(17)<<cur->mTimeStamp<<","<<e->hostFrame<<","<<e->hostFeature<<","<<e->targetFrame<<","<<e->targetFeature<<","<<e->level()<<","<<e->posterior_metric;
 for(int i=0;i<4;++i)o<<","<<e->error()[i];o<<","<<(e->J.transpose()*e->error()).norm();
 for(const auto&P:{A,B}){for(int i=0;i<3;++i)for(int j=0;j<3;++j)o<<","<<P.Rwb(i,j);for(int i=0;i<3;++i)o<<","<<P.twb[i];for(int i=0;i<3;++i)for(int j=0;j<3;++j)o<<","<<P.Rbc[0](i,j);for(int i=0;i<3;++i)o<<","<<P.tbc[0][i];}
 o<<"\n";log("VIO_A5_SNAPSHOT_LOG",o.str());
}
inline std::vector<Edge*> attach(g2o::SparseOptimizer&optimizer,Frame*prev,Frame*cur,VertexPose*A,VertexPose*B){
 event("current_before",cur->mnId,0,A,B);
 std::vector<Edge*>edges;
 if(!prev||!prev->mpcpi||prev->Nleft!=-1||cur->Nleft!=-1||!(prev->mbf>0))return edges;
 const bool previousKF=cur->mpLastKeyFrame&&cur->mpLastKeyFrame->mnFrameId==prev->mnId;
 if(previousKF&&!std::getenv("VIO_A4_CROSS_KF"))return edges;
 std::vector<cv::Point2f>predicted;predicted.reserve(prev->N);
 for(const auto&k:prev->mvKeysUn)predicted.push_back(k.pt);
 std::vector<int>matches;ORBmatcher matcher(.6,true);matcher.SearchForInitialization(*prev,*cur,predicted,matches,100);
 auto proposals=a14::propose(prev,cur,matches);
 for(int i=0;i<prev->N;++i){int j=matches[i];if(j<0||prev->mvpMapPoints[i]||cur->mvpMapPoints[j]||owned(prev,i)||owned(cur,j)||!a1_birth::neural(*prev,i)||!(prev->mvDepth[i]>0))continue;
  // All arms compute matching proposals; only active arms consume observations.
  if(!active())continue;
  if(previousKF&&std::getenv("VIO_A4_CROSS_KF")&&!a4_owner::seedAcquire(cur->mpLastKeyFrame,i))continue;
  Edge*e=new Edge(prev,cur,i,j);if(std::getenv("VIO_A14_USE_LEARNED")){e->alternativeU=proposals[i].u;e->alternativeV=proposals[i].v;e->alternativeValid=proposals[i].valid;e->alternativeRight=proposals[i].right;e->alternativeStereo=proposals[i].stereo;}e->setVertex(0,A);e->setVertex(1,B);optimizer.addEdge(e);if(std::getenv("VIO_A4_OWNERSHIP_ONLY"))e->setLevel(1);edges.push_back(e);
  prev->mvA3Consumed[i]=1;cur->mvA3Consumed[j]=1;
  e->host=previousKF?cur->mpLastKeyFrame:nullptr;e->hostFrame=prev->mnId;e->targetFrame=cur->mnId;e->hostFeature=i;e->targetFeature=j;
  cur->mvA5Pending.push_back(std::shared_ptr<A5SourceRecord>(new A5SourceRecord(*e)));recordSource(cur,e);
  {std::ostringstream z;z<<std::setprecision(17)<<cur->mTimeStamp<<","<<e->hostFrame<<","<<i<<","<<cur->mnId<<","<<j<<","<<e->alternativeRight<<","<<e->alternativeStereo<<","<<e->completeAlternative<<","<<proposals[i].seedRight<<","<<cur->mA15StereoCostThreshold<<"\n";log("VIO_A15_BUNDLE_LOG",z.str());}

  {std::ostringstream z;z<<std::setprecision(17)<<cur->mTimeStamp<<","<<e->hostFrame<<","<<i<<","<<cur->mnId<<","<<j<<","<<e->obs[0]<<","<<e->obs[1]<<","<<proposals[i].u<<","<<proposals[i].v<<","<<proposals[i].valid<<","<<e->alternativeU<<","<<e->alternativeV<<","<<e->components<<"\n";log("VIO_A14_ASSOCIATION_LOG",z.str());}
  std::ostringstream s;s<<std::setprecision(17)<<cur->mTimeStamp<<","<<prev->mnId<<","<<i<<","<<cur->mnId<<","<<j<<","<<e->rho[4]<<","<<e->sigma<<"\n";log("VIO_A3_OWNER_LOG",s.str());
 }
 {std::ostringstream o;o<<std::setprecision(17)<<cur->mTimeStamp<<",0,"<<prev->mnId<<",0,0,0,"<<edges.size()<<"\n";log("VIO_A4_ENTRY_LOG",o.str());}
 return edges;
}

// Anchor only at the actual native LastKeyFrame vertex. Never attach a
// previous-frame measurement to a vertex with a different timestamp.
inline std::vector<Edge*> attachKF(g2o::SparseOptimizer&optimizer,KeyFrame*kf,Frame*cur,VertexPose*A,VertexPose*B){
 event("current_before",cur->mnId,0,A,B);
 std::vector<Edge*>edges;if(!kf||kf->mnFrameId>=cur->mnId||kf->isBad()||cur->Nleft!=-1||!(kf->mbf>0))return edges;
 Frame proxy;proxy.mvKeysUn=kf->mvKeysUn;proxy.mDescriptors=kf->mDescriptors;
 std::vector<cv::Point2f>predicted;for(const auto&k:kf->mvKeysUn)predicted.push_back(k.pt);
 std::vector<int>matches;ORBmatcher matcher(.6,true);matcher.SearchForInitialization(proxy,*cur,predicted,matches,100);
 auto proposals=a14::propose(kf,cur,matches);
 int matched=0,eligible=0,rejectedOwnership=0;
 for(size_t i=0;i<matches.size();++i){int j=matches[i];if(j<0)continue;++matched;
  if(cur->mvpMapPoints[j]||owned(cur,j)||!a1_birth::neural(*kf,i)||!(kf->mvDepth[i]>0))continue;
  ++eligible;if(!active()||!std::getenv("VIO_A4_CROSS_KF"))continue;
  if(!a4_owner::seedAcquire(kf,i)){++rejectedOwnership;continue;}
  auto*e=new Edge(kf,cur,i,j);if(std::getenv("VIO_A14_USE_LEARNED")){e->alternativeU=proposals[i].u;e->alternativeV=proposals[i].v;e->alternativeValid=proposals[i].valid;e->alternativeRight=proposals[i].right;e->alternativeStereo=proposals[i].stereo;}e->setVertex(0,A);e->setVertex(1,B);optimizer.addEdge(e);
  if(std::getenv("VIO_A4_OWNERSHIP_ONLY"))e->setLevel(1);edges.push_back(e);cur->mvA3Consumed[j]=1;
  e->host=kf;e->hostFrame=kf->mnFrameId;e->targetFrame=cur->mnId;e->hostFeature=i;e->targetFeature=j;
  cur->mvA5Pending.push_back(std::shared_ptr<A5SourceRecord>(new A5SourceRecord(*e)));recordSource(cur,e);
  {std::ostringstream z;z<<std::setprecision(17)<<cur->mTimeStamp<<","<<e->hostFrame<<","<<i<<","<<cur->mnId<<","<<j<<","<<e->alternativeRight<<","<<e->alternativeStereo<<","<<e->completeAlternative<<","<<proposals[i].seedRight<<","<<cur->mA15StereoCostThreshold<<"\n";log("VIO_A15_BUNDLE_LOG",z.str());}

  {std::ostringstream z;z<<std::setprecision(17)<<cur->mTimeStamp<<","<<e->hostFrame<<","<<i<<","<<cur->mnId<<","<<j<<","<<e->obs[0]<<","<<e->obs[1]<<","<<proposals[i].u<<","<<proposals[i].v<<","<<proposals[i].valid<<","<<e->alternativeU<<","<<e->alternativeV<<","<<e->components<<"\n";log("VIO_A14_ASSOCIATION_LOG",z.str());}
  if(cur->mpPrevFrame&&cur->mpPrevFrame->mnId==kf->mnFrameId&&i<cur->mpPrevFrame->mvA3Consumed.size())cur->mpPrevFrame->mvA3Consumed[i]=1;
  std::ostringstream o;o<<std::setprecision(17)<<cur->mTimeStamp<<","<<kf->mnFrameId<<","<<i<<","<<cur->mnId<<","<<j<<","<<e->rho[4]<<","<<e->sigma<<"\n";log("VIO_A3_OWNER_LOG",o.str());
 }
 std::ostringstream o;o<<std::setprecision(17)<<cur->mTimeStamp<<",1,"<<kf->mnFrameId<<","<<matched<<","<<eligible<<","<<rejectedOwnership<<","<<edges.size()<<"\n";log("VIO_A4_ENTRY_LOG",o.str());return edges;
}

inline std::vector<Edge*> replayMap(g2o::SparseOptimizer&optimizer,const std::vector<KeyFrame*>&targets,unsigned long callID){
 static std::atomic<unsigned long> sequence(0);const unsigned long graph=sequence.fetch_add(1)+1;
 std::vector<Edge*>edges;std::set<std::tuple<unsigned long,unsigned long,unsigned long,unsigned long>>seen;
 for(auto*t:targets){
  if(!t||t->isBad())continue;auto*B=dynamic_cast<VertexPose*>(optimizer.vertex(t->mnId));if(!B)continue;
  for(const auto&record:t->mvA5Sources){
   auto*h=record->host;if(!h||h->isBad()||h->GetMap()!=t->GetMap())continue;
   auto id=std::make_tuple(record->hostFrame,record->hostFeature,record->targetFrame,record->targetFeature);
   if(!seen.insert(id).second)throw std::runtime_error("A5 duplicated source in one native graph");
   auto*A=dynamic_cast<VertexPose*>(optimizer.vertex(h->mnId));
   if(!A){A=new VertexPose(h);A->setId(h->mnId);A->setFixed(true);optimizer.addVertex(A);}
   if(A->fixed()&&B->fixed())continue;
   auto*e=new Edge(*record);e->graphSerial=graph;e->setVertex(0,A);e->setVertex(1,B);optimizer.addEdge(e);
   if(!std::getenv("VIO_A5_REPLAY"))e->setLevel(1);edges.push_back(e);
 event("map_before",callID,graph,A,B);
   std::ostringstream o;o<<std::setprecision(17)<<t->mTimeStamp<<","<<callID<<","<<graph<<","<<record->hostFrame<<","<<record->hostFeature<<","<<record->targetFrame<<","<<record->targetFeature<<","<<h->mnId<<","<<t->mnId<<","<<e->level()<<","<<record->components<<"\n";log("VIO_A5_MAP_LOG",o.str());
  }
 }
 return edges;
}
inline void mapReceipt(unsigned long callID,int kind,const std::vector<Edge*>&edges,int iterations){
 size_t active=0;double cost=0;unsigned long graph=edges.empty()?0:edges[0]->graphSerial;
 for(auto*e:edges)event("map_published",callID,graph,static_cast<VertexPose*>(e->vertex(0)),static_cast<VertexPose*>(e->vertex(1)));
 for(auto*e:edges)if(e->level()==0){e->computeError();++active;cost+=e->chi2();}
 std::ostringstream o;o<<std::setprecision(17)<<callID<<","<<graph<<","<<kind<<","<<iterations<<","<<edges.size()<<","<<active<<","<<cost<<"\n";log("VIO_A5_RECEIPT_LOG",o.str());
}
} }

