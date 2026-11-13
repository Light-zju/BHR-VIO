#include "A3SeedMotion.h"
#include "Optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include <iostream>
#include <thread>
#include <atomic>
using namespace ORB_SLAM3;using namespace Eigen;
ImuCamPose pose(Vector3d r,Vector3d t){ImuCamPose p;p.Rwb=ExpSO3(r);p.twb=t;p.its=0;p.Rbc={ExpSO3(Vector3d(.01,-.02,.005))};p.tbc={Vector3d(.06,.01,.02)};p.Rcb={p.Rbc[0].transpose()};return p;}
struct OwnershipFixture{std::vector<unsigned char>mvA3Consumed{0},mvA4MapClaimed{0};void*GetMapPoint(size_t){return nullptr;}};
int main(){
 int ownershipCases=0;for(int k=0;k<200;++k){OwnershipFixture f;bool map=false,seed=false;std::thread a([&]{map=!a4_owner::mapAcquire(&f,0);}),b([&]{seed=a4_owner::seedAcquire(&f,0);});a.join();b.join();if(int(map)+int(seed)!=1)return 21;if(seed&&a4_owner::seedAcquire(&f,0))return 22;++ownershipCases;}
 Frame a,b;for(Frame*f:{&a,&b}){f->fx=400;f->fy=410;f->cx=320;f->cy=240;f->mbf=200;f->mnMaxX=640;f->mnMinX=0;f->mnMaxY=480;f->mnMinY=0;}
 a.mvKeysUn={cv::KeyPoint(300,220,1)};b.mvKeysUn={cv::KeyPoint(304,221,1)};a.mvLevelSigma2={1};b.mvLevelSigma2={1};a.mvuRight={280};b.mvuRight={284};
 double maxerr=0,maxgauge=0,maxcost=0,maxschur=0,maxJ=0,minPosterior=1;int tests=0;
 for(int k=0;k<100;++k){
  a.mvuRight[0]=300-(2.+.4*k);b.mvuRight[0]=(k%2)?284:-1;a3_seed::Edge e(&a,&b,0,0);
  ImuCamPose A=pose(Vector3d(.02,-.01,.005),Vector3d(.2,-.1,.3)),B=pose(Vector3d(.021+.0001*k,-.012,.006),Vector3d(.23,-.095,.305));
  double rho=(300-a.mvuRight[0])/a.mbf;Vector3d body=B.Rwb.transpose()*(A.Rwb*(A.Rbc[0]*e.ray+rho*A.tbc[0])+rho*(A.twb-B.twb));Vector3d cp=B.Rcb[0]*(body-rho*B.tbc[0]);e.obs<<e.fx*cp.x()/cp.z()+e.cx+.1,e.fy*cp.y()/cp.z()+e.cy-.08,e.fx*cp.x()/cp.z()+e.cx-e.bf*rho/cp.z()+.12;
  if(!e.stereo)e.obs[2]=-1;
  Vector4d r;Matrix<double,4,12>J;e.evaluate(A,B,r,J);maxJ=std::max(maxJ,J.norm());minPosterior=std::min(minPosterior,e.posterior_metric);
  Matrix<double,4,12>num;const double h=1e-6;
  for(int c=0;c<12;++c){double delta[6]={0,0,0,0,0,0};delta[c%6]=h;auto ap=A,bp=B,am=A,bm=B;if(c<6)ap.Update(delta);else bp.Update(delta);delta[c%6]=-h;if(c<6)am.Update(delta);else bm.Update(delta);Vector4d rp,rm;Matrix<double,4,12>tmp;e.evaluate(ap,bp,rp,tmp);e.evaluate(am,bm,rm,tmp);num.col(c)=(rp-rm)/(2*h);}
  maxerr=std::max(maxerr,(J-num).norm()/std::max(1.,num.norm()));
  auto G=ExpSO3(Vector3d(.3,-.2,.1));auto ag=A,bg=B;ag.Rwb=G*A.Rwb;ag.twb=G*A.twb+Vector3d(2,3,4);bg.Rwb=G*B.Rwb;bg.twb=G*B.twb+Vector3d(2,3,4);Vector4d rg;Matrix<double,4,12>jg;e.evaluate(ag,bg,rg,jg);maxgauge=std::max(maxgauge,(r-rg).norm());
  VertexPose v0,v1;v0.setEstimate(A);v1.setEstimate(B);e.setVertex(0,&v0);e.setVertex(1,&v1);e.computeError();g2o::OptimizableGraph::Edge*base=&e;maxcost=std::max(maxcost,std::abs(base->chi2()-r.squaredNorm()));
  Matrix<double,30,30>H=Matrix<double,30,30>::Identity();auto hh=e.hessian();H.block<6,6>(0,0)+=hh.block<6,6>(0,0);H.block<6,6>(0,15)+=hh.block<6,6>(0,6);H.block<6,6>(15,0)+=hh.block<6,6>(6,0);H.block<6,6>(15,15)+=hh.block<6,6>(6,6);
  Matrix15d independent=H.block<15,15>(15,15)-H.block<15,15>(15,0)*H.block<15,15>(0,0).ldlt().solve(H.block<15,15>(0,15));auto actual=Optimizer::Marginalize(H,0,14);maxschur=std::max(maxschur,(independent-actual.block<15,15>(15,15)).norm()/std::max(1.,independent.norm()));++tests;
 }
 bool passed=maxerr<3e-5&&maxgauge<1e-9&&maxcost<1e-9&&maxschur<1e-6&&maxJ>1&&minPosterior>.5;
 std::cout<<std::setprecision(17)<<"{\"ownership_race_cases\":"<<ownershipCases<<",\"passed\":"<<(passed?"true":"false")<<",\"actual_production_cases\":"<<tests<<",\"analytic_vs_native_tangent_relative\":"<<maxerr<<",\"world_gauge_error\":"<<maxgauge<<",\"virtual_optimizer_cost_error\":"<<maxcost<<",\"four_block_native_schur_relative\":"<<maxschur<<",\"largest_actual_J_norm\":"<<maxJ<<",\"smallest_metric_posterior\":"<<minPosterior<<"}"<<std::endl;
 return passed?0:1;
}
