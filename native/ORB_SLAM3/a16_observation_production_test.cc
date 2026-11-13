#include "G2oTypes.h"
#include "CameraModels/Pinhole.h"
#include <iostream>
#include <iomanip>
using namespace Eigen;using namespace ORB_SLAM3;
ImuCamPose makePose(Pinhole*cam){ImuCamPose p;p.Rwb=ExpSO3(Vector3d(.02,-.01,.005));p.twb=Vector3d(.2,-.1,.3);p.Rbc={ExpSO3(Vector3d(.01,-.02,.005))};p.tbc={Vector3d(.06,.01,.02)};p.Rcb={p.Rbc[0].transpose()};p.tcb={-p.Rcb[0]*p.tbc[0]};p.Rcw={p.Rcb[0]*p.Rwb.transpose()};p.tcw={-p.Rcw[0]*p.twb+p.tcb[0]};p.pCamera={cam};p.bf=200;p.its=0;return p;}
void vec(const Vector3d&x){std::cout<<"["<<x[0]<<","<<x[1]<<","<<x[2]<<"]";}
int main(){Pinhole camera(std::vector<float>{400,410,320,240});double maxJ=0,maxH=0,maxCost=0,maxGrad=0;int count=0;
 for(int mode=0;mode<4;++mode)for(int k=0;k<100;++k){
  auto A=makePose(&camera);Vector3d point(.1+.01*k,-.15+.002*k,2.+.2*k),pred=A.ProjectStereo(point,0);double inv=1./(1.+.01*k);
  Vector3d obs=pred+Vector3d(.1+.001*k,-.08,.12),alt=obs+Vector3d(.37,-.23,.29);bool stereo=mode!=2;if(mode==0)alt=obs;if(mode==3){obs+=Vector3d(100,100,100);alt=obs+Vector3d(.37,-.23,.29);}
  VertexPose vp;vp.setEstimate(A);g2o::VertexSBAPointXYZ vx;vx.setEstimate(point);EdgeStereo e;e.setVertex(0,&vx);e.setVertex(1,&vp);e.setMeasurement(obs);e.setA16Alternative(alt,stereo,true);e.setInformation(Matrix3d::Identity()*inv);e.computeError();g2o::JacobianWorkspace workspace;workspace.updateSize(&e);workspace.allocate();static_cast<g2o::OptimizableGraph::Edge*>(&e)->linearizeOplus(workspace);auto analytic=e.GetJacobian();Matrix<double,4,9>num;double h=1e-6;
  for(int c=0;c<9;++c){Vector4d rr[2];for(int sign=-1;sign<=1;sign+=2){auto B=A;Vector3d p=point;if(c<3)p[c]+=sign*h;else{double delta[6]={};delta[c-3]=sign*h;B.Update(delta);}vp.setEstimate(B);vx.setEstimate(p);e.computeError();rr[(sign+1)/2]=e.error();}num.col(c)=(rr[1]-rr[0])/(2*h);}
  vp.setEstimate(A);vx.setEstimate(point);e.computeError();maxJ=std::max(maxJ,(analytic-num).norm()/std::max(1.,num.norm()));
  g2o::OptimizableGraph::Edge*base=&e;maxCost=std::max(maxCost,std::abs(base->chi2()-2*(e.a16Stats.constant+e.a16Stats.nll)));
  // Actual pose-only production edge and native body tangent.
  EdgeStereoOnlyPose unary(point.cast<float>());unary.setVertex(0,&vp);unary.setMeasurement(obs);unary.setA16Alternative(alt,stereo,true);unary.setInformation(Matrix3d::Identity()*inv);unary.computeError();g2o::JacobianWorkspace unaryWorkspace;unaryWorkspace.updateSize(&unary);unaryWorkspace.allocate();static_cast<g2o::OptimizableGraph::Edge*>(&unary)->linearizeOplus(unaryWorkspace);Matrix<double,4,6>nj;
  for(int c=0;c<6;++c){Vector4d rr[2];for(int sign=-1;sign<=1;sign+=2){auto B=A;double delta[6]={};delta[c]=sign*h;B.Update(delta);vp.setEstimate(B);unary.computeError();rr[(sign+1)/2]=unary.error();}nj.col(c)=(rr[1]-rr[0])/(2*h);}vp.setEstimate(A);unary.computeError();Matrix<double,6,6>H=unary.GetHessian();maxH=std::max(maxH,(H-nj.transpose()*nj).norm()/std::max(1.,H.norm()));
  a16_map::Stats stats;auto z=a16_map::evaluate(pred,obs,alt,stereo,inv,400,410,320,240,true,&stats);Vector3d grad=-z.transform.transpose()*z.residual,ng;
  for(int c=0;c<3;++c){auto pp=pred,pm=pred;pp[c]+=h;pm[c]-=h;a16_map::Stats sp,sm;a16_map::evaluate(pp,obs,alt,stereo,inv,400,410,320,240,true,&sp);a16_map::evaluate(pm,obs,alt,stereo,inv,400,410,320,240,true,&sm);ng[c]=(sp.nll-sm.nll)/(2*h);}maxGrad=std::max(maxGrad,(grad-ng).norm()/std::max(1.,ng.norm()));
  std::cout<<std::setprecision(17)<<"FIXTURE {\"mode\":"<<mode<<",\"prediction\":";vec(pred);std::cout<<",\"obs\":";vec(obs);std::cout<<",\"alt\":";vec(alt);std::cout<<",\"stereo\":"<<(stereo?"true":"false")<<",\"invSigma\":"<<inv<<",\"nll\":"<<stats.nll<<",\"cost\":"<<z.residual.squaredNorm()<<",\"constant\":"<<stats.constant<<",\"gradient\":";vec(grad);std::cout<<",\"posterior\":[";for(int j=0;j<5;++j)std::cout<<(j?",":"")<<stats.posterior[j];std::cout<<"]}\n";++count;
 }
 bool pass=maxJ<3e-5&&maxH<3e-5&&maxCost<1e-9&&maxGrad<1e-6;std::cout<<std::setprecision(17)<<"{\"passed\":"<<(pass?"true":"false")<<",\"actual_cases\":"<<count<<",\"point_pose_J_relative\":"<<maxJ<<",\"pose_only_H_relative\":"<<maxH<<",\"virtual_cost_error\":"<<maxCost<<",\"gradient_relative\":"<<maxGrad<<"}\n";return pass?0:1;
}
