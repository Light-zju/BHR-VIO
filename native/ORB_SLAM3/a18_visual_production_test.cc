#include "A18VisualObservation.h"
#include "Thirdparty/g2o/g2o/core/jacobian_workspace.h"
#include <iostream>
#include <random>
int main(){
 setenv("VIO_A18_EARLY","1",1);
 std::mt19937 rng(1718);std::uniform_real_distribution<double>d(-1,1);double worst=0,nativeWorst=0;
 for(int k=0;k<400;++k){
  g2o::VertexSE3Expmap pose;g2o::VertexSBAPointXYZ point;
  Eigen::Matrix<double,6,1>tw;for(int j=0;j<6;++j)tw[j]=.1*d(rng);
  auto original=g2o::SE3Quat::exp(tw);pose.setEstimate(original);Eigen::Vector3d xyz(d(rng),d(rng),4+2*d(rng));point.setEstimate(xyz);
  ORB_SLAM3::a18::PoseEdge u;ORB_SLAM3::a18::PointEdge b;
  u.setVertex(0,&pose);u.Xw=xyz;b.setVertex(0,&point);b.setVertex(1,&pose);
  for(auto*o:{static_cast<ORB_SLAM3::a18::Observation*>(&u),static_cast<ORB_SLAM3::a18::Observation*>(&b)}){o->fx=500;o->fy=490;o->cx=320;o->cy=240;o->bf=40;}
  auto pred=u.project(original.map(xyz));Eigen::Vector3d obs=pred+Eigen::Vector3d(d(rng),d(rng),d(rng));auto alt=obs+Eigen::Vector3d(.3*d(rng),.3*d(rng),.3*d(rng));
  u.setMeasurement(obs);b.setMeasurement(obs);u.setA16Alternative(alt,k%2,k%3!=0);b.setA16Alternative(alt,k%2,k%3!=0);u.setInformation(Eigen::Matrix3d::Identity()*1.7);b.setInformation(Eigen::Matrix3d::Identity()*1.7);
  g2o::JacobianWorkspace wu,wb;wu.updateSize(&u);wu.allocate();wb.updateSize(&b);wb.allocate();
  static_cast<g2o::OptimizableGraph::Edge*>(&u)->linearizeOplus(wu);static_cast<g2o::OptimizableGraph::Edge*>(&b)->linearizeOplus(wb);
  Eigen::Matrix<double,4,6>Ju=u.jacobianOplusXi(),Jbp=b.jacobianOplusXj();Eigen::Matrix<double,4,3>Jbx=b.jacobianOplusXi();
  double eps=1e-7;
  for(int j=0;u.a16Active&&j<6;++j){Eigen::Matrix<double,6,1>v=Eigen::Matrix<double,6,1>::Zero();v[j]=eps;pose.setEstimate(g2o::SE3Quat::exp(v)*original);u.computeError();b.computeError();auto up=u.error().eval(),bp=b.error().eval();pose.setEstimate(g2o::SE3Quat::exp(-v)*original);u.computeError();b.computeError();auto un=u.error().eval(),bn=b.error().eval();worst=std::max(worst,((up-un)/(2*eps)-Ju.col(j)).norm()/(1+Ju.col(j).norm()));worst=std::max(worst,((bp-bn)/(2*eps)-Jbp.col(j)).norm()/(1+Jbp.col(j).norm()));}
  pose.setEstimate(original);
  for(int j=0;b.a16Active&&j<3;++j){auto x=xyz;x[j]+=eps;point.setEstimate(x);b.computeError();auto p=b.error().eval();x=xyz;x[j]-=eps;point.setEstimate(x);b.computeError();auto n=b.error().eval();worst=std::max(worst,((p-n)/(2*eps)-Jbx.col(j)).norm()/(1+Jbx.col(j).norm()));}point.setEstimate(xyz);
  if(!u.a16Active){g2o::EdgeStereoSE3ProjectXYZOnlyPose native;native.setVertex(0,&pose);native.Xw=xyz;native.fx=u.fx;native.fy=u.fy;native.cx=u.cx;native.cy=u.cy;native.bf=u.bf;native.setMeasurement(obs);native.setInformation(Eigen::Matrix3d::Identity()*1.7);native.computeError();u.computeError();g2o::JacobianWorkspace w;w.updateSize(&native);w.allocate();static_cast<g2o::OptimizableGraph::Edge*>(&native)->linearizeOplus(w);nativeWorst=std::max(nativeWorst,(Ju.topRows<3>()-native.jacobianOplusXi()).norm());nativeWorst=std::max(nativeWorst,std::abs(native.chi2()-u.chi2()));}
 }
 std::cout<<"cases=400 pose_point_relative="<<worst<<" native_difference="<<nativeWorst<<std::endl;
 return worst<1e-5&&nativeWorst<1e-9?0:1;
}
