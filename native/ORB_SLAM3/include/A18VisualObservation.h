#pragma once
#include "A16ObservationLikelihood.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
// Preserve the A16/A17 likelihood in the native pre-inertial visual graph.
// The projection and left SE(3) retraction are the native g2o convention.
namespace ORB_SLAM3 { namespace a18 {
struct Observation {
 double fx=0,fy=0,cx=0,cy=0,bf=0,sourceInvSigma=1;
 bool a16Registered=false,a16Active=false,a16Stereo=false,a16Persistent=false;
 unsigned long a16Point=0,a16Graph=0;size_t a16Feature=0;double a16Time=0;
 Eigen::Vector3d a16Observation=Eigen::Vector3d::Zero(),a16Prediction=Eigen::Vector3d::Zero();
 a16_map::Stats a16Stats;source_stereo::Result result;
 void setA16Alternative(const Eigen::Vector3d&v,bool stereo,bool active){a16Observation=v;a16Stereo=stereo;a16Active=active&&std::getenv("VIO_A18_EARLY");a16Registered=true;}
 Eigen::Vector3d project(const Eigen::Vector3d&x)const{return {fx*x.x()/x.z()+cx,fy*x.y()/x.z()+cy,fx*x.x()/x.z()+cx-bf/x.z()};}
 Eigen::Matrix<double,3,3> projectionJac(const Eigen::Vector3d&x)const{
  double z=1/x.z(),zz=z*z;Eigen::Matrix3d J;J<<fx*z,0,-fx*x.x()*zz,0,fy*z,-fy*x.y()*zz,fx*z,0,(-fx*x.x()+bf)*zz;return J;
 }
 Eigen::Matrix<double,3,6> poseJac(const Eigen::Vector3d&x)const{
  Eigen::Matrix<double,3,6>D;D<<0,x.z(),-x.y(),1,0,0,-x.z(),0,x.x(),0,1,0,x.y(),-x.x(),0,0,0,1;return -projectionJac(x)*D;
 }
 Eigen::Vector4d residual(const Eigen::Vector3d&obs,const Eigen::Vector3d&x){
  a16Prediction=project(x);
  // The legacy g2o projection rounds inverse-z to float. Preserve its exact
  // inactive objective; the active likelihood uses a coherent double projection.
  if(!a16Active){const float iz=1.0f/x.z();a16Prediction<<x.x()*iz*fx+cx,x.y()*iz*fy+cy,x.x()*iz*fx+cx-bf*iz;}
  if(a16Registered)result=a16_map::evaluate(a16Prediction,obs,a16Observation,a16Stereo,sourceInvSigma,fx,fy,cx,cy,x.z()>1e-9&&x.allFinite(),&a16Stats);
  if(a16Active)return result.residual;Eigen::Vector4d r;r<<obs-a16Prediction,0;return r;
 }
 Eigen::Matrix<double,4,3> transform()const{if(a16Active)return result.transform;Eigen::Matrix<double,4,3>T;T.setZero();T.topRows<3>().setIdentity();return T;}
 double classification(double native,const Eigen::Vector3d&obs)const{
  double value=a16Active?result.classification:native;
  if(a16Registered)a16_map::record(a16Time,a16Point,a16Feature,a16Persistent,a16Active,a16Stats,obs,a16Observation,a16Prediction,value,a16Graph);return value;
 }
};
class PoseEdge:public g2o::BaseUnaryEdge<4,Eigen::Vector3d,g2o::VertexSE3Expmap>,public Observation{
public:
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 Eigen::Vector3d Xw;
 bool read(std::istream&){return false;}bool write(std::ostream&)const{return false;}
 Eigen::Vector3d cameraPoint()const{return static_cast<const g2o::VertexSE3Expmap*>(_vertices[0])->estimate().map(Xw);}
 bool isDepthPositive(){return cameraPoint().z()>0;}
 void setInformation(const Eigen::Matrix3d&I){sourceInvSigma=I(0,0);_information.setIdentity();if(!a16Active)_information.topLeftCorner<3,3>()=I;}
 void computeError(){_error=residual(_measurement,cameraPoint());}
 void linearizeOplus(){computeError();auto x=cameraPoint();if(a16Active&&(!(x.z()>1e-9)||!x.allFinite())){_jacobianOplusXi.setZero();return;}_jacobianOplusXi=transform()*poseJac(x);}
 double classificationChi2()const{return classification(chi2(),_measurement);}
};
class PointEdge:public g2o::BaseBinaryEdge<4,Eigen::Vector3d,g2o::VertexSBAPointXYZ,g2o::VertexSE3Expmap>,public Observation{
public:
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 bool read(std::istream&){return false;}bool write(std::ostream&)const{return false;}
 Eigen::Vector3d cameraPoint()const{return static_cast<const g2o::VertexSE3Expmap*>(_vertices[1])->estimate().map(static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0])->estimate());}
 bool isDepthPositive(){return cameraPoint().z()>0;}
 void setInformation(const Eigen::Matrix3d&I){sourceInvSigma=I(0,0);_information.setIdentity();if(!a16Active)_information.topLeftCorner<3,3>()=I;}
 void computeError(){_error=residual(_measurement,cameraPoint());}
 void linearizeOplus(){computeError();auto x=cameraPoint();if(a16Active&&(!(x.z()>1e-9)||!x.allFinite())){_jacobianOplusXi.setZero();_jacobianOplusXj.setZero();return;}auto T=transform();_jacobianOplusXi=-T*projectionJac(x)*static_cast<const g2o::VertexSE3Expmap*>(_vertices[1])->estimate().rotation().toRotationMatrix();_jacobianOplusXj=T*poseJac(x);}
 double classificationChi2()const{return classification(chi2(),_measurement);}
};
}}
