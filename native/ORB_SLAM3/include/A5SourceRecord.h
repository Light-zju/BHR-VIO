#pragma once
#include <Eigen/Core>
#include <array>
#include <memory>
namespace ORB_SLAM3 {
class KeyFrame;
struct A5SourceRecord {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 Eigen::Vector3d ray,obs;
 std::array<double,9>rho;
 std::array<Eigen::Vector3d,740>directions;
 std::array<double,740>inverseDistances,logWeights;
 double fx,fy,cx,cy,bf,sigma,broad;
 double sourceU,sourceV,sourceRight,sourceSigma,sourceFx,sourceFy,sourceCx,sourceCy,sourceBf;
 double alternativeU=0,alternativeV=0,alternativeRight=-1;
 bool alternativeStereo=false,completeAlternative=false;
 bool alternativeValid=false;
 bool stereo;int components=9;
 KeyFrame*host=nullptr;KeyFrame*target=nullptr;
 unsigned long graphSerial=0;
 unsigned long hostFrame=0,targetFrame=0,hostFeature=0,targetFeature=0;
};
}
