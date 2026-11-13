#include "RawImuInterval.h"
#include <iostream>
#include <iomanip>
#include <cmath>
int main(){using namespace ORB_SLAM3;IMU::Point l(0,0,0,0,0,0,0.),r(0,0,0,0,0,1.6f,1.6);IMU::Bias bias;IMU::Calib c(Sophus::SE3f(),1e-4f,1e-4f,1e-6f,1e-6f);IMU::Preintegrated p(bias,c);Eigen::Vector3f a,w;float dt;bool ok=measuredIntervalMean(l,r,.8,.9,a,w,dt);if(ok)p.IntegrateNewMeasurement(a,w,dt);auto R=p.GetOriginalDeltaRotation();double angle=std::atan2(R(1,0),R(0,0));std::cout<<std::setprecision(17)<<"NATIVE_RAW_INTERVAL analytic0.085 measured="<<angle<<" dt="<<dt<<std::endl;return ok && std::abs(angle-.085)<1e-6 && std::abs(dt-.1)<1e-7?0:1;}
