#pragma once
#include "ImuTypes.h"
#include <algorithm>
namespace ORB_SLAM3 {
inline bool measuredIntervalMean(const IMU::Point& l,const IMU::Point& r,double start,double end,Eigen::Vector3f& a,Eigen::Vector3f& w,float& dt){
 const double lo=std::max(start,l.t),hi=std::min(end,r.t),span=r.t-l.t;
 if(span<=0. || hi<=lo)return false;
 const float u0=float((lo-l.t)/span),u1=float((hi-l.t)/span);
 a=.5f*((l.a+u0*(r.a-l.a))+(l.a+u1*(r.a-l.a)));
 w=.5f*((l.w+u0*(r.w-l.w))+(l.w+u1*(r.w-l.w)));
 dt=float(hi-lo);return true;
}
}
