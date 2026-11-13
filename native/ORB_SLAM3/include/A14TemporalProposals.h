#pragma once
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <vector>
#include "ORBextractor.h"
#include <opencv2/imgproc.hpp>
namespace ORB_SLAM3 { namespace a14 {
struct Proposal{double u=0,v=0,right=-1,seedRight=-1;bool valid=false,stereo=false;};
inline bool stereoAt(const Frame*b,int j,double u,double v,double seed,double&right){
 if(std::getenv("VIO_ABLATION_RAW_STEREO_SEEDS")){
  if(j<0||j>=b->N||!std::isfinite(seed)||!std::isfinite(u)||!std::isfinite(v))return false;
  const auto&im=b->mpORBextractorLeft->mvImagePyramid[0];
  if(seed<0||seed>=im.cols||u<0||u>=im.cols||v<0||v>=im.rows||!(u-seed>0&&u-seed<b->fx))return false;
  right=seed;return true;
 }

 if(!(seed>=0)||!(b->mA15StereoCostThreshold>0)||j<0||j>=b->N)return false;
 const int level=b->mvKeysUn[j].octave,w=5,L=5;const float inv=b->mvInvScaleFactors[level],scale=b->mvScaleFactors[level];
 const cv::Mat&li=b->mpORBextractorLeft->mvImagePyramid[level];const cv::Mat&ri=b->mpORBextractorRight->mvImagePyramid[level];
 const double x=u*inv,y=v*inv,r0=seed*inv;
 auto inside=[&](const cv::Mat&im,double px){return px>=w+1&&px<im.cols-w-1&&y>=w+1&&y<im.rows-w-1;};
 if(!inside(li,x))return false;
 cv::Mat left;cv::getRectSubPix(li,cv::Size(11,11),cv::Point2f(x,y),left,CV_32F);left-=left.at<float>(w,w);
 float values[11],minimum=std::numeric_limits<float>::infinity();int best=-1;
 for(int k=-L;k<=L;++k){float cost=std::numeric_limits<float>::infinity();if(inside(ri,r0+k)){cv::Mat rr;cv::getRectSubPix(ri,cv::Size(11,11),cv::Point2f(r0+k,y),rr,CV_32F);rr-=rr.at<float>(w,w);cost=float(cv::norm(left,rr,cv::NORM_L1));}values[k+L]=cost;if(cost<minimum){minimum=cost;best=k+L;}}
 if(best<=0||best>=2*L||!(minimum<b->mA15StereoCostThreshold))return false;
 double den=2.*(values[best-1]+values[best+1]-2*values[best]);if(!std::isfinite(den)||den<=0)return false;
 double delta=(values[best-1]-values[best+1])/den;if(!std::isfinite(delta)||std::abs(delta)>1)return false;
 right=scale*(r0+best-L+delta);double disparity=u-right;
 return std::isfinite(right)&&right>=0&&disparity>0&&disparity<b->fx;
}
template<class Anchor> std::vector<Proposal> propose(const Anchor*a,const Frame*b,const std::vector<int>&matches){
 std::vector<Proposal>out(matches.size());std::vector<int>ids;
 for(size_t i=0;i<matches.size();++i)if(matches[i]>=0){out[i].u=b->mvKeysUn[matches[i]].pt.x;out[i].v=b->mvKeysUn[matches[i]].pt.y;ids.push_back(i);}
 if(!std::getenv("VIO_A14_PROPOSE")||ids.empty())return out;
 const char*path=std::getenv("VIO_RECTIFIED_STEREO_SOCKET");if(!path)throw std::runtime_error("A14 socket missing");
 int fd=socket(AF_UNIX,SOCK_STREAM,0);if(fd<0)throw std::runtime_error("A14 socket create");
 sockaddr_un addr{};addr.sun_family=AF_UNIX;if(strlen(path)>=sizeof(addr.sun_path)){close(fd);throw std::runtime_error("A14 socket path");}strcpy(addr.sun_path,path);
 if(connect(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))<0){close(fd);throw std::runtime_error("A14 socket connect");}
 FILE*f=fdopen(fd,"r+");if(!f){close(fd);throw std::runtime_error("A14 fdopen");}
 fprintf(f,"T %.17g %.17g %zu\n",a->mTimeStamp,b->mTimeStamp,ids.size());
 for(int i:ids){int j=matches[i];fprintf(f,"%d %d %.17g %.17g %.17g %.17g\n",i,j,double(a->mvKeysUn[i].pt.x),double(a->mvKeysUn[i].pt.y),double(b->mvKeysUn[j].pt.x),double(b->mvKeysUn[j].pt.y));}
 fflush(f);size_t count=0;if(fscanf(f,"%zu",&count)!=1||count!=ids.size()){fclose(f);throw std::runtime_error("A14 proposal count");}
 for(int i:ids){int valid=0;if(fscanf(f,"%lf %lf %d %lf",&out[i].u,&out[i].v,&valid,&out[i].seedRight)!=4||!std::isfinite(out[i].u)||!std::isfinite(out[i].v)){fclose(f);throw std::runtime_error("A14 proposal malformed");}out[i].valid=valid;}
 fclose(f);
 for(int i:ids){int j=matches[i];auto&v=out[i];if(!v.valid){v.right=b->mvuRight[j];v.stereo=v.right>=0;}else v.stereo=stereoAt(b,j,v.u,v.v,v.seedRight,v.right);if(!v.stereo)v.right=-1;}
 return out;
}
}}
