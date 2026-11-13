#include <opencv2/opencv.hpp>
#include <vector>

#pragma once
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
inline bool sourceRectifiedStereo(double stamp,const cv::Mat&left,const cv::Mat&right,const std::vector<cv::Point2f>&points,std::vector<cv::Point2f>&seeds){
 const char*path=std::getenv("VIO_RECTIFIED_STEREO_SOCKET");if(!path)return false;
 int fd=socket(AF_UNIX,SOCK_STREAM,0);sockaddr_un addr{};addr.sun_family=AF_UNIX;strncpy(addr.sun_path,path,sizeof(addr.sun_path)-1);
 if(connect(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))!=0)throw std::runtime_error("foundation stereo connection failed");
 FILE*f=fdopen(fd,"r+");fprintf(f,"%.17g %d %d %zu\n",stamp,left.cols,left.rows,points.size());
 for(auto&p:points)fprintf(f,"%.9g %.9g\n",p.x,p.y);
 cv::Mat l=left.isContinuous()?left:left.clone(),r=right.isContinuous()?right:right.clone();fwrite(l.data,1,l.total(),f);fwrite(r.data,1,r.total(),f);fflush(f);
 int n=0;if(fscanf(f,"%d",&n)!=1||n!=int(points.size())){fclose(f);throw std::runtime_error("foundation stereo protocol failed");}
 seeds.resize(n);for(auto&p:seeds)if(fscanf(f,"%f %f",&p.x,&p.y)!=2){fclose(f);throw std::runtime_error("foundation stereo point read failed");}fclose(f);return true;
}
