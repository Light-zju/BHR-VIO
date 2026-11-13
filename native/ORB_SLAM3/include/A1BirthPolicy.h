#pragma once
#include "Frame.h"
#include "KeyFrame.h"
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <sstream>
#include <mutex>
#include <stdexcept>
namespace ORB_SLAM3 { namespace a1_birth {
inline void writeLine(const std::string&path,const std::string&line){
 static std::mutex mutex;std::lock_guard<std::mutex>lock(mutex);
 std::ofstream out(path,std::ios::app);out.write(line.data(),line.size());out.flush();
 if(!out)throw std::runtime_error("A1 atomic diagnostic write failed");
}
template<class F>inline bool neural(const F&f,size_t i){return i<f.mvA1NNOrigin.size()&&f.mvA1NNOrigin[i]!=0;}
inline bool block(const Frame&f,size_t i,int stage){
 bool nn=neural(f,i),blocked=nn&&std::getenv("VIO_A1_GEOMETRIC_BIRTH");
 if(const char*p=std::getenv("VIO_A1_BIRTH_LOG")){std::ostringstream out;out<<std::setprecision(17)<<f.mTimeStamp<<","<<f.mnId<<","<<i<<","<<stage<<","<<nn<<","<<blocked<<"\n";writeLine(std::string(p)+"_tracking.csv",out.str());}
 return blocked;
}
inline bool directStereoAllowed(const KeyFrame*f,size_t i){
 bool nn=neural(*f,i),blocked=nn&&std::getenv("VIO_A1_GEOMETRIC_BIRTH");
 if(const char*p=std::getenv("VIO_A1_BIRTH_LOG")){std::ostringstream out;out<<std::setprecision(17)<<f->mTimeStamp<<","<<f->mnId<<","<<i<<","<<nn<<","<<blocked<<"\n";writeLine(std::string(p)+"_mapping.csv",out.str());}
 return !blocked;
}
} }
