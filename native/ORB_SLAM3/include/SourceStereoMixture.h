#pragma once
#include <Eigen/Core>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <fstream>
namespace source_stereo {
struct Result {Eigen::Vector4d residual;Eigen::Matrix<double,4,3> transform;double classification;};
inline bool enabled(const Eigen::Vector3d&obs,double alt,double invSigma){
 if(!std::getenv("VIO_MAP_SOURCE_MIXTURE"))return false;
 bool distinct=std::isfinite(alt)&&alt>=0&&std::abs(alt-obs.z())>1e-3;
 return distinct || obs.x()-obs.z()<3*std::sqrt(2./invSigma);
}
inline void recordUse(bool alternate){const char*path=std::getenv("VIO_MAP_SOURCE_ACTIVATION");if(!path)return;static std::once_flag flags[2];std::call_once(flags[alternate?1:0],[path,alternate](){std::ofstream file(path,std::ios::app);file<<(alternate?"alternative_source_evaluated":"partial_source_evaluated")<<"\n";});}
inline Result evaluate(const Eigen::Vector3d&prediction,const Eigen::Vector3d&obs,double alt,double invSigma,double fx,double fy,double cx,double cy,bool validGeometry=true){
 Result out;out.residual.setZero();out.transform.setZero();
 const bool hasAlt=std::isfinite(alt)&&alt>=0&&std::abs(alt-obs.z())>1e-3;int N=hasAlt?4:3;
 double s=std::sqrt(invSigma),bx=std::max(1.,fx),by=std::max(1.,fy);Eigen::Vector3d r[4];Eigen::Matrix3d J[4];double u[4],ll[4];
 for(int i=0;i<N;++i){r[i].setZero();J[i].setZero();}
 Eigen::Vector3d raw=obs-prediction;r[0]=s*raw;J[0]=s*Eigen::Matrix3d::Identity();u[0]=std::log(.8/(hasAlt?2:1));
 if(hasAlt){u[0]-=std::log(bx*s)+.5*std::pow((alt-cx)/bx,2);r[1]=s*raw;r[1].z()+=s*(alt-obs.z());J[1]=J[0];u[1]=std::log(.4)-std::log(bx*s)-.5*std::pow((obs.z()-cx)/bx,2);}
 int dir=hasAlt?2:1,bg=N-1;
 double rightClutter=std::log(bx*s)+.5*std::pow((obs.z()-cx)/bx,2);if(hasAlt)rightClutter+=std::log(bx*s)+.5*std::pow((alt-cx)/bx,2);
 r[dir].head<2>()=s*raw.head<2>();J[dir](0,0)=J[dir](1,1)=s;u[dir]=std::log(.1)-rightClutter;
 u[bg]=std::log(.1)-rightClutter-std::log(bx*s)-std::log(by*s)-.5*(std::pow((obs.x()-cx)/bx,2)+std::pow((obs.y()-cy)/by,2));
 recordUse(hasAlt);if(!validGeometry||!prediction.allFinite()){out.residual.w()=std::sqrt(2*(std::log(double(N))+1-u[bg]));out.classification=1e12;return out;}
 int best=0;for(int i=0;i<N;++i){ll[i]=u[i]-.5*r[i].squaredNorm();if(ll[i]>ll[best])best=i;}
 double mass=0;for(int i=0;i<N;++i)mass+=std::exp(ll[i]-ll[best]);double lse=ll[best]+std::log(mass);Eigen::Vector3d g=Eigen::Vector3d::Zero();for(int i=0;i<N;++i)g+=std::exp(ll[i]-lse)*J[i].transpose()*r[i];
 out.residual.head<3>()=r[best];out.residual.w()=std::sqrt(std::max(1e-16,2*(std::log(double(N))+1-u[best]-(lse-ll[best]))));out.transform.topRows<3>()=J[best];out.transform.row(3)=(g-J[best].transpose()*r[best]).transpose()/out.residual.w();
 out.classification=best==bg?1e12:r[best].squaredNorm()*(best==dir?7.815/5.991:1.);return out;
}
template<class FrameType>inline double alternative(FrameType*frame,size_t index){return index<frame->mvuRightAlternative.size()?frame->mvuRightAlternative[index]:std::numeric_limits<double>::quiet_NaN();}
}
