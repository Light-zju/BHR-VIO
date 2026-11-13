#include <string>
#pragma once
#include "SourceStereoMixture.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <mutex>
#include <iomanip>
#include <stdexcept>
// A replacement observation likelihood, not five independent measurements.
// Existing Gaussian-mixture and MaxSum operators are not a novelty claim.
namespace a16_map {
struct Stats {std::array<double,5> posterior{};double nll=0,constant=0;int winner=0;double invSigma=0,fx=0,fy=0,cx=0,cy=0;bool stereo=false,valid=false;};
inline source_stereo::Result evaluate(const Eigen::Vector3d&prediction,const Eigen::Vector3d&obs,const Eigen::Vector3d&alternative,bool alternateStereo,double invSigma,double fx,double fy,double cx,double cy,bool validGeometry=true,Stats*stats=nullptr){
 const char*policyValue=std::getenv("VIO_ABLATION_SOURCE_POLICY");const bool frozenPolicy=policyValue&&std::string(policyValue)=="frozen";bool cachedSource=false;int cachedWinner=0;if(frozenPolicy&&stats){for(double q:stats->posterior)cachedSource=cachedSource||q!=0.;cachedWinner=stats->winner;}
 if(stats){stats->invSigma=invSigma;stats->fx=fx;stats->fy=fy;stats->cx=cx;stats->cy=cy;stats->stereo=alternateStereo;stats->valid=validGeometry&&prediction.allFinite();}
 source_stereo::Result out;out.residual.setZero();out.transform.setZero();out.classification=1e12;
 if(!(invSigma>0)||!obs.allFinite()||!alternative.allFinite())throw std::runtime_error("A16 invalid observation");
 const double s=std::sqrt(invSigma),bx=std::max(1.,fx),by=std::max(1.,fy),log2pi=std::log(2.*3.14159265358979323846);
 auto clutter=[&](double x,double center,double width){return -std::log(width)-.5*log2pi-.5*std::pow((x-center)/width,2);};
 // Missing alternative-right is integrated out: use an arbitrary common
 // dummy coordinate with its same normalized clutter in every hypothesis.
 Eigen::Vector3d alt=alternative;if(!alternateStereo)alt.z()=obs.z();
 const double cnr=clutter(obs.z(),cx,bx),car=clutter(alt.z(),cx,bx);
 const double cn=clutter(obs.x(),cx,bx)+clutter(obs.y(),cy,by)+cnr;
 const double ca=clutter(alt.x(),cx,bx)+clutter(alt.y(),cy,by)+car;
 const double norm2=2*std::log(s)-log2pi,norm3=3*std::log(s)-1.5*log2pi;
 Eigen::Vector3d r[5];Eigen::Matrix3d J[5];double u[5],ll[5];
 for(int i=0;i<5;++i){r[i].setZero();J[i].setZero();}
 r[0]=s*(obs-prediction);J[0]=s*Eigen::Matrix3d::Identity();u[0]=std::log(.4)+norm3+ca;
 r[1]=s*(alt-prediction);J[1]=s*Eigen::Matrix3d::Identity();u[1]=std::log(.4)+norm3+cn;
 if(!alternateStereo){r[1].z()=0;J[1](2,2)=0;u[1]=std::log(.4)+norm2+car+cn;}
 r[2].head<2>()=s*(obs-prediction).head<2>();J[2](0,0)=J[2](1,1)=s;u[2]=std::log(.05)+norm2+cnr+ca;
 r[3].head<2>()=s*(alt-prediction).head<2>();J[3](0,0)=J[3](1,1)=s;u[3]=std::log(.05)+norm2+car+cn;
 u[4]=std::log(.1)+cn+ca;
 const double constant=std::max(0.,*std::max_element(u,u+5))+std::log(5.)+1.;
 if(!validGeometry||!prediction.allFinite()){out.residual.w()=std::sqrt(2*(constant-u[4]));if(stats){if(!frozenPolicy){stats->posterior={0,0,0,0,1};stats->winner=4;}stats->nll=-u[4];stats->constant=constant;}return out;}
 int best=0;for(int i=0;i<5;++i){ll[i]=u[i]-.5*r[i].squaredNorm();if(ll[i]>ll[best])best=i;}

 if(const char*policy=std::getenv("VIO_ABLATION_SOURCE_POLICY")){
  if(std::string(policy)!="max"&&!frozenPolicy)throw std::runtime_error("Unknown source ablation policy");if(frozenPolicy&&cachedSource){if(cachedWinner<0||cachedWinner>4)throw std::runtime_error("Invalid cached source");best=cachedWinner;}
  out.residual.head<3>()=r[best];out.residual.w()=std::sqrt(std::max(1e-16,2*(constant-u[best])));
  out.transform.topRows<3>()=J[best];out.transform.row(3).setZero();
  const bool bearing=best==2||best==3||(best==1&&!alternateStereo);
  out.classification=best==4?1e12:r[best].squaredNorm()*(bearing?7.815/5.991:1.);
  if(stats){stats->posterior.fill(0.);stats->posterior[best]=1.;stats->winner=best;stats->nll=-ll[best];stats->constant=constant;}
  return out;
 }
 double mass=0;for(int i=0;i<5;++i)mass+=std::exp(ll[i]-ll[best]);double lse=ll[best]+std::log(mass);Eigen::Vector3d g=Eigen::Vector3d::Zero();
 for(int i=0;i<5;++i){double q=std::exp(ll[i]-lse);g+=q*J[i].transpose()*r[i];if(stats)stats->posterior[i]=q;}
 out.residual.head<3>()=r[best];out.residual.w()=std::sqrt(std::max(1e-16,2*(constant-u[best]-(lse-ll[best]))));
 out.transform.topRows<3>()=J[best];out.transform.row(3)=(g-J[best].transpose()*r[best]).transpose()/out.residual.w();
 // Native outlier classification remains a physical selected-component
 // chi-square, never the optimizer cost including an embedding constant.
 const bool bearing=best==2||best==3||(best==1&&!alternateStereo);
 out.classification=best==4?1e12:r[best].squaredNorm()*(bearing?7.815/5.991:1.);
 if(stats){stats->winner=best;stats->nll=-lse;stats->constant=constant;}
 return out;
}
inline void record(double time,unsigned long point,size_t feature,bool persistent,bool active,const Stats&stats,const Eigen::Vector3d&obs,const Eigen::Vector3d&alt,const Eigen::Vector3d&prediction,double classification,unsigned long graph){
 const char*p=std::getenv("VIO_A16_SAMPLE_LOG");if(!p||point%20)return;
 static std::mutex mutex;std::lock_guard<std::mutex>lock(mutex);std::ofstream f(p,std::ios::app);f<<std::setprecision(17)<<time<<","<<point<<","<<feature<<","<<persistent<<","<<active<<","<<stats.winner<<","<<stats.nll<<","<<classification;
 for(double v:stats.posterior)f<<","<<v;for(int i=0;i<3;++i)f<<","<<obs[i];for(int i=0;i<3;++i)f<<","<<alt[i];for(int i=0;i<3;++i)f<<","<<prediction[i];f<<","<<stats.invSigma<<","<<stats.fx<<","<<stats.fy<<","<<stats.cx<<","<<stats.cy<<","<<stats.stereo<<","<<stats.valid<<","<<graph;f<<"\n";
}

}
