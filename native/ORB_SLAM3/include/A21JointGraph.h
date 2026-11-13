#pragma once
#include <Eigen/Core>
#include <fstream>
#include <map>
#include <vector>
#include <iomanip>
#include <chrono>
// Read-only reconstruction of the accepted graph's robust GN blocks.
// Fixed vertices remain conditioned, as in the actual native optimization.
namespace a21_joint {
struct Point { Eigen::Matrix3d H=Eigen::Matrix3d::Zero(); Eigen::MatrixXd B; };
struct Graph {
 std::map<int,int> offsets; std::vector<std::pair<int,int>> layout;
 Eigen::MatrixXd H; std::map<unsigned long,Point> points; int D=0;
 void vertex(g2o::OptimizableGraph::Vertex*v){
  if(!v||v->fixed()||offsets.count(v->id()))return;
  offsets[v->id()]=D;layout.emplace_back(v->id(),v->dimension());D+=v->dimension();
 }
 void start(){H=Eigen::MatrixXd::Zero(D,D);}
 template<class E> double weight(E*e){e->computeError();if(!e->robustKernel())return 1.;Eigen::Vector3d rho;e->robustKernel()->robustify(e->chi2(),rho);return rho[1];}
 template<class E> void state(E*e,const std::vector<int>&dims){
  if(!e||e->level()!=0)return; Eigen::MatrixXd h=e->GetHessian()*weight(e);int r=0;
  for(size_t i=0;i<dims.size();++i){auto ai=offsets.find(e->vertex(i)->id());int c=0;
   for(size_t j=0;j<dims.size();++j){auto aj=offsets.find(e->vertex(j)->id());
    if(ai!=offsets.end()&&aj!=offsets.end())H.block(ai->second,aj->second,dims[i],dims[j])+=h.block(r,c,dims[i],dims[j]);c+=dims[j];}r+=dims[i];}
 }
 template<class E> void visual(E*e,unsigned long id){
  if(e->level()!=0)return;Eigen::Matrix<double,9,9> h=e->GetHessian()*weight(e);
  auto&p=points[id];if(p.B.size()==0)p.B=Eigen::MatrixXd::Zero(3,D);p.H+=h.topLeftCorner<3,3>();
  auto k=offsets.find(e->vertex(1)->id());if(k!=offsets.end()){
   p.B.block(0,k->second,3,6)+=h.topRightCorner<3,6>();H.block(k->second,k->second,6,6)+=h.bottomRightCorner<6,6>();}
 }
 void save(const std::string&base,double t,unsigned long k,unsigned long velocity){
  std::ofstream meta(base+".meta");meta<<std::setprecision(17)<<t<<" "<<k<<" "<<D<<" "<<points.size()<<" "<<velocity<<"\n";
  for(auto x:layout)meta<<x.first<<" "<<x.second<<"\n";
  std::ofstream f(base+".bin",std::ios::binary);f.write((char*)H.data(),sizeof(double)*H.size());
  for(auto&x:points){meta<<x.first<<"\n";f.write((char*)x.second.H.data(),9*sizeof(double));f.write((char*)x.second.B.data(),sizeof(double)*x.second.B.size());}
  f.close();meta.close();
 }
}
;
}
