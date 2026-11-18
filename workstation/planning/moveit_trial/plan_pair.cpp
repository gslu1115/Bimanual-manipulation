#include <moveit/ompl_interface/ompl_interface.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <ompl/base/StateSampler.h>
#include <geometric_shapes/shapes.h>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <octomap/OcTree.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <chrono>
#include <atomic>

using Clock = std::chrono::steady_clock;
using State = moveit::core::RobotState;
using Context = ompl_interface::ModelBasedPlanningContextPtr;

// Bias the bidirectional tree towards the direct approach and outward postures.
// A fraction of unmodified uniform samples preserves access to the centre.
class LaneSampler : public ompl::base::StateSampler {
public:
  LaneSampler(const ompl_interface::ModelBasedStateSpacePtr& space, const State& start,
              const State& goal,double middle,std::atomic<unsigned long>& count)
    : StateSampler(space.get()), space_(space), start_(start), middle_(middle),
      sampler_(space->allocDefaultStateSampler()), count_(count) {
    a_=space_->allocState(); b_=space_->allocState(); scratch_=space_->allocState();
    space_->copyToOMPLState(a_,start);space_->copyToOMPLState(b_,goal);
  }
  ~LaneSampler(){space_->freeState(a_);space_->freeState(b_);space_->freeState(scratch_);}
  double inward(const ompl::base::State* sample) const {
    State state(start_);
    space_->copyToRobotState(state,sample);
    const auto left = state.getGlobalLinkTransform("left_panda_link4").translation().x();
    const auto right = state.getGlobalLinkTransform("right_panda_link4").translation().x();
    return std::max(0.0,left-middle_)+std::max(0.0,middle_-right);
  }
  void sampleUniform(ompl::base::State* out) override {
    ++count_;
    const auto u=rng_.uniform01();
    if(u<0.55){space_->interpolate(a_,b_,rng_.uniform01(),out);return;}
    sampler_->sampleUniform(out);
    if(u<0.90){sampler_->sampleUniform(scratch_);
      if(inward(scratch_)<inward(out))space_->copyState(out,scratch_);}
  }
  void sampleUniformNear(ompl::base::State* out,const ompl::base::State* near,double d) override {
    sampler_->sampleUniformNear(out,near,d);
  }
  void sampleGaussian(ompl::base::State* out,const ompl::base::State* mean,double d) override {
    sampler_->sampleGaussian(out,mean,d);
  }
private:
  ompl_interface::ModelBasedStateSpacePtr space_;
  State start_;
  double middle_;
  ompl::base::StateSamplerPtr sampler_;
  std::atomic<unsigned long>& count_;
  ompl::base::State *a_,*b_,*scratch_;
};

void plan(const YAML::Node& input, YAML::Node& result) {
  auto urdf = urdf::parseURDF(input["urdf"].as<std::string>());
  auto semantic = std::make_shared<srdf::Model>();
  if (!urdf || !semantic->initString(*urdf,input["srdf"].as<std::string>()))
    throw std::runtime_error("invalid dual arm description");
  auto model = std::make_shared<moveit::core::RobotModel>(urdf,semantic);
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  // Use the same sphere separation as the workstation before accepting an OMPL state.
  struct Ball { const moveit::core::LinkModel* link; Eigen::Vector3d local; double radius; int index; bool left; };
  std::vector<Ball> balls;
  for(const auto* link:model->getLinkModelsWithCollisionGeometry()){
    const auto& name=link->getName();
    int index=name.find("finger_envelope")!=std::string::npos ? 9 :
        name.find("panda_hand")!=std::string::npos ? 8 : name.back()-'0';
    for(size_t k=0;k<link->getShapes().size();++k){
      auto sphere=std::dynamic_pointer_cast<const shapes::Sphere>(link->getShapes()[k]);
      if(!sphere) throw std::runtime_error("unexpected body primitive");
      balls.push_back({link,link->getCollisionOriginTransforms()[k].translation(),sphere->radius,index,
                       name.rfind("left_",0)==0});
    }
  }
  auto separated=[balls](const State& state,bool){
    std::vector<Eigen::Vector3d> centres;
    for(const auto& b:balls) centres.push_back(state.getGlobalLinkTransform(b.link)*b.local);
    for(size_t i=0;i<balls.size();++i) for(size_t j=0;j<i;++j){
      const auto& a=balls[i];const auto& b=balls[j];
      if(a.left==b.left && std::abs(a.index-b.index)<=2) continue;
      const double limit=a.radius+b.radius+0.008;
      if((centres[i]-centres[j]).squaredNorm()<limit*limit) return false;
    }
    return true;
  };
  scene->setStateFeasibilityPredicate(separated);
  const auto* group = model->getJointModelGroup("both_arms");
  auto names = input["joint_names"].as<std::vector<std::string>>();
  if (!group || names.size()!=14 || group->getVariableCount()!=14)
    throw std::runtime_error("expected 14 planning variables");
  auto q0=input["start"].as<std::vector<double>>(), q1=input["goal"].as<std::vector<double>>();
  if(q0.size()!=14 || q1.size()!=14) throw std::runtime_error("bad endpoints");
  auto& start=scene->getCurrentStateNonConst();
  for(size_t j=0;j<names.size();++j) start.setVariablePosition(names[j],q0[j]);
  start.update();
  auto goal=start;
  for(size_t j=0;j<names.size();++j) goal.setVariablePosition(names[j],q1[j]);
  goal.update();
  for(const auto& box:input["boxes"]){
    const auto size=box["size"].as<std::vector<double>>(), centre=box["centre"].as<std::vector<double>>();
    Eigen::Isometry3d pose=Eigen::Isometry3d::Identity();
    pose.translation()=Eigen::Vector3d(centre.at(0),centre.at(1),centre.at(2));
    auto name=box["name"].as<std::string>();
    scene->getWorldNonConst()->addToObject(name,std::make_shared<shapes::Box>(size.at(0),size.at(1),size.at(2)),pose);
    for(const auto& side: {"left","right"})
      if(name=="panda_"+std::string(side)+"_mount")
        scene->getAllowedCollisionMatrixNonConst().setEntry(std::string(side)+"_panda_link0",name,true);
  }
  auto tree=std::make_shared<octomap::OcTree>(input["resolution"].as<double>());
  for(const auto& xyz:input["occupied"]) tree->updateNode(octomap::point3d(xyz[0].as<float>(),xyz[1].as<float>(),xyz[2].as<float>()),true,true);
  tree->updateInnerOccupancy();
  if(tree->size()) scene->getWorldNonConst()->addToObject("observed_depth",
      std::make_shared<shapes::OcTree>(tree),Eigen::Isometry3d::Identity());
  for(auto item: {std::make_pair("start",&start),std::make_pair("goal",&goal)}){
    if(!item.second->satisfiesBounds(group)) throw std::runtime_error(std::string(item.first)+" outside joint limits");
    if(!separated(*item.second,false)) throw std::runtime_error(std::string(item.first)+" below 8 mm robot clearance");
    collision_detection::CollisionRequest request;
    request.contacts=true; request.max_contacts=16;
    collision_detection::CollisionResult collision;
    scene->checkCollision(request,collision,*item.second);
    if(collision.collision){
      for(auto& [links,contacts]:collision.contacts)
        result["contacts"].push_back(links.first+" / "+links.second);
      throw std::runtime_error(std::string(item.first)+" in collision");
    }
  }
  auto node=rclcpp::Node::make_shared("fourbox_pair_planner");
  planning_interface::PlannerConfigurationSettings settings;
  settings.name=settings.group="both_arms";
  settings.config={{"type","geometric::RRTConnect"},{"longest_valid_segment_fraction","0.001"}};
  ompl_interface::OMPLInterface planner(model,{{settings.name,settings}},node,"ompl");
  planning_interface::MotionPlanRequest req;
  req.group_name=settings.group; req.allowed_planning_time=2.0; req.num_planning_attempts=1;
  moveit::core::robotStateToRobotStateMsg(start,req.start_state);
  req.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(goal,group,0.00001));
  auto context=planner.getPlanningContext(scene,req);
  if(!context) throw std::runtime_error("OMPL context unavailable");
  std::atomic<unsigned long> samples{0};
  auto space=context->getOMPLStateSpace();
  space->setStateSamplerAllocator([&,middle=input["midline_x"].as<double>()](const auto*){
      return std::make_shared<LaneSampler>(space,start,goal,middle,samples);});
  auto begin=Clock::now();
  bool direct=input["prefer_direct"].as<bool>(true);
  const double distance=(Eigen::Map<Eigen::VectorXd>(q1.data(),14)-
                         Eigen::Map<Eigen::VectorXd>(q0.data(),14)).norm();
  // Bound the full 14-D step, also respecting the planner's configured extent fraction.
  const double step=std::min(0.005,space->getMaximumExtent()*0.001);
  const int segments=std::max(1,static_cast<int>(std::ceil(distance/step)));
  State interpolated(start);
  for(int i=1;direct && i<segments;++i){
    start.interpolate(goal,static_cast<double>(i)/segments,interpolated);
    interpolated.update();
    direct=interpolated.satisfiesBounds(group) && separated(interpolated,false) &&
           !scene->isStateColliding(interpolated);
  }
  result["direct_check_s"]=std::chrono::duration<double>(Clock::now()-begin).count();
  planning_interface::MotionPlanResponse solution;
  if(direct){
    solution.trajectory=std::make_shared<robot_trajectory::RobotTrajectory>(model,group);
    solution.trajectory->addSuffixWayPoint(start,0.);
    solution.trajectory->addSuffixWayPoint(goal,0.);
  }else{
    begin=Clock::now();
    context->solve(solution);
    if(!solution.error_code || !solution.trajectory) throw std::runtime_error("no joint path");
  }
  result["route"]=direct?"direct":"rrt_connect";
  result["search_s"]=direct?0.:std::chrono::duration<double>(Clock::now()-begin).count();
  result["sampler_calls"]=samples.load();
  auto& trajectory=*solution.trajectory;
  trajectory_processing::TimeOptimalTrajectoryGeneration timing(0.001,0.02);
  std::unordered_map<std::string,double> acceleration;
  auto limits=input["acceleration_limits"].as<std::vector<double>>();
  if(limits.size()!=7) throw std::runtime_error("expected Panda acceleration limits");
  for(size_t i=0;i<names.size();++i) acceleration[names[i]]=limits[i%7];
  if(!timing.computeTimeStamps(trajectory,{},acceleration,1.0,1.0)) throw std::runtime_error("time parameterization failed");
  for(size_t i=0;i<trajectory.getWayPointCount();++i){
    const auto& state=trajectory.getWayPoint(i);
    if(scene->isStateColliding(state)||!separated(state,false)) throw std::runtime_error("timed waypoint in collision");
    std::vector<double> row;
    for(const auto& name:names) row.push_back(state.getVariablePosition(name));
    result["positions"].push_back(row);
    result["times_s"].push_back(trajectory.getWayPointDurationFromStart(i));
  }
}
int main(int argc,char** argv){
  if(argc!=3) return 2;
  rclcpp::init(argc,argv);
  YAML::Node result;
  try {plan(YAML::LoadFile(argv[1]),result); result["ok"]=true;}
  catch(const std::exception& error){result["ok"]=false;result["error"]=error.what();}
  std::ofstream out(argv[2]);out<<result;
  rclcpp::shutdown();
  return result["ok"].as<bool>() ? 0 : 1;
}
