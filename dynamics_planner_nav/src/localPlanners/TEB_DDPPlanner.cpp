// TEB_DDPPlanner.cpp — Hybrid planner combining TEB and DDP
// This version uses TEB as primary planner with DDP fallback

#include "localPlanners/TEB_DDPPlanner.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/Marker.h>
#include <teb_local_planner/robot_footprint_model.h>

namespace Antipatrea {

    TEB_DDPPlanner::TEB_DDPPlanner() : teb_failure_count_(0), use_ddp_fallback_(false) {
        // Initialize TEB configuration matching TebLocalPlannerROS parameters
        teb_config_.optim.no_inner_iterations = 4;
        teb_config_.optim.no_outer_iterations = 2;
        teb_config_.optim.penalty_epsilon = 0.05;

        // Robot kinematics
        teb_config_.robot.max_vel_x = 0.5;
        teb_config_.robot.max_vel_x_backwards = 0.25;
        teb_config_.robot.max_vel_theta = 1.8;
        teb_config_.robot.acc_lim_x = 2.5;
        teb_config_.robot.acc_lim_theta = 3.5;

        // Goal tolerance
        teb_config_.goal_tolerance.xy_goal_tolerance = 0.2;
        teb_config_.goal_tolerance.yaw_goal_tolerance = 0.1;
        teb_config_.goal_tolerance.free_goal_vel = false;

        // Obstacles
        teb_config_.obstacles.min_obstacle_dist = 0.15;
        teb_config_.obstacles.inflation_dist = 0.1;
        teb_config_.obstacles.include_costmap_obstacles = false;
        teb_config_.obstacles.costmap_obstacles_behind_robot_dist = 0.5;
        teb_config_.obstacles.include_dynamic_obstacles = false;

        // Trajectory
        teb_config_.trajectory.teb_autosize = true;
        teb_config_.trajectory.dt_ref = 0.25;
        teb_config_.trajectory.dt_hysteresis = 0.05;
        teb_config_.trajectory.min_samples = 3;
        teb_config_.trajectory.max_samples = 20;
        teb_config_.trajectory.global_plan_overwrite_orientation = true;
        teb_config_.trajectory.feasibility_check_no_poses = 2;

        // Weights
        teb_config_.optim.weight_max_vel_x = 2.0;
        teb_config_.optim.weight_max_vel_theta = 1.0;
        teb_config_.optim.weight_acc_lim_x = 1.0;
        teb_config_.optim.weight_acc_lim_theta = 1.0;
        teb_config_.optim.weight_kinematics_nh = 100.0;
        teb_config_.optim.weight_kinematics_forward_drive = 1.0;
        teb_config_.optim.weight_kinematics_turning_radius = 1.0;
        teb_config_.optim.weight_optimaltime = 2.0;
        teb_config_.optim.weight_obstacle = 30.0;
        teb_config_.optim.weight_inflation = 0.2;
        teb_config_.optim.weight_dynamic_obstacle = 10.0;

        // Create obstacle container
        obstacles_ = boost::make_shared<teb_local_planner::ObstContainer>();

        // Create robot footprint model - polygon matching TebLocalPlannerROS
        // vertices: [[-0.254, -0.215], [-0.254, 0.215], [0.254, 0.215], [0.254, -0.215]]
        teb_local_planner::Point2dContainer footprint_vertices;
        footprint_vertices.push_back(Eigen::Vector2d(-0.254, -0.215));
        footprint_vertices.push_back(Eigen::Vector2d(-0.254, 0.215));
        footprint_vertices.push_back(Eigen::Vector2d(0.254, 0.215));
        footprint_vertices.push_back(Eigen::Vector2d(0.254, -0.215));
        robot_model_ = boost::make_shared<teb_local_planner::PolygonRobotFootprint>(footprint_vertices);

        // Create TEB planner instance
        teb_planner_ = boost::make_shared<teb_local_planner::TebOptimalPlanner>(
            teb_config_,
            obstacles_.get(),
            robot_model_,
            teb_local_planner::TebVisualizationPtr(),
            nullptr  // No via points
        );

        ROS_INFO("TEB_DDPPlanner initialized - TEB with DDP fallback");
    }

    void TEB_DDPPlanner::updateRobotState() {
        // Thread-safe read: get snapshot of current state
        auto current_state = robot->getPoseStateSafe();

        parent = {0, 0, 0, current_state.velocity_, current_state.angular_velocity_, true};
        parent_odom = current_state;
    }

    bool TEB_DDPPlanner::Solve(int nrIters, double dt, bool &canBeSolved) {
        geometry_msgs::Twist cmd_vel;
        if (!robot) return false;

        updateRobotState();

        commonParameters(*robot);

        switch (robot->getRobotState()) {
            case Robot_config::NO_MAP_PLANNING:
                return handleNoMapPlanning(cmd_vel);

            case Robot_config::NORMAL_PLANNING:
                return handleNormalSpeedPlanning(cmd_vel, dt);

            case Robot_config::LOW_SPEED_PLANNING:
                return handleLowSpeedPlanning(cmd_vel, dt);

            default:
                return handleAbnormalPlaning(cmd_vel, dt);
        }
    }

    void TEB_DDPPlanner::commonParameters(Robot_config &robot) {
        local_goal = robot.getLocalGoalCfg();
        global_paths = robot.global_paths;

        // Update TEB config from robot config
        teb_config_.robot.max_vel_x = robot.max_vel_x;
        teb_config_.robot.max_vel_theta = robot.max_vel_theta;
    }

    void TEB_DDPPlanner::normalParameters(Robot_config &robot) {
        teb_config_.robot.max_vel_x = robot.max_vel_x;
        teb_config_.optim.weight_obstacle = 30.0;
    }

    void TEB_DDPPlanner::lowSpeedParameters(Robot_config &robot) {
        teb_config_.robot.max_vel_x = std::min(robot.max_vel_x, 0.75);
        teb_config_.optim.weight_obstacle = 50.0;  // More conservative
    }

    bool TEB_DDPPlanner::handleNoMapPlanning(geometry_msgs::Twist &cmd_vel) {
        normalParameters(*robot);

        const double angle_to_goal = std::atan2(
            robot->getGlobalGoalCfg()[1] - parent_odom.y_,
            robot->getGlobalGoalCfg()[0] - parent_odom.x_
        );

        double angular = std::clamp(normalizeAngle(angle_to_goal - parent_odom.theta_), -1.0, 1.0);
        angular = (angular > 0) ? std::max(angular, 0.1) : std::min(angular, -0.1);

        publishCommand(cmd_vel, robot->max_vel_x, angular);
        return true;
    }

    bool TEB_DDPPlanner::handleNormalSpeedPlanning(geometry_msgs::Twist &cmd_vel, double dt) {
        normalParameters(*robot);

        // Try TEB first
        bool success = callTEBPlanner(cmd_vel);

        if (!success) {
            teb_failure_count_++;
            ROS_WARN("TEB planning failed (count: %d), trying DDP fallback", teb_failure_count_);

            // Use DDP as fallback
            success = callDDPPlanner(cmd_vel, dt);

            if (!success) {
                // Both failed - stop
                ROS_WARN("Both TEB and DDP failed, stopping");
                publishCommand(cmd_vel, 0, 0);
            } else {
                use_ddp_fallback_ = true;
            }
        } else {
            // TEB succeeded - reset failure count
            if (teb_failure_count_ > 0) {
                ROS_INFO("TEB recovered after %d failures", teb_failure_count_);
            }
            teb_failure_count_ = 0;
            use_ddp_fallback_ = false;
        }

        return true;
    }

    bool TEB_DDPPlanner::handleLowSpeedPlanning(geometry_msgs::Twist &cmd_vel, double dt) {
        lowSpeedParameters(*robot);

        // Try TEB first
        bool success = callTEBPlanner(cmd_vel);

        if (!success) {
            teb_failure_count_++;
            ROS_WARN("TEB planning failed (low speed), trying DDP fallback");

            // Use DDP as fallback
            success = callDDPPlanner(cmd_vel, dt);

            if (!success) {
                ROS_WARN("Both TEB and DDP failed, stopping");
                publishCommand(cmd_vel, 0, 0);
            }
        } else {
            teb_failure_count_ = 0;
        }

        return true;
    }

    bool TEB_DDPPlanner::handleAbnormalPlaning(geometry_msgs::Twist &cmd_vel, double dt) {
        if (robot->getRobotState() == Robot_config::BRAKE_PLANNING) {
            auto current_state = robot->getPoseStateSafe();
            if (current_state.velocity_ > 0.01) {
                publishCommand(cmd_vel, -0.1, 0.0);
            } else {
                publishCommand(cmd_vel, 0.0, 0.0);
                robot->setRobotState(Robot_config::RECOVERY);
            }
            return true;
        }

        if (robot->getRobotState() == Robot_config::ROTATE_PLANNING) {
            auto current_state = robot->getPoseStateSafe();
            double angle = normalizeAngle(robot->rotating_angle - current_state.theta_);

            if (fabs(angle) <= 0.10) {
                robot->setRobotState(Robot_config::NORMAL_PLANNING);
                return true;
            }

            double z = angle > 0 ? std::min(angle, 1.0) : std::max(angle, -1.0);
            z = z > 0 ? std::max(z, 0.5) : std::min(z, -0.5);
            publishCommand(cmd_vel, 0.0, z);
            return true;
        }

        if (robot->getRobotState() == Robot_config::RECOVERY) {
            double front_obs_snapshot, latter_obs_snapshot;
            robot->getObstacleDistanceSafe(front_obs_snapshot, latter_obs_snapshot);

            if (front_obs_snapshot <= 0.10) {
                robot->setRobotState(Robot_config::BACKWARD);
                return true;
            }

            // Simple recovery: rotate in place
            robot->rotating_angle = normalizeAngle(parent_odom.theta_ + M_PI / 4);
            robot->setRobotState(Robot_config::ROTATE_PLANNING);
            return true;
        }

        if (robot->getRobotState() == Robot_config::BACKWARD) {
            double front_obs_snapshot, latter_obs_snapshot;
            robot->getObstacleDistanceSafe(front_obs_snapshot, latter_obs_snapshot);

            if (front_obs_snapshot >= 0.10) {
                robot->setRobotState(Robot_config::RECOVERY);
                return true;
            }

            publishCommand(cmd_vel, -0.3, 0);
        }

        return true;
    }

    bool TEB_DDPPlanner::callTEBPlanner(geometry_msgs::Twist &cmd_vel) {
        // 1. Build global plan from local_goal
        std::vector<geometry_msgs::PoseStamped> global_plan;
        buildGlobalPlan(global_plan);

        if (global_plan.size() < 2) {
            ROS_WARN("Global plan too short for TEB");
            return false;
        }

        // 2. Get current robot velocity
        geometry_msgs::Twist robot_vel;
        robot_vel.linear.x = parent_odom.velocity_;
        robot_vel.angular.z = parent_odom.angular_velocity_;

        // 3. Clear and rebuild obstacles
        obstacles_->clear();
        buildObstacles();

        // 4. Plan!
        bool success = teb_planner_->plan(
            global_plan,
            &robot_vel,
            false  // Don't free goal velocity
        );

        if (!success) {
            ROS_WARN("TEB optimization failed");
            return false;
        }

        // 5. Extract velocity command (4 parameters: vx, vy, omega, look_ahead_poses)
        double vx, vy, omega;
        success = teb_planner_->getVelocityCommand(vx, vy, omega, 1);

        if (!success) {
            ROS_WARN("Failed to get velocity from TEB");
            return false;
        }

        // 6. Set command velocity (differential drive: ignore vy)
        cmd_vel.linear.x = vx;
        cmd_vel.angular.z = omega;

        // 7. Publish command
        robot->Control().publish(cmd_vel);

        // 8. Visualize trajectory (optional)
        visualizeTEBTrajectory();

        return true;
    }

    bool TEB_DDPPlanner::callDDPPlanner(geometry_msgs::Twist &cmd_vel, double dt) {
        // TODO: Implement DDP fallback logic
        // This is a placeholder for DDP integration
        // You would need to instantiate and call a DDP solver here

        ROS_WARN("DDP fallback not yet implemented");
        return false;
    }

    void TEB_DDPPlanner::buildGlobalPlan(std::vector<geometry_msgs::PoseStamped> &plan) {
        plan.clear();

        // Start pose (current robot pose in odom frame)
        geometry_msgs::PoseStamped start;
        start.header.frame_id = "odom";
        start.header.stamp = ros::Time::now();
        start.pose.position.x = parent_odom.x_;
        start.pose.position.y = parent_odom.y_;

        tf2::Quaternion q_start;
        q_start.setRPY(0, 0, parent_odom.theta_);
        start.pose.orientation = tf2::toMsg(q_start);

        plan.push_back(start);

        double cos_theta = std::cos(parent_odom.theta_);
        double sin_theta = std::sin(parent_odom.theta_);

        // Use global_paths from global planner (in baseline frame, transform to odom)
        if (!global_paths.empty()) {
            // Add ALL points from global_paths to give TEB the full global plan
            // TEB will optimize around this reference path
            for (size_t i = 0; i < global_paths.size(); ++i) {
                geometry_msgs::PoseStamped waypoint;
                waypoint.header = start.header;

                // Transform from baseline to odom
                double x_baseline = global_paths[i][0];
                double y_baseline = global_paths[i][1];
                waypoint.pose.position.x = parent_odom.x_ + (x_baseline * cos_theta - y_baseline * sin_theta);
                waypoint.pose.position.y = parent_odom.y_ + (x_baseline * sin_theta + y_baseline * cos_theta);

                // Compute orientation from path direction
                if (i + 1 < global_paths.size()) {
                    double dx = global_paths[i + 1][0] - global_paths[i][0];
                    double dy = global_paths[i + 1][1] - global_paths[i][1];
                    double path_angle = std::atan2(dy, dx);

                    tf2::Quaternion q;
                    q.setRPY(0, 0, parent_odom.theta_ + path_angle);
                    waypoint.pose.orientation = tf2::toMsg(q);
                } else {
                    waypoint.pose.orientation = start.pose.orientation;
                }

                plan.push_back(waypoint);
            }
        }

        // Always add local_goal as final goal (even if global_paths is empty)
        geometry_msgs::PoseStamped goal;
        goal.header = start.header;

        goal.pose.position.x = parent_odom.x_ + (local_goal[0] * cos_theta - local_goal[1] * sin_theta);
        goal.pose.position.y = parent_odom.y_ + (local_goal[0] * sin_theta + local_goal[1] * cos_theta);

        double goal_angle = std::atan2(local_goal[1], local_goal[0]);
        tf2::Quaternion q_goal;
        q_goal.setRPY(0, 0, parent_odom.theta_ + goal_angle);
        goal.pose.orientation = tf2::toMsg(q_goal);

        plan.push_back(goal);
    }

    void TEB_DDPPlanner::buildObstacles() {
        // Get laser data (in baseline/robot frame)
        // Note: Already filtered in laserScanCallback (valid range + 1cm deduplication)
        auto laser_points = robot->getLaserDataSafe();

        double cos_theta = std::cos(parent_odom.theta_);
        double sin_theta = std::sin(parent_odom.theta_);

        // Only consider obstacles within planning horizon
        static constexpr double MAX_OBSTACLE_DIST = 3.0;  // 3m - beyond this, irrelevant for local planning
        const double max_dist_sq = MAX_OBSTACLE_DIST * MAX_OBSTACLE_DIST;

        for (const auto &point : laser_points) {
            // Additional distance filter for TEB
            double dist_sq = point.x() * point.x() + point.y() * point.y();
            if (dist_sq > max_dist_sq) {
                continue;  // Skip distant obstacles
            }

            // Transform from baseline to odom
            double x_odom = parent_odom.x_ + (point.x() * cos_theta - point.y() * sin_theta);
            double y_odom = parent_odom.y_ + (point.x() * sin_theta + point.y() * cos_theta);

            // Create point obstacle and add to container
            auto obs = boost::make_shared<teb_local_planner::PointObstacle>(x_odom, y_odom);
            obstacles_->push_back(obs);
        }
    }

    void TEB_DDPPlanner::visualizeTEBTrajectory() {
        // Get optimized trajectory from TEB
        std::vector<teb_local_planner::TrajectoryPointMsg> teb_trajectory;
        teb_planner_->getFullTrajectory(teb_trajectory);

        if (teb_trajectory.empty()) {
            return;
        }

        // Convert to PoseState for robot->viewTrajectories
        std::vector<PoseState> trajectory;
        std::vector<double> time_diffs;

        for (const auto &traj_point : teb_trajectory) {
            PoseState state;
            state.x_ = traj_point.pose.position.x;
            state.y_ = traj_point.pose.position.y;

            tf2::Quaternion q;
            tf2::fromMsg(traj_point.pose.orientation, q);
            double roll, pitch, yaw;
            tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
            state.theta_ = yaw;

            state.velocity_ = traj_point.velocity.linear.x;
            state.angular_velocity_ = traj_point.velocity.angular.z;
            state.valid_ = true;
            trajectory.push_back(state);

            time_diffs.push_back(traj_point.time_from_start.toSec());
        }

        if (!trajectory.empty() && !time_diffs.empty()) {
            robot->viewTrajectories(trajectory, std::min(20, static_cast<int>(trajectory.size())), time_diffs);
        }
    }

    void TEB_DDPPlanner::publishCommand(geometry_msgs::Twist &cmd_vel, double linear, double angular) {
        cmd_vel.linear.x = linear;
        cmd_vel.angular.z = angular;
        robot->Control().publish(cmd_vel);
    }

    double TEB_DDPPlanner::normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2 * M_PI;
        while (angle < -M_PI) angle += 2 * M_PI;
        return angle;
    }

}  // namespace Antipatrea
