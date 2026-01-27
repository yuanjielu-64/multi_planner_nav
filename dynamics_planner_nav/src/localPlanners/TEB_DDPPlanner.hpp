/*
 * TEB_DDPPlanner.hpp — Hybrid planner combining TEB and DDP
 * Uses TEB for local planning with DDP as fallback/refinement
 */

#ifndef Antipatrea__TEB_DDPPlanner_HPP_
#define Antipatrea__TEB_DDPPlanner_HPP_

#include "../robot/Jackal.hpp"

// Official TEB planner headers
#include <teb_local_planner/optimal_planner.h>
#include <teb_local_planner/teb_config.h>
#include <teb_local_planner/obstacles.h>
#include <teb_local_planner/visualization.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <memory>

namespace Antipatrea {
    using PoseState = Robot_config::PoseState;

    /**
     * @brief Hybrid planner combining TEB and DDP
     *
     * This class uses TEB for primary local planning and DDP as a fallback
     * or refinement mechanism when TEB fails or needs assistance.
     */
    class TEB_DDPPlanner {
    public:
        TEB_DDPPlanner();
        ~TEB_DDPPlanner() = default;

        bool Solve(int nrIters, double dt, bool &canBeSolved);

        Robot_config *robot = nullptr;

    protected:
        void updateRobotState();

        void commonParameters(Robot_config &robot);
        void normalParameters(Robot_config &robot);
        void lowSpeedParameters(Robot_config &robot);

        bool handleNoMapPlanning(geometry_msgs::Twist &cmd_vel);
        bool handleNormalSpeedPlanning(geometry_msgs::Twist &cmd_vel, double dt);
        bool handleLowSpeedPlanning(geometry_msgs::Twist &cmd_vel, double dt);
        bool handleAbnormalPlaning(geometry_msgs::Twist &cmd_vel, double dt);

        void publishCommand(geometry_msgs::Twist &cmd_vel, double linear, double angular);

        /**
         * @brief Call the official TEB planner
         */
        bool callTEBPlanner(geometry_msgs::Twist &cmd_vel);

        /**
         * @brief Call DDP planner as fallback
         */
        bool callDDPPlanner(geometry_msgs::Twist &cmd_vel, double dt);

        /**
         * @brief Build global plan from local_goal for TEB
         */
        void buildGlobalPlan(std::vector<geometry_msgs::PoseStamped> &plan);

        /**
         * @brief Convert laser data to TEB obstacles
         */
        void buildObstacles();

        /**
         * @brief Visualize TEB trajectory
         */
        void visualizeTEBTrajectory();

        /**
         * @brief Normalize angle to [-π, π]
         */
        double normalizeAngle(double angle);

        // Official TEB planner instance
        boost::shared_ptr<teb_local_planner::TebOptimalPlanner> teb_planner_;

        // TEB configuration
        teb_local_planner::TebConfig teb_config_;

        // Obstacle container
        boost::shared_ptr<teb_local_planner::ObstContainer> obstacles_;

        // Robot footprint model
        teb_local_planner::RobotFootprintModelPtr robot_model_;

        // State tracking
        PoseState parent;
        PoseState parent_odom;
        std::vector<double> local_goal;
        std::vector<std::vector<double>> global_paths;

        // Hybrid control tracking
        int teb_failure_count_;
        bool use_ddp_fallback_;
    };
}

#endif
