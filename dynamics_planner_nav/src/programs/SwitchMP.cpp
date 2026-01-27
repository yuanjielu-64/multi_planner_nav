#include "../robot/Jackal.hpp"
#include "../localPlanners/DDP.hpp"
#include "../localPlanners/DWAPlanner.hpp"
#include "../localPlanners/DWA_DDPPlanner.hpp"
#include "../localPlanners/MPPIPlanner.hpp"
#include "../localPlanners/MPPI_DDPPlanner.hpp"
#include "../localPlanners/TEBPlanner.hpp"
#include "../localPlanners/TEB_DDPPlanner.hpp"
#include "../utils/Timer.hpp"
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <memory>

namespace {
    Robot_config::Algorithm parse_algorithm(std::string s, bool &valid) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(::toupper(c)); });
        std::replace(s.begin(), s.end(), '-', '_');

        static const std::unordered_map<std::string, Robot_config::Algorithm> kLut = {
            {"DDP",        Robot_config::DDP},
            {"DWA",        Robot_config::DWA},
            {"DWA_DDP",    Robot_config::DWA_DDP},
            {"MPPI",       Robot_config::MPPI},
            {"MPPI_DDP",   Robot_config::MPPI_DDP},
            {"TEB",        Robot_config::TEB},
            {"TEB_DDP",    Robot_config::TEB_DDP},
        };
        const auto it = kLut.find(s);
        valid = (it != kLut.end());
        return valid ? it->second : Robot_config::DDP;
    }

    // Global state for dynamic planner switching
    Robot_config::Algorithm g_current_algo = Robot_config::DDP;
    bool g_planner_initialized = false;
    bool g_planner_changed = false;

    void plannerSwitchCallback(const std_msgs::String::ConstPtr& msg) {
        bool valid = false;
        auto new_algo = parse_algorithm(msg->data, valid);
        if (valid && (new_algo != g_current_algo || !g_planner_initialized)) {
            g_current_algo = new_algo;
            g_planner_initialized = true;
            g_planner_changed = true;
            ROS_INFO("Received planner switch request: %s -> Algorithm ID: %d", msg->data.c_str(), (int)new_algo);
        }
    }
}

extern "C" int RunMP(int argc, char **argv) {
    // Resolve initial planner selection from CLI or ROS param
    std::string planner_arg;
    for (int i = 1; i + 1 < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--planner" || flag == "-p") { planner_arg = argv[i + 1]; break; }
    }

    // Initialize ROS
    ros::M_string remappings;
    ros::init(remappings, "dynamics_planner_nav");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // Fallback to ROS private param if not set via CLI
    if (planner_arg.empty()) { (void)pnh.getParam("planner", planner_arg); }

    Robot_config robot;

    // Check if planner was specified
    if (!planner_arg.empty()) {
        bool valid = false;
        g_current_algo = parse_algorithm(planner_arg, valid);
        if (valid) {
            g_planner_initialized = true;
            robot.setAlgorithm(g_current_algo);
            ROS_INFO("Initial planner set to: %s", planner_arg.c_str());
        } else {
            ROS_WARN("Invalid planner specified: %s. Waiting for valid planner via /planner_switch topic...", planner_arg.c_str());
        }
    } else {
        ROS_WARN("No planner specified. Waiting for planner via /planner_switch topic...");
    }

    double n = 20;  // 20 Hz
    robot.setDt(1.0/n);

    ros::Rate rate(n);

    // Subscribe to planner switch topic
    ros::Subscriber planner_sub = nh.subscribe("/planner_switch", 10, plannerSwitchCallback);
    ROS_INFO("Subscribed to /planner_switch topic for dynamic planner switching");

    // Instantiate all planners with unique_ptr (lazy initialization)

    std::unique_ptr<Antipatrea::DDP> ddp_planner;

    std::unique_ptr<Antipatrea::DWAPlanner> dwa_planner;
    std::unique_ptr<Antipatrea::DDPDWAPlanner> dwa_ddp_planner;
    std::unique_ptr<Antipatrea::MPPIPlanner> mppi_planner;
    std::unique_ptr<Antipatrea::DDPMPPIPlanner> mppi_ddp_planner;
    std::unique_ptr<Antipatrea::TEBPlanner> teb_planner;
    std::unique_ptr<Antipatrea::TEB_DDPPlanner> teb_ddp_planner;

    std::function<void()> solve_step;

    // Lambda to create/switch planner based on current algorithm
    auto setup_planner = [&]() {
        robot.setAlgorithm(g_current_algo);

        switch (g_current_algo) {
            case Robot_config::DWA: {
                if (!dwa_planner) dwa_planner = std::make_unique<Antipatrea::DWAPlanner>();
                dwa_planner->robot = &robot;
                ROS_INFO("Switched to DWA planner");
                solve_step = [&](){ (void)dwa_planner->Solve(1, robot.getDt(), robot.canBeSolved); };
                break;
            }
            case Robot_config::DWA_DDP: {
                if (!dwa_ddp_planner) dwa_ddp_planner = std::make_unique<Antipatrea::DDPDWAPlanner>();
                dwa_ddp_planner->robot = &robot;
                ROS_INFO("Switched to DWA_DDP planner");
                solve_step = [&](){ (void)dwa_ddp_planner->Solve(1, robot.getDt(), robot.canBeSolved); };
                break;
            }
            case Robot_config::MPPI: {
                if (!mppi_planner) mppi_planner = std::make_unique<Antipatrea::MPPIPlanner>();
                mppi_planner->robot = &robot;
                ROS_INFO("Switched to MPPI planner");
                solve_step = [&](){ (void)mppi_planner->Solve(1, robot.getDt(), robot.canBeSolved); };
                break;
            }
            case Robot_config::MPPI_DDP: {
                if (!mppi_ddp_planner) mppi_ddp_planner = std::make_unique<Antipatrea::DDPMPPIPlanner>();
                mppi_ddp_planner->robot = &robot;
                ROS_INFO("Switched to MPPI_DDP planner");
                solve_step = [&](){ (void)mppi_ddp_planner->Solve(1, robot.getDt(), robot.canBeSolved); };
                break;
            }
            case Robot_config::TEB: {
                if (!teb_planner) teb_planner = std::make_unique<Antipatrea::TEBPlanner>();
                teb_planner->robot = &robot;
                ROS_INFO("Switched to TEB planner");
                solve_step = [&](){ (void)teb_planner->Solve(1, robot.getDt(), robot.canBeSolved); };
                break;
            }
            case Robot_config::TEB_DDP: {
                if (!teb_ddp_planner) teb_ddp_planner = std::make_unique<Antipatrea::TEB_DDPPlanner>();
                teb_ddp_planner->robot = &robot;
                ROS_INFO("Switched to TEB_DDP planner");
                solve_step = [&](){ (void)teb_ddp_planner->Solve(1, robot.getDt(), robot.canBeSolved); };
                break;
            }
            case Robot_config::DDP:
            default: {
                if (!ddp_planner) ddp_planner = std::make_unique<Antipatrea::DDP>();
                ddp_planner->robot = &robot;
                ROS_INFO("Switched to DDP planner");
                solve_step = [&](){ (void)ddp_planner->Solve(1, robot.getDt(), robot.canBeSolved); };
                break;
            }
        }
        g_planner_changed = false;
    };

    // Initialize planner only if one was specified
    if (g_planner_initialized) {
        setup_planner();
    }

    while (ros::ok()) {
        ros::spinOnce();

        // Wait until planner is initialized
        if (!g_planner_initialized) {
            ROS_WARN_THROTTLE(5, "No planner initialized. Waiting for planner specification via /planner_switch topic...");
            rate.sleep();
            continue;
        }

        // Check if planner needs to be switched
        if (g_planner_changed) {
            setup_planner();
        }

        if (!robot.setup()) {
            if (robot.getRobotState() == Robot_config::BRAKE_PLANNING)
                rate.sleep();
            continue;
        }

        // Call selected planner solve function
        solve_step();
    }

    return 0;
}

int main(int argc, char **argv) {
    return RunMP(argc, argv);
}
