from gym.envs.registration import register

register(
    id="dwa_param-v0",
    entry_point="envs.DWA.parameter_tuning_envs:DWAParamContinuousLaser"
)

register(
    id="teb_param-v0",
    entry_point="envs.Teb.parameter_tuning_envs:TebParamContinuousLaser"
)

register(
    id="mppi_param-v0",
    entry_point="envs.MPPI.parameter_tuning_envs:MPPIParamContinuousLaser"
)

register(
    id="ddp_param-v0",
    entry_point="envs.DDP.parameter_tuning_envs:DDPParamContinuousLaser"
)