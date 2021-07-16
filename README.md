1.Requirement

python 3.5.0
tensorflow 1.8.0
numpy 1.14.5
gym 0.16.0
pyflann 1.6.14

2.Setup

This folder provides the interface of our method, you can edit it to apply to different CPS system. You change the dimensionality of action and state, with corresponding reward. You choose the type of action by the parameter "Discrete", and the parameter "RND" decides whether to use the RND mechanism.

3.Code Structure

The folder src/ddpg contains the implement of the DDPG agent, and the src/util folder implements the RND and results storage. The src/wolp_agent.py file implements the agent of our method for discrete action. The cps_env.py is the interface to CPS system, and the generate.py file is the start of our method.

4.Usage

Edit and run "generate.py" to start our method and generate the failure-inducing input for target CPS system.
