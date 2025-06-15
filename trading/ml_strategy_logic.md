           +---------------------------+
           |    LSTM + Feature Inputs  |
           +---------------------------+
                         |
                         v
               +-------------------+
               |  XGBoost Signal   |
               +-------------------+
                         |
                         v
     +--------------------------------------+
     |  RL Agent (DQN / PPO via SB3)        |
     | - Takes in market state + signal     |
     | - Outputs: stake %, skip, or confirm |
     +--------------------------------------+
                         |
                         v
            +----------------------------+
            |    Broker Execution Logic  |
            +----------------------------+
