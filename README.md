# Motivation

Overfitting is the enemy of many machine learning projects and many financial models. The risk of overfitting, however fatal, comes with a silver lining in trading. This is because in trading, there are asymmetric payoffs to strategy identification. Namely, finding a good strategy is much more valuable than finding a strategy that you think is good but is actually bad. Given the choice, you would prefer to have 3 good strategies and 7 bad ones (and not know which are which) than to have just 1 good trading strategy that you know is good. This is because as time passes, you can allocate capital to the strategies that are performing well and away from the others...and now you've got 3 good strategies instead of just one.

With this in mind, we can try to "throw the kitchen sink" of machine learning at multiple bundles of assets, multiple neural network architectures, and multiple network updating strategies. Normally, this scattershot approach would be a disastrous recipe for overfitting. But, due to the asymmetry discussed above, we can afford merely to manage this risk and hope for the best, rather than letting the risk of overfitting completely overwhelm our efforts. 

What follows from this motivation is an autoML framework for identifying, training, and evaluating reinforcement strategies.

# Running the code
After installing the dependencies, I can start automatically training strategies by running `main.py`. I have the locations of required directories saved in `global_vars.py`. 

# Under the hood
##### Setting global variables
`global_vars.py` determines whether we are running on my local computer or on an Intel DevCloud (where most of the computation took place in practice) and sets some global variables accordingly.

##### Reading initial work
`main.py` reads in the results of a correlation study previously performed on many thousands of pairs of assets, and identifies asset bundles where the primary asset is correlated enough (but not *too* correlated) to some number of other assets.

##### Training and iterating the models
Passing these bundles into `train.py`, `train.py` gets to work on the autoML. It prepares the data and checks any existing model progress. If the current model doesn't look promising, it moves on to next model. If we've run through all of the models, we move on to the next bundle. 

##### Preparing the data
The preparation of data is performed in `data.py`, and consists of reading raw 1-second OHLCV tick data, normalizing it so it is relatively stationary, and calculating some second-order statistics like change on X days, the ratio of open prices of the other assets in relation to the primary asset, the EMA and sum of OHLC across some number of minutes. TickImbalance, a popular signal in trading, is also calculated across some number of minutes. We also split the data into train and validation sets, making sure to have no overlap of any features and also leaving off data from the end of the validation set so we can use that data cleanly if we wish to further evaluate a strategy.

##### Types of RL models
The models are reinforcement learning models. Their possible architectures are defined in `global_vars.py` and their possible update-rules are 
- DQN (normal DQN)
- Target DQN (target network weights are fixed for some time increment)
- Double DQN (action is taken with a policy network, Q Value is calculated using a separate network).  

##### Model/bundle evaluation
As `train.py` trains and iterates our models, it also takes momentary breaks to evaluate the models. Here, we set the probability of taking a random action to 0, and we do not update any network weights.

At the end of every episode, both training and evaluatory, or even if our episode is interrupted, we save our results in `save_results.py`. `save_results.py` saves all of the hard summary data on the episode, but it also creates human-readable graphs and formats all of the details of the model into a nice HTML file. This way, if a human wants to understand the state and progress of a model, we can look at the HTML file to quickly understand. Here is an example of some of the outputs of these files: 

Trade/Pnl visualization: all episode, the worst segment of the episode, and the best segment of the episode.

[![N|Solid](https://lh3.googleusercontent.com/gX-D6C7Hjf4G3O4ZaVXF6WSBdl5WXEIoDwpfRwuDSYfa1vJuKxgfBnRveCF4h_72FGL6_irrHeyLRThA4VRoD0EzI2ZJ_IokYpIzWeAE_WURQ4cqaec9m68-ajDJpIOxRAsbWmxxzQ=w2400)](https://lh3.googleusercontent.com/gX-D6C7Hjf4G3O4ZaVXF6WSBdl5WXEIoDwpfRwuDSYfa1vJuKxgfBnRveCF4h_72FGL6_irrHeyLRThA4VRoD0EzI2ZJ_IokYpIzWeAE_WURQ4cqaec9m68-ajDJpIOxRAsbWmxxzQ=w2400)

Summary of key episodes, plus visualization for how model is progressing over time.
[![N|Solid](https://lh3.googleusercontent.com/aYkRNgdw-V0ZLGFZN3-NfvqM_t5CIKGTh7VYakhbIwGS5Xezt28fihAihZkJax0MMDKRjgc-DaLu085ta_G0wenqpfeDDlU8h97R9MyW9kDd2cDSKpanp47hCaKTADQouMvskaWJwA=w2400)](https://lh3.googleusercontent.com/aYkRNgdw-V0ZLGFZN3-NfvqM_t5CIKGTh7VYakhbIwGS5Xezt28fihAihZkJax0MMDKRjgc-DaLu085ta_G0wenqpfeDDlU8h97R9MyW9kDd2cDSKpanp47hCaKTADQouMvskaWJwA=w2400)

Visualization for how likely the agent is to cross large bid-ask spreads
[![N|Solid](https://lh3.googleusercontent.com/Qskg-v_0HT98bPKndTTaUKlDurgmOaE1QpLBk9D8DU3KwOOg_y1T_eAI51ibmC9w5OLrwK3Tta5jBAit_-QZDZHf6oWDFcRmbSdoHTPfFNZvGwm4gP3FkCoi3Oalu4_XrBWewQ7NXg=w2400)](https://lh3.googleusercontent.com/Qskg-v_0HT98bPKndTTaUKlDurgmOaE1QpLBk9D8DU3KwOOg_y1T_eAI51ibmC9w5OLrwK3Tta5jBAit_-QZDZHf6oWDFcRmbSdoHTPfFNZvGwm4gP3FkCoi3Oalu4_XrBWewQ7NXg=w2400)

Percent of timesteps spent flat, long, and short, plus sharpe on the right.
[![N|Solid](https://lh3.googleusercontent.com/aSbvRr_aII_IbRTuFS_vaEFRaz-iuNUcfb7RpLXUS_4Zn8zunMchFlT65FWCSLEB8FlRKbTjYpgM7n3CuYz3m1xdi5bMw9eQ4gJO3JoC0KJNaxwsWZYxnoDRYGfZyDL2ZoEC5j5kPA=w2400)](https://lh3.googleusercontent.com/aSbvRr_aII_IbRTuFS_vaEFRaz-iuNUcfb7RpLXUS_4Zn8zunMchFlT65FWCSLEB8FlRKbTjYpgM7n3CuYz3m1xdi5bMw9eQ4gJO3JoC0KJNaxwsWZYxnoDRYGfZyDL2ZoEC5j5kPA=w2400)

# Results
There were no strategies which are promising enough to warrant excitement. Whenever a strategy generates a positive Sharpe, it is clearly unstable (subsequent episodes of training do not replicate well). However, there is a very pessimistic execution assumption underlying this study, namely that each trade must completely cross the bid-ask spread, and the only possible positions are long one contract, short one contract, or flat. These must be relaxed for more promising results to surface, but relaxing these assumptions opens up a different world of uncertainty.

Please feel free to reach out to andrew.hochstadt@gmail.com if you have any further questions.
