from global_vars import omni_dir, tdm_dir, day_chg_incs, minute_incs
from utils import getChiTimeNow, tqdm
from methods import calcOpenPnl
from data import locateConfigDir, processData, DataObj
from models import ModelConfig
from agent import Agent
from save_results import saveProgressSummary, updateSummarySheet

import os
import pandas as pd
import numpy as np

def endEpisode(agent, dataObj, batch_size, epsilon_start, avg_loss_array, sess_type, start_time):
    agent.saveTradeLog(sess_type)

#         if agent.episode % 10 == 0:
    if agent.episode % 1 == 0 and sess_type == 'train':
        agent.saveModelWeights()

    epsilon_end = agent.epsilon if sess_type == 'train' else np.nan
    end_time = getChiTimeNow()

    saveProgressSummary(agent, dataObj, batch_size, epsilon_start, epsilon_end, avg_loss_array, \
                        sess_type, start_time, end_time)
    agent.episode += 1
    updateSummarySheet(agent.config_dir)

def train_model(agent, dataObj, ep_count=20, batch_size=32, evaluate_every=5):
    start_ep = agent.episode+0
    end_ep = start_ep + ep_count
    for i in range(ep_count):
        if (i+1) % evaluate_every == 0:
            evaluate_model(agent, dataObj, batch_size=batch_size)


        start_time = getChiTimeNow()
        epsilon_start = agent.epsilon
        total_profit = 0
        data_length = len(dataObj.trainDF) - 1
        avg_loss_array = []
        agent.reset()

        state = agent.getState(dataObj, 0)
        try:
            for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(agent.episode, end_ep)):
                reward = 0

                # select an action
                action = agent.act(state)

                open_pnl = calcOpenPnl(agent, dataObj, t+1)
                day_pnl = evaluateAction(action, agent, dataObj, t+1)
                reward = open_pnl + day_pnl
                total_profit += reward

                next_state = agent.getState(dataObj, t+1)
                done = (t == data_length - 1)
                agent.remember(state, action, reward, next_state, done)

                if len(agent.memory) > batch_size:
                    # train every batch_size
                    if t % batch_size == 0:
                        loss = agent.train_experience_replay(batch_size)
                        avg_loss_array.append(loss)

                state = next_state

            endEpisode(agent, dataObj, batch_size, epsilon_start, avg_loss_array, \
                                'train', start_time)
        except (KeyboardInterrupt, SystemExit):
            print('KeyboardInterrupt or SystemExit. Ending current episode.')
            endEpisode(agent, dataObj, batch_size, epsilon_start, avg_loss_array, \
                                'train', start_time)
            raise
        except:
            print('Unknown error...Ending current episode.')
            endEpisode(agent, dataObj, batch_size, epsilon_start, avg_loss_array, \
                                'train', start_time)
            raise

def evaluate_model(agent, dataObj, debug=False, batch_size=32):
    print('Evaluating Model')
    start_time = getChiTimeNow()
    epsilon_start = np.nan
    total_profit = 0
    data_length = len(dataObj.trainDF) - 1
    avg_loss_array = []
    agent.reset()

    state = agent.getState(dataObj, 0)

    try:
        for t in tqdm(range(data_length)):
            reward = 0
            # select an action
            action = agent.act(state, is_eval=True)

            open_pnl = calcOpenPnl(agent, dataObj, t+1)
            day_pnl = evaluateAction(action, agent, dataObj, t+1)
            reward = open_pnl + day_pnl
            total_profit += reward

            next_state = agent.getState(dataObj, t+1)

            done = (t == data_length - 1)

    #         agent.memory.append((state, action, reward, next_state, done)) # don't know why this line was here instead of the below
            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > batch_size:
                # train every batch_size
                if t % batch_size == 0:
                    loss = agent.evaluate_experience_replay(batch_size)
                    avg_loss_array.append(loss)

            state = next_state
        endEpisode(agent, dataObj, batch_size, epsilon_start, avg_loss_array, \
                   'eval', start_time)
    except (KeyboardInterrupt, SystemExit):
        print('KeyboardInterrupt or SystemExit. Ending current episode.')
        endEpisode(agent, dataObj, batch_size, epsilon_start, avg_loss_array, \
                            'eval', start_time)
        raise
    except:
        print('Unknown error...Ending current episode.')
        endEpisode(agent, dataObj, batch_size, epsilon_start, avg_loss_array, \
                            'eval', start_time)
        raise

def modelHasPromise(model_str, config_dir, exhaustiveness_level):
    print('Determining if model has promise:',model_str)
    all_progress_summary_fn = config_dir + 'Deep_Models/all_progress_summary.csv'
    if not os.path.exists(all_progress_summary_fn):
        print('all_progress_summary doesnt exist. Still has promise.')
        return True

    all_progress_summaryDF = pd.read_csv(all_progress_summary_fn)
    progress_subDF = all_progress_summaryDF.loc[all_progress_summaryDF.Model == model_str]
    progress_subDF = progress_subDF.loc[progress_subDF.SessType == 'train']
    progress_subDF = progress_subDF.loc[progress_subDF.TsSeen >= 10000-2]
    progress_subDF.reset_index(inplace=True, drop=True)
    if len(progress_subDF) < 10:
        print("We've only had", len(progress_subDF), 'sufficient episodes. Still has promise.')
        return True

    min_loss_updates = 1000*exhaustiveness_level
    if progress_subDF.LossUpd.sum() < min_loss_updates:
        print('We need', min_loss_updates-progress_subDF.LossUpd.sum(),'/', min_loss_updates, 'more loss updates for this exhaustiveness level.')
        return True

    min_target_updates = round(min_loss_updates*.5)
    if progress_subDF.TargetUpd.sum() < min_target_updates:
        print('We need', min_target_updates-progress_subDF.TargetUpd.sum(),'/', min_target_updates, 'more target updates for this exhaustiveness level.')
        return True

    # check if the trajectory is improving. Make sure to divide loss and pnl ts_seen
    loss_tolerance_pct = .02*exhaustiveness_level
    pnl_tolerance_pct = .02*exhaustiveness_level

    # check if the best of the past 3 trials is at least <tolerance> better than the avg of (t-5, t-10)
    best_recent_loss = (progress_subDF.iloc[-3:].AvgLoss/progress_subDF.iloc[-3:].TsSeen).min()
    best_recent_pnl = (progress_subDF.iloc[-3:].PNL/progress_subDF.iloc[-3:].TsSeen).max()
    avg_prev_loss = (progress_subDF.iloc[-10:-5].AvgLoss/progress_subDF.iloc[-10:-5].TsSeen).mean()
    avg_prev_pnl = (progress_subDF.iloc[-10:-5].PNL/progress_subDF.iloc[-10:-5].TsSeen).mean()
    loss_improvement = (avg_prev_loss - best_recent_loss)/avg_prev_loss
    pnl_improvement = (best_recent_pnl - avg_prev_pnl)/avg_prev_pnl
    if pnl_improvement < pnl_tolerance_pct:
        print('PNL improvement is only ',round(pnl_improvement*100, 3),'%. Does not qualify as promising.')
        return False

    if loss_improvement < loss_tolerance_pct:
        print('Loss improvement is only ',round(loss_improvement*100, 3),'%. Does not qualify as promising.')
        return False
    print('PNL and Loss Improvement checks passed with values', round(pnl_improvement*100, 3),
          '% and', round(loss_improvement*100, 3),'%. Still has promise.')
    return True

def chooseStrategy(strategies, config_dir, model_str):
    """Randomly chooses a strategy, but gives less probability to recent strategies."""
    all_progress_summary_fn = config_dir+'Deep_Models/all_progress_summary.csv'
    if not os.path.exists(all_progress_summary_fn):
        print(all_progress_summary_fn, 'doesnt exist. Returning random strategy.')
        return np.random.choice(strategies)
    all_progress_summaryDF = pd.read_csv(all_progress_summary_fn)
    model_subDF = all_progress_summaryDF.loc[(all_progress_summaryDF.Model == model_str) & (all_progress_summaryDF.SessType == 'train')]
    if len(model_subDF) == 0:
        print(model_str, 'Has no training sessions yet. Returning random strategy.')
        return np.random.choice(strategies)
    # make probabilities reverse-proportional to their TsSeen cum_sums X their TsSeen, effectively downweighting recent ts
    proportions = []
    for strategy in strategies:
        stratDF = model_subDF.loc[model_subDF.Strat == strategy]
        proportions.append((stratDF.TsSeen.cumsum() * stratDF.TsSeen).sum())
    proportions = np.nan_to_num(proportions)+.01 # we don't want to divide by 0
    probs = 1/np.array(proportions)
    probs = probs/np.sum(probs) #normalize to 1
    print('Choosing', strategies, 'with the following probabilities:', [round(p, 3) for p in probs])
    return np.random.choice(strategies, p=probs)

# def get_state(agent, dataObj, t):
#     """Returns the position and the t-th row of the data"""
#     ret_array = [agent.position] + list(dataObj.trainDF.iloc[t].values)
#     ret_array = np.reshape(ret_array, len(ret_array))
#     return(np.array([ret_array]))

def evaluateAction(action, agent, dataObj, t):
    """returns day_pnl and updates agent's position and trade_log"""
    close_prices = dataObj.train_close_pricesDF[['C_B1', 'C_A1']].iloc[t]
    t_midpt = sum(close_prices.values)/2
    minute = dataObj.train_minutes.iloc[t]

    actual_action = np.nan
    new_position = agent.position
    day_pnl = 0

    # BUY
    if action == 1:
        if agent.position != 1: #actually take the action
            actual_action = 1
            day_pnl = t_midpt - close_prices['C_A1']
            new_position = agent.position + 1
        else:
            actual_action = 0

    # SELL
    elif action == 2:
        if agent.position != -1: #actually take the action
            actual_action = 2
            day_pnl = close_prices['C_B1'] - t_midpt
            new_position = agent.position - 1
        else:
            actual_action = 0

    # HOLD
    elif action == 0:
        actual_action = 0

    else:
        raise ValueError('Action '+str(action)+' not recognized.')

    agent.position = new_position
    agent.updateTradeLog(minute, action, actual_action, new_position, close_prices)
    return day_pnl

def autoTrain(sec, combDF, keras_models, omni_dir=omni_dir, minute_incs=minute_incs, day_chg_incs=day_chg_incs,
              nrows=None, exhaustiveness_level=2, email_progress=False):
    """
    1. Locates or prepares config_dir
    2. Locates or prepares postprepared data
    3. Check model progress. If model doesn't look promising, move to next model.
            If model still has promise, keep training it
    4. Optionally emails results
    """
    assert exhaustiveness_level in [1,2,3,4,5]

    comb_row = combDF.loc[combDF.Sec1 == sec].iloc[0]
    config_dir = locateConfigDir(comb_row, omni_dir, day_chg_incs, minute_incs)
    train_close_pricesDF, train_minutesDF, val_close_pricesDF, val_minutesDF = \
            processData(comb_row, config_dir, day_chg_incs, minute_incs)
    dataObj = DataObj(config_dir, train_close_pricesDF, train_minutesDF, val_close_pricesDF, val_minutesDF, nrows=nrows)
    # [x] Check model progress
    for model_array in keras_models:
        model_cfg = ModelConfig(dataObj.getStateSize(), model_array=model_array)
        model_has_promise = modelHasPromise(model_cfg.model_str, config_dir, exhaustiveness_level)
        while model_has_promise:
#             strategy = np.random.choice(['dqn', 't-dqn', 'double-dqn'])
            strategy = chooseStrategy(['dqn', 't-dqn', 'double-dqn'], config_dir, model_cfg.model_str)
            print('\n\n', '='*40, '\nUsing strategy =', strategy)
            agent = Agent(model_cfg, config_dir, strategy=strategy)
            agent.setQPremiums(dataObj=dataObj)
            train_model(agent, dataObj, ep_count=10, evaluate_every=5)
            model_has_promise = modelHasPromise(model_cfg.model_str, config_dir, exhaustiveness_level)
        else:
            print('\n', '='*50, '\nModel', model_cfg.model_str,'does not have promise. Moving on.\n', '='*50, '\n\n')
